"""Decode multiple SeedVR2 latent batches with one TensorRT engine/context."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import tensorrt_rtx as trt


def positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def feather(length: int, overlap: int, left: bool, right: bool, device: torch.device) -> torch.Tensor:
    weight = torch.ones(length, device=device, dtype=torch.float32)
    if left and overlap:
        weight[:overlap] = torch.linspace(0.0, 1.0, overlap + 1, device=device)[1:]
    if right and overlap:
        weight[-overlap:] = torch.minimum(
            weight[-overlap:], torch.linspace(1.0, 0.0, overlap + 1, device=device)[1:]
        )
    return weight


class PersistentDecoder:
    def __init__(self, engine_path: Path, channels: int, frames: int, tile: int) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT-RTX engine: {engine_path}")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.output_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        expected = tuple(self.engine.get_tensor_shape(self.input_name))
        requested = (1, channels, frames, tile, tile)
        if expected != requested:
            raise ValueError(f"Engine input {expected} does not match tile {requested}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Could not create TensorRT execution context: {engine_path}")
        self.stream = torch.cuda.Stream()

    def decode(self, latent_path: Path, output_path: Path, tile: int, overlap: int) -> float:
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        source = payload["latent"].to(device="cuda", dtype=torch.float16).contiguous()
        _, _channels, frames, height, width = source.shape
        if overlap <= 0 or overlap >= tile:
            raise ValueError("overlap_latent must be between 1 and tile_latent-1")
        out_h, out_w = height * 8, width * 8
        out_tile, out_overlap = tile * 8, overlap * 8
        ys, xs = positions(height, tile, overlap), positions(width, tile, overlap)
        padded_h, padded_w = max(height, ys[-1] + tile), max(width, xs[-1] + tile)
        latent = torch.nn.functional.pad(source, (0, padded_w - width, 0, padded_h - height))
        output_frames = frames * 4 - 3
        result = torch.zeros((1, 3, output_frames, out_h, out_w), device="cuda", dtype=torch.float32)
        weights = torch.zeros_like(result)
        jobs: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        for y in ys:
            for x in xs:
                tile_input = latent[:, :, :, y:y + tile, x:x + tile].contiguous()
                tile_output = torch.empty((1, 3, output_frames, out_tile, out_tile), device="cuda", dtype=torch.float16)
                jobs.append((y, x, tile_input, tile_output))
        started = time.perf_counter()
        with torch.cuda.stream(self.stream):
            for y, x, tile_input, tile_output in jobs:
                self.context.set_tensor_address(self.input_name, tile_input.data_ptr())
                self.context.set_tensor_address(self.output_name, tile_output.data_ptr())
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError(f"TensorRT execution failed at tile y={y}, x={x}")
                self.stream.synchronize()
        for y, x, _tile_input, tile_output in jobs:
            wy = feather(out_tile, out_overlap, y != ys[0], y != ys[-1], tile_output.device)
            wx = feather(out_tile, out_overlap, x != xs[0], x != xs[-1], tile_output.device)
            window = (wy[:, None] * wx[None, :]).view(1, 1, 1, out_tile, out_tile)
            oy, ox = y * 8, x * 8
            valid_h, valid_w = min(out_tile, out_h - oy), min(out_tile, out_w - ox)
            window = window[..., :valid_h, :valid_w]
            result[..., oy:oy + valid_h, ox:ox + valid_w] += tile_output[..., :valid_h, :valid_w].float() * window
            weights[..., oy:oy + valid_h, ox:ox + valid_w] += window
        elapsed = time.perf_counter() - started
        result = (result / weights.clamp_min(1e-6)).clamp(-2.0, 2.0)
        decoded = result[:, :, :, :out_h, :out_w].cpu()
        original_frames = payload.get("original_frames")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "video": decoded,
            "original_frames": int(original_frames) if original_frames is not None else None,
            "source_latent": str(latent_path),
        }, output_path)
        del payload, source, latent, result, weights, decoded, jobs
        gc.collect()
        torch.cuda.empty_cache()
        return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs") or []
    if not jobs:
        raise ValueError("Persistent decoder manifest contains no jobs")
    tile = int(manifest["tile_latent"])
    overlap = int(manifest["overlap_latent"])
    first_payload = torch.load(Path(jobs[0]["latent"]), map_location="cpu", weights_only=False)
    first_shape = tuple(first_payload["latent"].shape)
    del first_payload
    decoder = PersistentDecoder(Path(manifest["engine"]), first_shape[1], first_shape[2], tile)
    print(f"Persistent TensorRT decoder ready: {len(jobs)} batches, one engine/context", flush=True)
    total_started = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        elapsed = decoder.decode(Path(job["latent"]), Path(job["output"]), tile, overlap)
        print(f"PERSISTENT_PROGRESS {index}/{len(jobs)} {elapsed:.3f}", flush=True)
    print(f"Persistent TensorRT decode complete in {time.perf_counter() - total_started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
