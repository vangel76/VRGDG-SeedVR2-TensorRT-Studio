"""Decode a full SeedVR2 latent using an overlapping TensorRT tile engine."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("latent", type=Path)
    parser.add_argument("--tile-latent", type=int, default=64)
    parser.add_argument("--overlap-latent", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=3)
    args = parser.parse_args()
    payload = torch.load(args.latent, map_location="cpu", weights_only=False)
    source = payload["latent"].to(device="cuda", dtype=torch.float16).contiguous()
    _, channels, frames, height, width = source.shape
    tile = args.tile_latent
    overlap = args.overlap_latent
    if overlap <= 0 or overlap >= tile:
        raise ValueError("overlap-latent must be between 1 and tile-latent-1")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize TensorRT-RTX engine")
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    expected = tuple(engine.get_tensor_shape(input_name))
    if expected != (1, channels, frames, tile, tile):
        raise ValueError(f"Engine input {expected} does not match tile {(1, channels, frames, tile, tile)}")

    out_h, out_w = height * 8, width * 8
    out_tile = tile * 8
    out_overlap = overlap * 8
    ys, xs = positions(height, tile, overlap), positions(width, tile, overlap)
    padded_h, padded_w = max(height, ys[-1] + tile), max(width, xs[-1] + tile)
    latent = torch.nn.functional.pad(source, (0, padded_w - width, 0, padded_h - height))
    result = torch.zeros((1, 3, frames * 4 - 3, out_h, out_w), device="cuda", dtype=torch.float32)
    weights = torch.zeros_like(result)
    stream_count = max(1, min(args.streams, len(ys) * len(xs)))
    # A context's tensor addresses are mutable. For the safe/default path we
    # execute one tile at a time so addresses cannot be overwritten by a later
    # queued tile (which previously left the canvas gray except for the last
    # tile). Multi-stream execution can be reintroduced with a context pool and
    # per-context locking once correctness is established.
    stream_count = 1
    contexts = [engine.create_execution_context()]
    streams = [torch.cuda.Stream() for _ in range(stream_count)]
    jobs = []
    for tile_index, (y, x) in enumerate(( (y, x) for y in ys for x in xs )):
        tile_input = latent[:, :, :, y:y + tile, x:x + tile].contiguous()
        tile_output = torch.empty((1, 3, frames * 4 - 3, out_tile, out_tile), device="cuda", dtype=torch.float16)
        context = contexts[0]
        stream = streams[tile_index % stream_count]
        context.set_tensor_address(input_name, tile_input.data_ptr())
        context.set_tensor_address(output_name, tile_output.data_ptr())
        jobs.append((y, x, tile_input, tile_output, context, stream))

    def launch(job):
        y, x, tile_input, tile_output, context, stream = job
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed at tile y={y}, x={x}")

    started = time.perf_counter()
    for y, x, tile_input, tile_output, context, stream in jobs:
        context.set_tensor_address(input_name, tile_input.data_ptr())
        context.set_tensor_address(output_name, tile_output.data_ptr())
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed at tile y={y}, x={x}")
        stream.synchronize()
    for y, x, tile_input, tile_output, context, stream in jobs:
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    decoded = result[:, :, :, :out_h, :out_w].cpu()
    original_frames = payload.get("original_frames")
    torch.save({
        "video": decoded,
        "original_frames": int(original_frames) if original_frames is not None else None,
        "source_latent": str(args.latent),
    }, args.output)
    print(f"Latent: {args.latent} {tuple(source.shape)}")
    print(f"Tiles: {len(ys) * len(xs)} ({ys} × {xs}), overlap={overlap} latent px")
    print(f"Output: {(1, 3, frames * 4 - 3, out_h, out_w)}")
    print(f"TensorRT tiled GPU decode: {elapsed * 1000:.2f} ms")
    print(f"Output range: {result.min().item():.6f} .. {result.max().item():.6f}")
    if original_frames is not None:
        print(f"Original frames: {int(original_frames)} (padded decode frames retained for context)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
