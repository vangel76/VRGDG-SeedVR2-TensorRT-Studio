"""Upscale a video with NVIDIA RTX Video Super Resolution (Video Effects SDK via the nvidia-vfx wheel).

Frames stream from ffmpeg (rawvideo RGB) through `nvvfx.VideoSuperRes` on the GPU and either go
straight into an ffmpeg H.264 encoder (fast path) or into 21-frame `decoded_*.pt` batches in the
same layout the TensorRT decoder produces, so the Studio post stage (color correction, faces,
skin, grain) and post-only reprocess work unchanged.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import torch

QUALITIES = ("BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA")
MAX_SCALE = 4.0


def _ffmpeg() -> str:
    try:
        from seedvr_studio.media import _tool
        return _tool("ffmpeg")
    except Exception:
        return "ffmpeg"


def even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def plan_passes(src_w: int, src_h: int, dst_w: int, dst_h: int) -> list[tuple[int, int]]:
    """VSR runs up to 4x per pass; larger factors chain passes (each even-sized)."""
    scale = max(dst_w / src_w, dst_h / src_h)
    if scale <= MAX_SCALE:
        return [(dst_w, dst_h)]
    passes: list[tuple[int, int]] = []
    w, h = src_w, src_h
    while max(dst_w / w, dst_h / h) > MAX_SCALE:
        w, h = even(w * MAX_SCALE), even(h * MAX_SCALE)
        passes.append((w, h))
    passes.append((dst_w, dst_h))
    return passes


def read_exact(stream, size: int) -> bytes:
    """Pipes may return short reads; collect exactly `size` bytes or whatever remains at EOF."""
    parts: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def frame_count(source: Path, fps: float) -> int:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets,nb_frames,duration",
                            "-of", "default=nw=1", str(source)], capture_output=True, text=True)
    values: dict[str, str] = {}
    for line in probe.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip()
    for key in ("nb_read_packets", "nb_frames"):
        if values.get(key, "").isdigit() and int(values[key]) > 0:
            return int(values[key])
    try:
        return max(1, int(round(float(values.get("duration", "0")) * fps)))
    except ValueError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, help="MP4 to write directly (fast path)")
    parser.add_argument("--batches-dir", type=Path, help="Write decoded_NNN.pt batches here instead of an MP4")
    parser.add_argument("--batch-frames", type=int, default=21)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--src-width", type=int, required=True)
    parser.add_argument("--src-height", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--quality", choices=QUALITIES, default="ULTRA")
    parser.add_argument("--frames", type=int, default=0, help="Total frames (for progress); probed when 0")
    args = parser.parse_args()
    if not args.output and not args.batches_dir:
        raise SystemExit("Give --output or --batches-dir")

    import nvvfx

    total = args.frames or frame_count(args.source, args.fps)
    passes = plan_passes(args.src_width, args.src_height, args.width, args.height)
    print(f"RTX Video Super Resolution {args.quality}: {args.src_width}x{args.src_height} -> {args.width}x{args.height} in {len(passes)} pass(es) {passes}", flush=True)
    quality = getattr(nvvfx.effects.QualityLevel, args.quality)
    effects = []
    for out_w, out_h in passes:
        effect = nvvfx.VideoSuperRes(quality)
        effect.output_width, effect.output_height = out_w, out_h
        effect.load()
        effects.append(effect)

    decode = subprocess.Popen([_ffmpeg(), "-v", "error", "-i", str(args.source), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                              stdout=subprocess.PIPE, bufsize=0)
    assert decode.stdout is not None
    frame_bytes = args.src_width * args.src_height * 3
    encode = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_name(args.output.stem + "_noaudio.mp4")
        encode = subprocess.Popen([_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{args.width}x{args.height}", "-r", f"{args.fps:.9g}",
                                   "-i", "pipe:0", "-c:v", "libx264", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)],
                                  stdin=subprocess.PIPE, bufsize=0)
        assert encode.stdin is not None
    batches_written = 0
    pending: list[torch.Tensor] = []

    def flush_batch(final: bool = False) -> None:
        nonlocal batches_written, pending
        if not pending or (not final and len(pending) < args.batch_frames):
            return
        frames = torch.stack(pending[: args.batch_frames], dim=1).unsqueeze(0)  # (1, 3, T, H, W) in [0, 1]
        real = frames.shape[2]
        if real < args.batch_frames:  # pad like the TensorRT decoder so the post stage sees uniform batches
            frames = torch.cat([frames, frames[:, :, -1:].expand(-1, -1, args.batch_frames - real, -1, -1)], dim=2)
        batches_written += 1
        args.batches_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"video": (frames * 2.0 - 1.0).to(torch.float16).cpu(), "original_frames": real, "source": "rtx_vsr"},
                   args.batches_dir / f"decoded_{batches_written:03d}.pt")
        pending = pending[args.batch_frames:]

    done = 0
    try:
        while True:
            chunk = read_exact(decode.stdout, frame_bytes)
            if not chunk or len(chunk) < frame_bytes:
                break
            frame = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).reshape(args.src_height, args.src_width, 3).permute(2, 0, 1).float().div_(255).cuda().contiguous()
            for effect in effects:
                frame = torch.from_dlpack(effect.run(frame).image).contiguous()
            if encode is not None:
                encode.stdin.write((frame.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
            else:
                pending.append(frame.clamp(0, 1).cpu())
                flush_batch()
            done += 1
            if done % 12 == 0 or done == total:
                print(f"RTX_PROGRESS {done}/{max(total, done)}", flush=True)
        if encode is None:
            while pending:
                flush_batch(final=True)
    finally:
        decode.stdout.close()
        decode.wait()
        for effect in effects:
            effect.close()
    print(f"RTX_PROGRESS {done}/{done}", flush=True)
    if done == 0:
        print("No frames decoded from the source", file=sys.stderr)
        return 1
    if encode is not None:
        encode.stdin.close()
        if encode.wait():
            print("ffmpeg encoder failed", file=sys.stderr)
            return 1
        mux = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", str(temp), "-i", str(args.source), "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                              "-shortest", str(args.output)], capture_output=True, text=True)
        if mux.returncode:
            temp.replace(args.output)
        else:
            temp.unlink(missing_ok=True)
        print(f"Saved: {args.output} ({done} frames, {args.width}x{args.height} @ {args.fps:g} fps)")
    else:
        print(f"Wrote {batches_written} batch file(s) with {done} frames to {args.batches_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
