"""Apply optional GPU post-processing to a decoded SeedVR tensor batch."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "seedvr2"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
COLOR_MODES = ("none", "lab", "wavelet", "wavelet_adaptive", "hsv", "adain")


def _ffmpeg() -> str:
    try:
        from seedvr_studio.media import _tool
        return _tool("ffmpeg")
    except Exception:
        return "ffmpeg"


def load_reference_frames(source: Path, fps: float, frame_start: int, count: int, width: int, height: int, device: torch.device) -> torch.Tensor:
    """Decode `count` source frames from `frame_start`, scaled to (width, height), as (T, 3, H, W) in [-1, 1].

    Uses ffmpeg's accurate input seek so long sources are not decoded from the beginning.
    Missing frames past the end of the source repeat the last available frame.
    """
    seek = math.floor(max(0, frame_start) / max(fps, 1e-6) * 1_000_000) / 1_000_000
    command = [
        _ffmpeg(), "-v", "error", "-ss", f"{seek:.6f}", "-i", str(source), "-frames:v", str(count),
        "-vf", f"scale={width}:{height}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffmpeg could not read reference frames: {result.stderr.decode('utf-8', 'replace')[-2000:]}")
    frame_bytes = width * height * 3
    available = len(result.stdout) // frame_bytes
    if available == 0:
        raise RuntimeError(f"No reference frames decoded from {source} at frame {frame_start}")
    raw = torch.frombuffer(bytearray(result.stdout[: available * frame_bytes]), dtype=torch.uint8)
    frames = raw.reshape(available, height, width, 3).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    if available < count:
        frames = torch.cat([frames, frames[-1:].expand(count - available, -1, -1, -1)], dim=0)
    return frames / 127.5 - 1.0


def apply_color_correction(x: torch.Tensor, mode: str, reference: torch.Tensor, true_height: int, true_width: int, chunk: int = 4) -> torch.Tensor:
    """Pull the restored batch `x` (1, 3, T, H, W) in [0, 1] toward the source colors.

    Reuses SeedVR2's own color-fix implementations, applied to the unpadded region only.
    """
    if mode == "none":
        return x
    if mode not in COLOR_MODES:
        raise ValueError(f"Unknown color correction mode: {mode}")
    sys.path.insert(0, str(VENDOR))
    from src.utils.color_fix import (  # noqa: E402
        adaptive_instance_normalization, hsv_saturation_histogram_match, lab_color_transfer,
        wavelet_adaptive_color_correction, wavelet_reconstruction,
    )
    from src.utils.debug import Debug  # noqa: E402

    debug = Debug(enabled=False)
    functions = {
        "lab": lambda c, r: lab_color_transfer(c, r, debug, luminance_weight=0.8),
        "wavelet_adaptive": lambda c, r: wavelet_adaptive_color_correction(c, r, debug),
        "wavelet": lambda c, r: wavelet_reconstruction(c, r, debug),
        "hsv": lambda c, r: hsv_saturation_histogram_match(c, r, debug),
        "adain": lambda c, r: adaptive_instance_normalization(c, r),
    }
    apply = functions[mode]
    frames = x.shape[2]
    true_height = min(true_height or x.shape[3], x.shape[3])
    true_width = min(true_width or x.shape[4], x.shape[4])
    content = x[0].permute(1, 0, 2, 3)  # (T, 3, H, W)
    for start in range(0, frames, chunk):
        end = min(frames, start + chunk)
        region = content[start:end, :, :true_height, :true_width] * 2.0 - 1.0
        corrected = apply(region.contiguous(), reference[start:end].contiguous())
        content[start:end, :, :true_height, :true_width] = ((corrected.clamp(-1, 1) + 1.0) * 0.5).to(content.dtype)
        del region, corrected
    return content.permute(1, 0, 2, 3).unsqueeze(0)


def _box_blur(value: torch.Tensor, kernel: int) -> torch.Tensor:
    radius = kernel // 2
    return F.avg_pool2d(F.pad(value, (radius, radius, radius, radius), mode="replicate"), kernel, stride=1)


REFERENCE_HEIGHT = 720


def _kernel(base: int, height: int) -> int:
    """Scale a pixel kernel tuned for 720p with the frame height, keeping it odd and >= base."""
    scaled = max(base, int(round(base * max(1.0, height / REFERENCE_HEIGHT))))
    return scaled if scaled % 2 else scaled + 1


def _skin_likelihood(flat: torch.Tensor) -> torch.Tensor:
    red, green, blue = flat[:, 0:1], flat[:, 1:2], flat[:, 2:3]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 0.5 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    skin = (
        torch.sigmoid((cb - 0.26) * 32.0)
        * torch.sigmoid((0.55 - cb) * 32.0)
        * torch.sigmoid((cr - 0.50) * 36.0)
        * torch.sigmoid((0.73 - cr) * 32.0)
        * torch.sigmoid((cr - cb - 0.015) * 30.0)
        * torch.sigmoid((luma - 0.055) * 28.0)
        * torch.sigmoid((0.97 - luma) * 28.0)
    )
    return _box_blur(skin, _kernel(9, flat.shape[-2]))


def _stable_center_mask(skin: torch.Tensor, batch: int, ext_frames: int, local_start: int, local_end: int) -> torch.Tensor:
    shaped = skin.reshape(batch, ext_frames, 1, skin.shape[-2], skin.shape[-1]).permute(0, 2, 1, 3, 4)
    shaped = F.avg_pool3d(F.pad(shaped, (0, 0, 0, 0, 1, 1), mode="replicate"), (3, 1, 1), stride=1)
    return shaped[:, :, local_start:local_end].permute(0, 2, 1, 3, 4).reshape(-1, 1, skin.shape[-2], skin.shape[-1])


def apply_skin_finishing(
    video: torch.Tensor,
    evenness: float,
    smoothing: float,
    redness: float,
    shine: float,
    blemish_mode: str,
    preserve_marks: bool,
) -> torch.Tensor:
    """Apply conservative, non-generative complexion finishing."""
    evenness = max(0.0, min(1.0, float(evenness)))
    smoothing = max(0.0, min(1.0, float(smoothing)))
    redness = max(0.0, min(1.0, float(redness)))
    shine = max(0.0, min(1.0, float(shine)))
    blemish_mode = blemish_mode if blemish_mode in {"off", "subtle", "strong"} else "off"
    if max(evenness, smoothing, redness, shine) <= 0 and blemish_mode == "off":
        return video
    batch, _, frame_count, _, _ = video.shape
    for start in range(0, frame_count, 4):
        end = min(frame_count, start + 4)
        ext_start, ext_end = max(0, start - 1), min(frame_count, end + 1)
        frames = video[:, :, ext_start:ext_end].permute(0, 2, 1, 3, 4)
        flat = frames.reshape(-1, 3, frames.shape[-2], frames.shape[-1])
        ext_frames = ext_end - ext_start
        local_start, local_end = start - ext_start, start - ext_start + (end - start)
        skin = _stable_center_mask(_skin_likelihood(flat), batch, ext_frames, local_start, local_end)
        center = flat.reshape(batch, ext_frames, 3, flat.shape[-2], flat.shape[-1])[:, local_start:local_end]
        result = center.reshape(-1, 3, flat.shape[-2], flat.shape[-1])
        luma = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
        height = result.shape[-2]
        k = lambda base: _kernel(base, height)  # noqa: E731
        medium = luma - _box_blur(luma, k(7))
        edge_guard = 1.0 - 0.88 * torch.sigmoid((medium.abs() - 0.065) * 45.0)
        mask = (skin * edge_guard).clamp(0, 1)

        if evenness > 0:
            # Remove blotchy mid-scale variation while returning the fine band.
            target = result + _box_blur(result, k(21)) - _box_blur(result, k(5))
            result = result.lerp(target.clamp(0, 1), mask * evenness * 0.75)
        if smoothing > 0:
            result = result.lerp(_box_blur(result, k(3)), mask * smoothing * 0.85)
        if blemish_mode != "off":
            luma = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            local_luma = _box_blur(luma, k(5))
            threshold = 0.085 if preserve_marks else 0.052
            spot = torch.sigmoid(((luma - local_luma).abs() - threshold) * 65.0) * mask
            amount = 0.24 if blemish_mode == "subtle" else 0.48
            result = result.lerp(_box_blur(result, k(5)), spot * amount)
        if redness > 0:
            red, green, blue = result[:, 0:1], result[:, 1:2], result[:, 2:3]
            excess = (red - 0.5 * (green + blue) - 0.025).clamp_min(0)
            correction = excess * mask * redness * 0.55
            result = torch.cat((red - correction, green + correction * 0.18, blue + correction * 0.08), dim=1).clamp(0, 1)
        if shine > 0:
            luma = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            local_luma = _box_blur(luma, k(15))
            highlight_floor = torch.maximum(local_luma + 0.045, torch.full_like(luma, 0.70))
            reduction = (luma - highlight_floor).clamp_min(0) * mask * shine * 0.75
            result = (result - reduction).clamp(0, 1)
        video[:, :, start:end] = result.reshape(batch, end - start, 3, result.shape[-2], result.shape[-1]).permute(0, 2, 1, 3, 4)
    return video


def apply_skin_microtexture(video: torch.Tensor, strength: float) -> torch.Tensor:
    """Enhance existing skin-scale luma detail without sharpening facial edges.

    The soft mask uses broad YCbCr skin likelihood rather than an identity-changing
    face restoration model. Processing a few frames at a time limits 4K VRAM use,
    while a three-frame mask average keeps the strength stable through motion.
    """
    strength = max(0.0, min(3.0, float(strength)))
    if strength <= 0:
        return video
    batch, _, frame_count, _, _ = video.shape
    for start in range(0, frame_count, 4):
        end = min(frame_count, start + 4)
        ext_start = max(0, start - 1)
        ext_end = min(frame_count, end + 1)
        frames = video[:, :, ext_start:ext_end].permute(0, 2, 1, 3, 4)
        flat = frames.reshape(-1, 3, frames.shape[-2], frames.shape[-1])
        skin = _skin_likelihood(flat)

        # Stabilize only the mask, not the pixels, avoiding temporal ghosting.
        ext_frames = ext_end - ext_start
        local_start = start - ext_start
        local_end = local_start + (end - start)
        skin = _stable_center_mask(skin, batch, ext_frames, local_start, local_end)

        center = flat.reshape(batch, ext_frames, 3, flat.shape[-2], flat.shape[-1])[:, local_start:local_end]
        center = center.reshape(-1, 3, flat.shape[-2], flat.shape[-1])
        center_luma = 0.299 * center[:, 0:1] + 0.587 * center[:, 1:2] + 0.114 * center[:, 2:3]
        height = center.shape[-2]
        fine = center_luma - _box_blur(center_luma, _kernel(3, height))
        medium = center_luma - _box_blur(center_luma, _kernel(7, height))
        detail = (0.78 * fine + 0.22 * medium).clamp(-0.08, 0.08)

        # Back away from strong contours such as eyes, nostrils, lips, and hair.
        contour = medium.abs()
        edge_guard = 1.0 - 0.85 * torch.sigmoid((contour - 0.065) * 45.0)
        delta = detail * skin * edge_guard * strength
        enhanced = (center + delta).clamp(0, 1)
        video[:, :, start:end] = enhanced.reshape(batch, end - start, 3, enhanced.shape[-2], enhanced.shape[-1]).permute(0, 2, 1, 3, 4)
    return video


def process_batch(args: argparse.Namespace, input_path: Path, output_path: Path, frame_start: int, cache: dict) -> None:
    """Post-process one decoded batch. `cache` keeps GPU models (face restorer, tracker) alive across batches."""
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "video" in payload:
        video = payload["video"]
    else:
        video = payload
        payload = {"video": video}
    video = video.to(device="cuda", dtype=torch.float32, non_blocking=True)
    # Decoder output is [-1, 1], while the effects operate in [0, 1].
    x = ((video.clamp(-1, 1) + 1.0) * 0.5)
    if args.color_correction != "none":
        if not args.source or not args.source.is_file():
            raise SystemExit("--color-correction needs --source pointing at the video SeedVR2 rendered")
        true_h = args.target_height or x.shape[3]
        true_w = args.target_width or x.shape[4]
        reference = load_reference_frames(args.source, args.source_fps or 24.0, frame_start, x.shape[2], true_w, true_h, x.device)
        x = apply_color_correction(x, args.color_correction, reference, true_h, true_w)
        del reference
        print(f"Color correction ({args.color_correction}) applied from {args.source.name} frames {frame_start}-{frame_start + x.shape[2] - 1}")
    if args.face_model != "none" and args.face_strength > 0:
        from face_restore import FaceRestorer, FaceTracker, load_state, restore_frames, save_state
        restorer = cache.get("restorer")
        if restorer is None:
            restorer = cache["restorer"] = FaceRestorer(args.face_model, args.face_weights, x.device, fidelity=args.face_fidelity)
        tracker = cache.get("tracker")
        if tracker is None:
            tracker = cache["tracker"] = FaceTracker.from_state(load_state(args.face_state) if frame_start > 0 else None)
        real_frames = int(payload.get("original_frames") or x.shape[2]) if isinstance(payload, dict) else x.shape[2]
        frames = x[0, :, :real_frames].permute(1, 0, 2, 3)  # (T, 3, H, W), a view into x
        _, faces = restore_frames(frames, restorer, tracker, strength=args.face_strength, keep_detail=args.face_detail,
                                  min_face=args.face_min_size, true_height=args.target_height or x.shape[3], true_width=args.target_width or x.shape[4])
        x[0, :, :real_frames] = frames.permute(1, 0, 2, 3)
        save_state(args.face_state, tracker)
        print(f"Face restoration ({args.face_model}, strength {args.face_strength:g}, fidelity {args.face_fidelity:g}): {faces} face crops restored in {real_frames} frames")

    x = apply_skin_finishing(x, args.skin_evenness, args.skin_smoothing, args.skin_redness,
                             args.skin_shine, args.blemish_mode, args.preserve_marks)
    x = apply_skin_microtexture(x, args.microtexture_strength)
    sharpen = max(0.0, min(10.0, float(args.sharpen_strength)))
    if sharpen > 0:
        b, c, t, h, w = x.shape
        spatial = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        blurred = _box_blur(spatial, _kernel(3, h))
        spatial = (spatial + sharpen * (spatial - blurred)).clamp(0, 1)
        x = spatial.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    intensity = max(0.0, min(1.0, float(args.grain_intensity)))
    if intensity > 0:
        saturation = max(0.0, min(1.0, float(args.grain_saturation)))
        frames = []
        for offset in range(x.shape[2]):
            generator = torch.Generator(device="cuda")
            generator.manual_seed((int(args.seed) + int(frame_start) + offset) & 0x7FFFFFFF)
            noise = torch.randn((x.shape[0], 1, x.shape[3], x.shape[4]), generator=generator,
                                device=x.device, dtype=x.dtype)
            # Mild chroma variation, matching the film-grain behavior used by
            # the reference tool while remaining stable across batch boundaries.
            colored = noise.repeat(1, 3, 1, 1)
            colored[:, 0] *= 2.0
            colored[:, 2] *= 3.0
            gray = colored[:, 1:2].repeat(1, 3, 1, 1)
            frames.append(saturation * colored + (1.0 - saturation) * gray)
        x = (x + torch.stack(frames, dim=2) * intensity).clamp(0, 1)
    payload["video"] = (x * 2.0 - 1.0).to(dtype=torch.float16).cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=Path, help="JSON list of {input, output, frame_start}; models load once for all of them")
    parser.add_argument("--sharpen-strength", type=float, default=0.0)
    parser.add_argument("--grain-intensity", type=float, default=0.0)
    parser.add_argument("--grain-saturation", type=float, default=0.5)
    parser.add_argument("--microtexture-strength", type=float, default=0.0)
    parser.add_argument("--skin-evenness", type=float, default=0.0)
    parser.add_argument("--skin-smoothing", type=float, default=0.0)
    parser.add_argument("--skin-redness", type=float, default=0.0)
    parser.add_argument("--skin-shine", type=float, default=0.0)
    parser.add_argument("--blemish-mode", choices=("off", "subtle", "strong"), default="off")
    parser.add_argument("--preserve-marks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--color-correction", choices=COLOR_MODES, default="none")
    parser.add_argument("--source", type=Path, help="Video SeedVR2 was fed; reference for color correction")
    parser.add_argument("--source-fps", type=float, default=0.0)
    parser.add_argument("--target-width", type=int, default=0, help="Unpadded output width (padding is skipped)")
    parser.add_argument("--target-height", type=int, default=0)
    parser.add_argument("--face-model", choices=("none", "codeformer", "gfpgan"), default="none")
    parser.add_argument("--face-strength", type=float, default=0.0, help="Blend of the restored face, 0-1")
    parser.add_argument("--face-fidelity", type=float, default=0.7, help="CodeFormer fidelity weight: 1 keeps identity, 0 hallucinates")
    parser.add_argument("--face-detail", type=float, default=0.5, help="Fine texture kept from SeedVR2 output, 0-1")
    parser.add_argument("--face-min-size", type=int, default=64, help="Ignore faces smaller than this (frame px)")
    parser.add_argument("--face-state", type=Path, help="JSON file carrying face tracks between batches")
    parser.add_argument("--face-weights", type=Path, default=ROOT / "models" / "faces")
    args = parser.parse_args()

    if args.jobs:
        jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        cache: dict = {}
        for index, job in enumerate(jobs, start=1):
            process_batch(args, Path(job["input"]), Path(job["output"]), int(job.get("frame_start", 0)), cache)
            print(f"POSTPROCESS_PROGRESS {index}/{len(jobs)}", flush=True)
        return
    if args.input is None or args.output is None:
        raise SystemExit("Give an input and --output, or --jobs manifest.json")
    process_batch(args, args.input, args.output, args.frame_start, {})


if __name__ == "__main__":
    main()
