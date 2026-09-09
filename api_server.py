"""Small local API/static host for the next SeedVR Studio frontend.

This intentionally does not replace the Gradio app or render pipeline yet.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4
import json
import os
import signal
import shutil
import subprocess
import time
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from seedvr_studio.backend import MODEL_FILES, RTX_BACKEND, RTX_QUALITIES, backend_status, render, reprocess_tensorrt, rtx_status, tensorrt_status
from seedvr_studio.cancellation import begin_render, cancel_current_render, cancellation_requested
from seedvr_studio.batch import BatchQueue, output_folder as batch_output_folder
from seedvr_studio.dialogs import DialogUnavailable, available_pickers, pick_folder, pick_video_file, pick_video_files
from seedvr_studio.jobs import _settings
from seedvr_studio.media import FrameRateRequiredError, align_clip_to_frames, concat_videos, make_center_crop, make_clip, make_downscaled, make_thumbnail, probe, resolve_frame_rate, trim_video_start
from seedvr_studio.paths import ensure_workspace
from seedvr_studio.updater import UpdateError, check_for_updates, launch_updater

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
OUTPUTS = ROOT / "outputs"

app = FastAPI(title="SeedVR Studio API", version="0.1.0")
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = Lock()

@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    response = await call_next(request)
    if request.url.path in {"/", "/index.html", "/app.js", "/styles.css"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response




def _update(job_id: str, **values: object) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(values)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "seedvr-js", "render_pipeline": "ready"}


OUTPUT_PRESETS: dict[str, dict[str, object]] = {
    "Original / enhancement only": {"scale": 1.0, "description": "Keep the source dimensions; enhancement only."},
    "1.5x source": {"scale": 1.5, "description": "Source dimensions × 1.5."},
    "2x source": {"scale": 2.0, "description": "Source dimensions × 2."},
    "3x source": {"scale": 3.0, "description": "Source dimensions × 3."},
    "4x source": {"scale": 4.0, "description": "Source dimensions × 4."},
    "720p": {"resolution": 720, "max_resolution": 1280, "description": "720p-class output."},
    "1K / 1080p": {"resolution": 1080, "max_resolution": 1920, "description": "1080p-class output."},
    "2K / 1440p": {"resolution": 1440, "max_resolution": 2560, "description": "1440p-class output."},
    "4K / 2160p": {"resolution": 2160, "max_resolution": 3840, "description": "2160p / UHD output."},
    "8K / 4320p": {"resolution": 4320, "max_resolution": 7680, "description": "4320p / 8K UHD output. Very slow; needs large VRAM."},
}
_OUTPUT_PRESET_ALIASES = {"Original enhancement only": "Original / enhancement only"}


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def resolve_output_size(preset_name: str, width: int, height: int) -> tuple[int, int]:
    """Return (short side target, long side cap) for a preset given the source dimensions.

    Fixed presets set the size regardless of the source; scale presets multiply the
    source dimensions and round to even pixels.
    """
    preset = OUTPUT_PRESETS.get(_OUTPUT_PRESET_ALIASES.get(preset_name, preset_name)) or OUTPUT_PRESETS["1K / 1080p"]
    if "scale" in preset:
        scale = float(preset["scale"])
        return _even(min(width, height) * scale), _even(max(width, height) * scale)
    return int(preset["resolution"]), int(preset["max_resolution"])


@app.get("/api/config")
def config() -> dict[str, object]:
    installed, engine = backend_status()
    trt_installed, trt = tensorrt_status()
    rtx_installed, rtx = rtx_status()
    return {
        "models": list(MODEL_FILES),
        "output_presets": OUTPUT_PRESETS,
        "crop_policies": ["Preserve original aspect ratio", "Center crop to 16:9"],
        "batch_sizes": {
            "SeedVR2 + TensorRT": [5, 21],
            "SeedVR2 (Legacy)": [1, 5, 9, 13, 17, 21, 33, 45],
            RTX_BACKEND: [21],
        },
        "engines": ["SeedVR2 + TensorRT", "SeedVR2 (Legacy)", RTX_BACKEND],
        "rtx_qualities": {"ULTRA": "Ultra (best)", "HIGH": "High", "MEDIUM": "Medium", "LOW": "Low", "BICUBIC": "Bicubic (no AI, reference)"},
        "seam_modes": {"off": "Off / original output", "noise": "Noise match — gentle", "match": "Color + noise match", "blend": "Boundary dissolve — strongest"},
        "features": {"skin_finishing": True, "open_output_folder": True, "saved_settings": True, "persistent_decoder": True, "file_dialog": bool(available_pickers()), "face_restoration": True},
        "face_models": {"none": "Off", "codeformer": "CodeFormer (fidelity control, recommended)", "gfpgan": "GFPGAN v1.4"},
        "decoder_modes": {"stable": "Stable", "optimized": "Optimized (Beta)", "optimized_fast": "Optimized Fast (Beta)"},
        "backend": {"seedvr": {"ready": installed, "message": engine}, "tensorrt": {"ready": trt_installed, "message": trt}, "rtx": {"ready": rtx_installed, "message": rtx}},
        "presets": {
            "Custom / manual": None,
            "8 GB VRAM — low memory": {"model": "3B FP8 — faster / less VRAM", "resolution": 480, "max_resolution": 1920, "batch_size": 5, "attention": "sageattn_2", "blocks": 36, "vae_tiling": True},
            "12 GB VRAM — mainstream": {"model": "3B FP8 — faster / less VRAM", "resolution": 720, "max_resolution": 1920, "batch_size": 5, "attention": "sageattn_2", "blocks": 24, "vae_tiling": True},
            "16 GB VRAM — high memory": {"model": "3B FP8 — faster / less VRAM", "resolution": 1080, "max_resolution": 2560, "batch_size": 5, "attention": "sageattn_2", "blocks": 12, "vae_tiling": True},
            "24 GB VRAM — enthusiast": {"model": "3B FP8 — faster / less VRAM", "resolution": 1080, "max_resolution": 3840, "batch_size": 21, "attention": "sageattn_2", "blocks": 0, "vae_tiling": False},
            "32 GB+ VRAM — workstation": {"model": "3B FP16 — best 3B quality", "resolution": 1440, "max_resolution": 3840, "batch_size": 21, "attention": "sageattn_2", "blocks": 0, "vae_tiling": False},
        },
    }


@app.get("/api/outputs")
def outputs() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    fps_by_directory: dict[Path, float] = {}
    for path in sorted(OUTPUTS.rglob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True):
        relative = path.relative_to(OUTPUTS).as_posix()
        if path.parent not in fps_by_directory:
            source_candidates = sorted(candidate for candidate in path.parent.glob("source.*") if candidate.is_file())
            try:
                fps_by_directory[path.parent] = probe(source_candidates[0] if source_candidates else path).fps
            except Exception:
                fps_by_directory[path.parent] = 30.0
        result.append({
            "name": path.name,
            "path": relative,
            "url": f"/media/{relative}",
            "bytes": path.stat().st_size,
            "modified": path.stat().st_mtime,
            "fps": fps_by_directory[path.parent],
            "reprocessable": any((path.parent / "tensorrt_decoded").glob("decoded_*.pt")),
        })
    return result


def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes", "on"}


SOURCE_DIVISORS = (1.0, 2.0, 3.0, 4.0)


def _source_divisor(values: dict[str, object]) -> float:
    """Pre-encode downscale factor: `source_scale` (1, 2, 3, 4) or the legacy `half_source` flag."""
    try:
        divisor = float(values.get("source_scale") or 0)
    except (TypeError, ValueError):
        divisor = 0.0
    if divisor <= 0 and _bool(values.get("half_source", False)):
        divisor = 2.0
    return divisor if divisor in SOURCE_DIVISORS else 1.0


def _resolve_source(source_path: str, upload: UploadFile | None) -> tuple[Path | None, str, Path | None]:
    """Return (local_file, original_name, delivery_dir) for a job.

    `source_path` may name a video file (used directly, no upload needed) or a folder
    (the uploaded file is rendered and the result is delivered into that folder).
    """
    if source_path.strip():
        candidate = Path(source_path.strip()).expanduser()
        if candidate.is_file():
            if upload is not None and upload.filename:
                # A fresh upload wins over a stale path; the path's folder still receives the result.
                return None, upload.filename, candidate.parent
            return candidate, candidate.name, candidate.parent
        if candidate.is_dir():
            if upload is None or not upload.filename:
                raise HTTPException(status_code=400, detail="Upload a video or enter the full path to a video file.")
            return None, upload.filename, candidate
        raise HTTPException(status_code=400, detail=f"Source path was not found: {candidate}")
    if upload is None or not upload.filename:
        raise HTTPException(status_code=400, detail="Upload a video or enter the full path to a video file.")
    return None, upload.filename, None


def _active_job_dirs() -> set[Path]:
    with JOBS_LOCK:
        return {Path(str(job.get("_job_dir"))) for job in JOBS.values() if job.get("status") in {"queued", "running"} and job.get("_job_dir")}


def _is_undelivered_result(job_dir: Path) -> bool:
    """A finished full render that has no home outside the workspace must survive cleanup."""
    if (job_dir / "delivered.json").exists():
        return False
    return any(path.is_file() for pattern in ("*-seed.mp4", "*-restored.mp4") for path in job_dir.glob(pattern))


def cleanup_workspace() -> dict[str, object]:
    """Delete temporary job folders; keep running jobs and undelivered full renders."""
    removed: list[str] = []
    kept: list[str] = []
    active = _active_job_dirs()
    if not OUTPUTS.is_dir():
        return {"removed": removed, "kept": kept}
    for job_dir in sorted(OUTPUTS.glob("js-*")):
        if not job_dir.is_dir():
            continue
        if job_dir.resolve() in {path.resolve() for path in active} or _is_undelivered_result(job_dir):
            kept.append(job_dir.name)
            continue
        try:
            shutil.rmtree(job_dir)
            removed.append(job_dir.name)
        except OSError as exc:
            kept.append(f"{job_dir.name} ({exc})")
    return {"removed": removed, "kept": kept}


@app.on_event("startup")
def _cleanup_on_startup() -> None:
    result = cleanup_workspace()
    if result["removed"]:
        print(f"Removed {len(result['removed'])} temporary job folder(s) from outputs/", flush=True)


@app.post("/api/workspace/clear")
def clear_workspace() -> dict[str, object]:
    return cleanup_workspace()


def _delivery_target(directory: Path, stem: str) -> Path:
    """Return `<stem>-seed.mp4` in `directory`, never overwriting an existing file."""
    target = directory / f"{stem}-seed.mp4"
    counter = 2
    while target.exists():
        target = directory / f"{stem}-seed-{counter}.mp4"
        counter += 1
    return target


CHUNK_LENGTH_OPTIONS = (30, 60, 120, 180, 300, 600, 900, 1800)


def _automatic_chunk_seconds(info, batch_size: int) -> int:
    """Choose about ten checkpoints using frame count and temporal-batch alignment."""
    fps = max(float(info.fps), 1.0)
    total_frames = max(1, round(float(info.duration) * fps))
    temporal_batch = max(1, int(batch_size))
    target_frames = max(temporal_batch, round(total_frames / 10 / temporal_batch) * temporal_batch)
    desired_seconds = target_frames / fps
    return min(CHUNK_LENGTH_OPTIONS, key=lambda seconds: abs(seconds - desired_seconds))


def _render_chunked(job_id: str, source: Path, job_dir: Path, output: Path, settings, info, values: dict[str, object], report) -> None:
    """Render long videos in checkpointed temporal chunks, resuming completed work."""
    if settings.stop_before_vae:
        raise RuntimeError("Chunked rendering is unavailable when Stop before VAE is enabled.")
    requested_chunk_seconds = float(values.get("chunk_seconds", 0))
    auto_chunking = requested_chunk_seconds <= 0
    chunk_seconds = float(_automatic_chunk_seconds(info, settings.batch_size) if auto_chunking else max(5.0, requested_chunk_seconds))
    if auto_chunking:
        frames_per_chunk = round(chunk_seconds * max(float(info.fps), 1.0))
        report(0.001, f"Auto chunk length selected: {chunk_seconds / 60:g} minutes ({frames_per_chunk} frames per chunk)")
    overlap_frames = max(0, min(20, int(settings.batch_size) - 1))
    overlap_seconds = overlap_frames / max(info.fps, 1.0)
    checkpoint_path = job_dir / "chunk-manifest.json"
    manifest: dict[str, object] = {"chunk_seconds": chunk_seconds, "chunk_mode": "auto" if auto_chunking else "manual", "source_frames": round(info.duration * info.fps), "overlap_frames": overlap_frames, "duration": info.duration, "completed": [], "chunks": []}
    if checkpoint_path.exists():
        try:
            existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if existing.get("chunk_seconds") == chunk_seconds and existing.get("overlap_frames") == overlap_frames:
                manifest.update(existing)
        except (OSError, ValueError, TypeError):
            pass
    total = max(1, int((info.duration + chunk_seconds - 1e-6) // chunk_seconds))
    completed = {int(index) for index in manifest.get("completed", [])}
    rendered: list[Path] = []
    for index in range(total):
        start = index * chunk_seconds
        if start >= info.duration:
            break
        chunk_end = min(info.duration, start + chunk_seconds)
        chunk_path = job_dir / "chunks" / f"chunk-{index:04d}.mp4"
        final_chunk = job_dir / "chunks" / f"final-{index:04d}.mp4"
        if index in completed and final_chunk.exists():
            rendered.append(final_chunk)
            report((index + 1) / total, f"Resuming: chunk {index + 1} of {total} already complete")
            continue
        render_start = max(0.0, start - overlap_seconds) if index else 0.0
        render_length = min(info.duration - render_start, (chunk_end - start) + (start - render_start))
        report(index / total, f"Rendering chunk {index + 1} of {total}")
        make_clip(source, job_dir / "chunks" / f"source-{index:04d}.mp4", render_start, render_length)
        # The backend name is supplied through the values map; this keeps chunk rendering identical to normal rendering.
        from seedvr_studio.backend import render as render_backend
        render_backend(str(values["backend_name"]), job_dir / "chunks" / f"source-{index:04d}.mp4", chunk_path, settings,
                       lambda progress, message: report((index + min(progress, 0.99)) / total, f"Chunk {index + 1}/{total}: {message}"))
        if index and overlap_seconds > 0:
            trim_video_start(chunk_path, final_chunk, overlap_seconds)
        else:
            final_chunk = chunk_path
        rendered.append(final_chunk)
        completed.add(index)
        manifest["completed"] = sorted(completed)
        manifest["chunks"] = [str(path.relative_to(job_dir)) for path in rendered]
        checkpoint_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _update(job_id, chunk_index=index + 1, chunk_total=total, resumable=True, checkpoint=str(checkpoint_path), log_file=str(job_dir / "render.log"))
    report(0.99, "Assembling completed chunks")
    concat_videos(rendered, output)
    manifest["assembled"] = str(output.relative_to(job_dir))
    checkpoint_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _render_job(job_id: str, source: Path, job_dir: Path, values: dict[str, object]) -> None:
    begin_render()
    started = time.perf_counter()
    try:
        source_info = probe(source)
        crop_policy = str(values.get("crop_policy", "Preserve original aspect ratio"))
        if crop_policy == "Center crop to 16:9":
            source = make_center_crop(source, job_dir / f"cropped{source.suffix or '.mp4'}")
        info = probe(source)
        resolved_fps = resolve_frame_rate(info.fps, values.get("source_fps", 0.0))
        info = replace(info, fps=resolved_fps, frames=info.frames or round(info.duration * resolved_fps))
        output_preset = str(values.get("output_preset", "1K / 1080p"))
        # Output size always derives from the original (cropped) source, never from the halved copy.
        values["resolution"], values["max_resolution"] = resolve_output_size(output_preset, info.width, info.height)
        display_source = source
        source_divisor = _source_divisor(values)
        if source_divisor > 1:
            _update(job_id, status="running", progress=0.01, message=f"Downscaling source to 1/{source_divisor:g} before SeedVR2")
            source = make_downscaled(source, job_dir / f"downscaled{source.suffix or '.mp4'}", source_divisor)
            smaller = probe(source)
            info = replace(info, width=smaller.width, height=smaller.height)
        job_type = str(values["job_type"])
        slot: dict[str, object] = {}
        if job_type == "preview":
            start = min(max(float(values["preview_start"]), 0.0), max(info.duration - 0.1, 0.0))
            length = min(float(values["preview_seconds"]), max(info.duration - start, 0.1))
            # Snap to whole frames so the preview's first frame is the frame the viewer shows at `start`.
            start, length, _first_frame = align_clip_to_frames(start, length, info.fps)
            source_for_render = make_clip(source, job_dir / "preview-source.mp4", start, length, info.fps)
            output = job_dir / "restored.mp4"
            slot = {"preview_start": start, "preview_seconds": length, "source_duration": info.duration, "crop_policy": crop_policy}
        else:
            source_for_render = source
            output = job_dir / f"{values.get('original_stem') or source.stem}-seed.mp4"
        settings = _settings(values["resolution"], values["max_resolution"], values["batch_size"], values["seed"], values["model_label"], values["color_correction"], values["attention_mode"], values["blocks_to_swap"], values["vae_tiling"], values["stop_before_vae"], values["sharpen_enabled"], values["sharpen_strength"], values["grain_enabled"], values["grain_intensity"], values["grain_saturation"], values.get("microtexture_enabled", False), values.get("microtexture_strength", .60), values.get("skin_finishing_enabled", False), values.get("skin_evenness", .25), values.get("skin_smoothing", .20), values.get("skin_redness", .15), values.get("skin_shine", .15), values.get("blemish_mode", "off"), values.get("preserve_marks", True), values.get("seam_mode", "match"), values.get("seam_frames", 2), values.get("decoder_mode", "optimized_fast"), info.fps)
        settings = replace(settings, face_model=str(values.get("face_model", "none")), face_strength=float(values.get("face_strength", .5)), face_fidelity=float(values.get("face_fidelity", .7)), face_detail=float(values.get("face_detail", .5)), face_min_size=int(values.get("face_min_size", 64)), rtx_quality=str(values.get("rtx_quality", "ULTRA")).upper())
        (job_dir / "job-manifest.json").write_text(json.dumps({
            "version": 1, "job_id": job_id, "job_type": job_type,
            "backend": str(values["backend_name"]), "source": str(source_for_render),
            "display_source": str(display_source), "preview": slot or None,
            "output": str(output), "settings": settings.__dict__,
        }, indent=2), encoding="utf-8")
        _update(job_id, status="running", progress=0.02, message="Starting SeedVR2", fps=info.fps, duration=length if job_type == "preview" else info.duration, started_at=time.time())
        log_path = job_dir / "render.log"
        def report(progress: float, message: str) -> None:
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{progress:.3f}] {message}\n")
            except OSError:
                pass
            _update(job_id, progress=float(progress), message=message)
        if _bool(values.get("chunked_render", False)) and job_type == "full":
            _render_chunked(job_id, source_for_render, job_dir, output, settings, info, values, report)
        else:
            render(str(values["backend_name"]), source_for_render, output, settings, report)
        if settings.stop_before_vae:
            _update(job_id, status="complete", progress=1.0, message="Latents captured; stopped before VAE decode", elapsed_seconds=time.perf_counter() - started, original_url=f"/media/{source_for_render.relative_to(OUTPUTS).as_posix()}", restored_url=None, output=None)
            return
        delivered: dict[str, object] = {}
        delivery_dir = values.get("delivery_dir")
        if job_type == "full" and delivery_dir:
            target = _delivery_target(Path(str(delivery_dir)), str(values.get("original_stem") or output.stem))
            try:
                report(0.995, f"Copying result to {target}")
                shutil.copy2(output, target)
                delivered = {"saved_path": str(target)}
                (job_dir / "delivered.json").write_text(json.dumps({"saved_path": str(target), "source": str(source)}, indent=2), encoding="utf-8")
            except OSError as exc:
                delivered = {"saved_error": f"Result stayed in the project folder; could not write {target}: {exc}"}
                report(0.995, str(delivered["saved_error"]))
        original_for_display = display_source if job_type == "full" else source_for_render
        _update(job_id, status="complete", progress=1.0, message="Render complete", elapsed_seconds=time.perf_counter() - started, original_url=f"/media/{original_for_display.relative_to(OUTPUTS).as_posix()}", restored_url=f"/media/{output.relative_to(OUTPUTS).as_posix()}", output=str(output), output_relative=output.relative_to(OUTPUTS).as_posix(), reprocessable=any((job_dir / "tensorrt_decoded").glob("decoded_*.pt")), **slot, **delivered)
    except Exception as exc:
        log_path = job_dir / "render.log"
        details = "\n".join([f"SeedVR Studio render failure at {time.strftime('%Y-%m-%d %H:%M:%S')}", repr(exc), traceback.format_exc()])
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(details + "\n")
        except OSError:
            pass
        final_status = "cancelled" if cancellation_requested() else "error"
        _update(job_id, status=final_status, message=str(exc), elapsed_seconds=time.perf_counter() - started, error=str(exc), failure_reason=str(exc), failure_code="fps_required" if isinstance(exc, FrameRateRequiredError) else None, log_file=str(log_path), resumable=_bool(values.get("chunked_render", False)))


def _reprocess_job(job_id: str, source_output: Path, values: dict[str, object]) -> None:
    started = time.perf_counter()
    job_dir = source_output.parent
    try:
        source_candidates = sorted(path for path in job_dir.glob("source.*") if "restored" not in path.stem and path.is_file())
        if not source_candidates:
            raise RuntimeError("The original source video for this result is missing.")
        source = source_candidates[0]
        display_source = source
        slot: dict[str, object] = {}
        try:
            manifest = json.loads((job_dir / "job-manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        # Reprocess against the clip SeedVR2 actually rendered (preview clip / halved copy), not the full upload.
        rendered_source = Path(str(manifest.get("source") or ""))
        if rendered_source.is_file():
            source = rendered_source
        if isinstance(manifest.get("preview"), dict):
            slot = dict(manifest["preview"])
            display_source = source
        elif Path(str(manifest.get("display_source") or "")).is_file():
            display_source = Path(str(manifest["display_source"]))
        output = job_dir / f"{source_output.stem}-post-{job_id}.mp4"
        log_path = job_dir / f"reprocess-{job_id}.log"
        _update(job_id, status="running", progress=0.01, message="Loading saved TensorRT batches", started_at=time.time(), log_file=str(log_path))
        def report(progress: float, message: str) -> None:
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{progress:.3f}] {message}\n")
            except OSError:
                pass
            _update(job_id, progress=float(progress), message=message)
        reprocess_tensorrt(job_dir, source, output,
                           sharpen_enabled=_bool(values.get("sharpen_enabled", False)), sharpen_strength=float(values.get("sharpen_strength", .25)),
                           grain_enabled=_bool(values.get("grain_enabled", False)), grain_intensity=float(values.get("grain_intensity", .02)), grain_saturation=float(values.get("grain_saturation", .5)),
                           microtexture_enabled=_bool(values.get("microtexture_enabled", False)), microtexture_strength=float(values.get("microtexture_strength", .60)),
                           skin_finishing_enabled=_bool(values.get("skin_finishing_enabled", False)), skin_evenness=float(values.get("skin_evenness", .25)), skin_smoothing=float(values.get("skin_smoothing", .20)),
                           skin_redness=float(values.get("skin_redness", .15)), skin_shine=float(values.get("skin_shine", .15)), blemish_mode=str(values.get("blemish_mode", "off")), preserve_marks=_bool(values.get("preserve_marks", True)),
                           seed=int(values.get("seed", 42)), seam_mode=str(values.get("seam_mode", "match")), seam_frames=int(values.get("seam_frames", 2)), color_correction=str(values.get("color_correction", "none")),
                           face_model=str(values.get("face_model", "none")), face_strength=float(values.get("face_strength", .5)), face_fidelity=float(values.get("face_fidelity", .7)),
                           face_detail=float(values.get("face_detail", .5)), face_min_size=int(values.get("face_min_size", 64)), progress_callback=report)
        _update(job_id, status="complete", progress=1.0, message="Post-only reprocess complete", elapsed_seconds=time.perf_counter() - started,
                original_url=f"/media/{display_source.relative_to(OUTPUTS).as_posix()}", restored_url=f"/media/{output.relative_to(OUTPUTS).as_posix()}",
                output=str(output), output_relative=output.relative_to(OUTPUTS).as_posix(), reprocessable=True, fps=probe(source).fps, **slot)
    except Exception as exc:
        log_path = job_dir / f"reprocess-{job_id}.log"
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{repr(exc)}\n{traceback.format_exc()}\n")
        except OSError:
            pass
        _update(job_id, status="error", message=str(exc), error=str(exc), failure_reason=str(exc), elapsed_seconds=time.perf_counter() - started, log_file=str(log_path), resumable=False)


@app.post("/api/jobs")
async def create_job(
    file: UploadFile | None = File(None),
    source_path: str = Form(""),
    job_type: str = Form("preview"), backend_name: str = Form("SeedVR2 + TensorRT"),
    output_preset: str = Form("1K / 1080p"), crop_policy: str = Form("Preserve original aspect ratio"),
    preview_start: float = Form(0), preview_seconds: float = Form(3), resolution: int = Form(1080), max_resolution: int = Form(3840), batch_size: int = Form(21), seed: int = Form(42),
    model_label: str = Form("3B FP16 — best 3B quality"), color_correction: str = Form("none"), attention_mode: str = Form("sageattn_2"), blocks_to_swap: int = Form(0), vae_tiling: str = Form("false"), stop_before_vae: str = Form("false"),
    sharpen_enabled: str = Form("false"), sharpen_strength: float = Form(.25), grain_enabled: str = Form("false"), grain_intensity: float = Form(.02), grain_saturation: float = Form(.5),
    microtexture_enabled: str = Form("false"), microtexture_strength: float = Form(.60),
    skin_finishing_enabled: str = Form("false"), skin_evenness: float = Form(.25), skin_smoothing: float = Form(.20),
    skin_redness: float = Form(.15), skin_shine: float = Form(.15), blemish_mode: str = Form("off"), preserve_marks: str = Form("true"),
    seam_mode: str = Form("match"), seam_frames: int = Form(2),
    chunked_render: str = Form("false"), chunk_seconds: float = Form(0), decoder_mode: str = Form("optimized_fast"), source_fps: float = Form(0),
    half_source: str = Form("false"), source_scale: float = Form(1),
    face_model: str = Form("none"), face_strength: float = Form(.5), face_fidelity: float = Form(.7), face_detail: float = Form(.5), face_min_size: int = Form(64),
    rtx_quality: str = Form("ULTRA"),
) -> dict[str, str]:
    form = dict(locals())
    form.pop("file", None)
    form.pop("source_path", None)
    if BATCH.state != "idle":
        raise HTTPException(status_code=409, detail="A batch is running. Pause or stop it before rendering single videos.")
    local_file, original_name, delivery_dir = _resolve_source(source_path, file)
    job_id, source, job_dir, values = _prepare_job(local_file, file, original_name, delivery_dir, form)
    Thread(target=_render_job, args=(job_id, source, job_dir, values), daemon=True).start()
    return {"id": job_id}


RENDER_DEFAULTS: dict[str, object] = {
    "backend_name": "SeedVR2 + TensorRT", "output_preset": "1K / 1080p", "crop_policy": "Preserve original aspect ratio",
    "preview_start": 0.0, "preview_seconds": 3.0, "resolution": 1080, "max_resolution": 3840, "batch_size": 21, "seed": 42,
    "model_label": "3B FP16 — best 3B quality", "color_correction": "none", "attention_mode": "sageattn_2", "blocks_to_swap": 0,
    "vae_tiling": False, "stop_before_vae": False, "sharpen_enabled": False, "sharpen_strength": 0.25,
    "grain_enabled": False, "grain_intensity": 0.02, "grain_saturation": 0.5, "microtexture_enabled": False, "microtexture_strength": 0.60,
    "skin_finishing_enabled": False, "skin_evenness": 0.25, "skin_smoothing": 0.20, "skin_redness": 0.15, "skin_shine": 0.15,
    "blemish_mode": "off", "preserve_marks": True, "seam_mode": "match", "seam_frames": 2,
    "chunked_render": False, "chunk_seconds": 0.0, "decoder_mode": "optimized_fast", "source_fps": 0.0, "half_source": False, "source_scale": 1.0,
    "face_model": "none", "face_strength": 0.5, "face_fidelity": 0.7, "face_detail": 0.5, "face_min_size": 64, "rtx_quality": "ULTRA",
}
RENDER_BOOL_FIELDS = ("vae_tiling", "stop_before_vae", "sharpen_enabled", "grain_enabled", "microtexture_enabled", "skin_finishing_enabled", "preserve_marks", "half_source", "chunked_render")
RENDER_INT_FIELDS = ("resolution", "max_resolution", "batch_size", "seed", "blocks_to_swap", "seam_frames", "face_min_size")
RENDER_FLOAT_FIELDS = ("source_scale", "face_strength", "face_fidelity", "face_detail", "preview_start", "preview_seconds", "sharpen_strength", "grain_intensity", "grain_saturation", "microtexture_strength", "skin_evenness", "skin_smoothing", "skin_redness", "skin_shine", "chunk_seconds", "source_fps")


def _prepare_job(local_file: Path | None, upload: UploadFile | None, original_name: str, delivery_dir: Path | None, form: dict[str, object]) -> tuple[str, Path, Path, dict[str, object]]:
    """Create the job folder, copy the source in, and register the job. Returns (job_id, source, job_dir, values)."""
    ensure_workspace()
    job_type = str(form.get("job_type", "preview"))
    job_id = uuid4().hex[:10]
    job_dir = OUTPUTS / f"js-{job_type}-{job_id}"
    job_dir.mkdir(parents=True)
    suffix = Path(original_name or "input.mp4").suffix or ".mp4"
    source = job_dir / f"source{suffix}"
    if local_file is not None:
        shutil.copy2(local_file, source)
    else:
        assert upload is not None
        with source.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
    values: dict[str, object] = {**RENDER_DEFAULTS, **{key: value for key, value in form.items() if value is not None and value != ""}}
    values.update({"job_type": job_type, "job_id": job_id, "job_dir": job_dir, "source": source, "original_name": original_name,
                   "original_stem": Path(original_name).stem, "delivery_dir": str(delivery_dir) if delivery_dir else ""})
    for field in RENDER_BOOL_FIELDS:
        values[field] = _bool(values.get(field, False))
    for field in RENDER_INT_FIELDS:
        if field in values and isinstance(values[field], str):
            values[field] = int(float(values[field] or 0))
    for field in RENDER_FLOAT_FIELDS:
        if field in values and isinstance(values[field], str):
            values[field] = float(values[field] or 0)
    chunk_seconds = float(values.get("chunk_seconds", 0) or 0)
    values["chunk_seconds"] = chunk_seconds if chunk_seconds > 0 else 0.0
    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "message": "Queued", "job_type": job_type, "resumable": bool(values["chunked_render"]) and job_type == "full", "_source": str(source), "_job_dir": str(job_dir), "_values": values}
    return job_id, source, job_dir, values


# ----- batch mode -----
def _batch_runner(item: dict[str, object], settings: dict[str, object], progress) -> dict[str, object]:
    """Render one batch item as a full job and wait for it; results land in <folder>/VRUpscale."""
    source_path = Path(str(item["path"]))
    if not source_path.is_file():
        return {"status": "failed", "message": f"Source is missing: {source_path}"}
    delivery_dir = batch_output_folder(source_path)
    delivery_dir.mkdir(parents=True, exist_ok=True)
    form = dict(settings)
    form["job_type"] = "full"
    job_id, source, job_dir, values = _prepare_job(source_path, None, source_path.name, delivery_dir, form)
    thread = Thread(target=_render_job, args=(job_id, source, job_dir, values), daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(0.5)
        with JOBS_LOCK:
            job = dict(JOBS.get(job_id, {}))
        progress(float(job.get("progress") or 0.0), str(job.get("message") or "Rendering"))
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id, {}))
    status = str(job.get("status"))
    if status == "complete":
        if job.get("saved_error"):
            return {"status": "failed", "message": str(job["saved_error"]), "job_id": job_id, "output": job.get("output")}
        return {"status": "done", "message": f"Saved to {job.get('saved_path')}", "job_id": job_id, "output": job.get("saved_path") or job.get("output")}
    if status == "cancelled":
        return {"status": "cancelled", "message": str(job.get("message") or "Cancelled"), "job_id": job_id}
    return {"status": "failed", "message": str(job.get("failure_reason") or job.get("error") or job.get("message") or status), "job_id": job_id}


BATCH = BatchQueue(OUTPUTS / "batch-queue.json", _batch_runner, cancel_current_render)


@app.get("/api/batch")
def batch_status() -> dict[str, object]:
    return BATCH.snapshot()


@app.post("/api/batch/add")
def batch_add(paths: str = Form(...), settings: str = Form("{}")) -> dict[str, object]:
    try:
        path_list = json.loads(paths)
        settings_map = json.loads(settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"paths and settings must be JSON: {exc}") from exc
    if not isinstance(path_list, list) or not isinstance(settings_map, dict):
        raise HTTPException(status_code=400, detail="paths must be a JSON list and settings a JSON object")
    settings_map.pop("job_type", None)
    result = BATCH.add([str(p) for p in path_list], settings_map)
    return {**result, **BATCH.snapshot()}


@app.post("/api/batch/start")
def batch_start() -> dict[str, object]:
    with JOBS_LOCK:
        busy = any(job.get("status") in {"queued", "running"} for job in JOBS.values())
    if busy and BATCH.state == "idle":
        raise HTTPException(status_code=409, detail="A render is in progress. Wait for it or stop it before starting the batch.")
    if not BATCH.start():
        raise HTTPException(status_code=400, detail="Nothing queued. Add a folder or files first.")
    return BATCH.snapshot()


@app.post("/api/batch/pause")
def batch_pause() -> dict[str, object]:
    BATCH.pause()
    return BATCH.snapshot()


@app.post("/api/batch/stop")
def batch_stop() -> dict[str, object]:
    BATCH.stop()
    return BATCH.snapshot()


@app.post("/api/batch/skip-current")
def batch_skip_current() -> dict[str, object]:
    BATCH.skip_current()
    return BATCH.snapshot()


@app.post("/api/batch/items/{item_id}/remove")
def batch_remove(item_id: str) -> dict[str, object]:
    if not BATCH.remove(item_id):
        raise HTTPException(status_code=409, detail="Item is running or unknown")
    return BATCH.snapshot()


@app.post("/api/batch/items/{item_id}/retry")
def batch_retry(item_id: str) -> dict[str, object]:
    if not BATCH.retry(item_id):
        raise HTTPException(status_code=409, detail="Only finished items can be retried")
    return BATCH.snapshot()


@app.post("/api/batch/items/{item_id}/move")
def batch_move(item_id: str, direction: int = Form(...)) -> dict[str, object]:
    BATCH.move(item_id, 1 if direction > 0 else -1)
    return BATCH.snapshot()


@app.post("/api/batch/clear-finished")
def batch_clear_finished() -> dict[str, object]:
    return {"removed": BATCH.clear_finished(), **BATCH.snapshot()}


@app.post("/api/pick-files")
def pick_files(initial: str = Form("")) -> dict[str, object]:
    try:
        chosen = pick_video_files(initial or None)
    except DialogUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"paths": chosen or [], "cancelled": chosen is None}


@app.post("/api/pick-folder")
def pick_folder_endpoint(initial: str = Form("")) -> dict[str, object]:
    try:
        chosen = pick_folder(initial or None)
    except DialogUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"path": chosen, "cancelled": chosen is None}


@app.post("/api/reprocess")
def create_reprocess_job(
    output_path: str = Form(...), seed: int = Form(42),
    sharpen_enabled: str = Form("false"), sharpen_strength: float = Form(.25),
    grain_enabled: str = Form("false"), grain_intensity: float = Form(.02), grain_saturation: float = Form(.5),
    microtexture_enabled: str = Form("false"), microtexture_strength: float = Form(.60),
    skin_finishing_enabled: str = Form("false"), skin_evenness: float = Form(.25), skin_smoothing: float = Form(.20),
    skin_redness: float = Form(.15), skin_shine: float = Form(.15), blemish_mode: str = Form("off"), preserve_marks: str = Form("true"),
    seam_mode: str = Form("match"), seam_frames: int = Form(2), color_correction: str = Form("none"),
    face_model: str = Form("none"), face_strength: float = Form(.5), face_fidelity: float = Form(.7), face_detail: float = Form(.5), face_min_size: int = Form(64),
) -> dict[str, str]:
    relative = output_path.removeprefix("/media/").lstrip("/\\")
    source_output = (OUTPUTS / relative).resolve()
    if OUTPUTS.resolve() not in source_output.parents or not source_output.is_file():
        raise HTTPException(status_code=404, detail="Selected output was not found")
    if not any((source_output.parent / "tensorrt_decoded").glob("decoded_*.pt")):
        raise HTTPException(status_code=400, detail="This result has no reusable TensorRT decoded batches")
    job_id = uuid4().hex[:10]
    values = {"seed": seed, "sharpen_enabled": _bool(sharpen_enabled), "sharpen_strength": sharpen_strength,
              "grain_enabled": _bool(grain_enabled), "grain_intensity": grain_intensity, "grain_saturation": grain_saturation,
              "microtexture_enabled": _bool(microtexture_enabled), "microtexture_strength": microtexture_strength,
              "skin_finishing_enabled": _bool(skin_finishing_enabled), "skin_evenness": skin_evenness, "skin_smoothing": skin_smoothing,
              "skin_redness": skin_redness, "skin_shine": skin_shine, "blemish_mode": blemish_mode, "preserve_marks": _bool(preserve_marks),
              "seam_mode": seam_mode, "seam_frames": seam_frames, "color_correction": color_correction,
              "face_model": face_model, "face_strength": face_strength, "face_fidelity": face_fidelity, "face_detail": face_detail, "face_min_size": face_min_size}
    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "message": "Queued for post-only reprocess", "job_type": "reprocess", "resumable": False,
                        "_source": str(source_output), "_job_dir": str(source_output.parent), "_values": values}
    Thread(target=_reprocess_job, args=(job_id, source_output, values), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {key: value for key, value in job.items() if not key.startswith("_")}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.get("resumable"):
            raise HTTPException(status_code=400, detail="This job was not configured for resumable chunk rendering")
        if job.get("status") not in {"error", "cancelled"}:
            raise HTTPException(status_code=409, detail="Only a failed or cancelled job can be resumed")
        source = Path(str(job["_source"]))
        job_dir = Path(str(job["_job_dir"]))
        values = dict(job["_values"])
        job["status"] = "queued"
        job["message"] = "Resuming from the last completed chunk"
        job["error"] = None
        job["failure_reason"] = None
    Thread(target=_render_job, args=(job_id, source, job_dir, values), daemon=True).start()
    return {"ok": True, "id": job_id}


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = Path(str(job.get("_job_dir", ""))) / "render.log"
    if not log_path.is_file():
        raise HTTPException(status_code=404, detail="No render log has been saved yet")
    return FileResponse(log_path, media_type="text/plain", filename=f"seedvr-{job_id}.log")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
    message = cancel_current_render()
    with JOBS_LOCK:
        started_at = JOBS[job_id].get("started_at")
    elapsed = time.time() - float(started_at) if started_at else 0.0
    _update(job_id, status="cancelled", message=message, elapsed_seconds=elapsed)
    return {"ok": True, "message": message}


@app.get("/api/update/check")
def update_check() -> dict[str, object]:
    if os.name != "nt":
        return {"supported": False, "update_available": False,
                "message": "Automatic updates are Windows-only. Update your Git checkout manually, then rerun scripts/install.sh."}
    return check_for_updates(ROOT)


@app.post("/api/update/apply")
def apply_update() -> dict[str, object]:
    if os.name != "nt":
        raise HTTPException(status_code=409, detail="Automatic updates are Windows-only. Update your Git checkout manually, then rerun scripts/install.sh.")
    with JOBS_LOCK:
        active_render = any(job.get("status") in {"queued", "running"} for job in JOBS.values())
    if active_render:
        raise HTTPException(status_code=409, detail="Finish or stop the active render before updating.")
    status = check_for_updates(ROOT)
    if not status["supported"]:
        raise HTTPException(status_code=409, detail=str(status["message"]))
    if not status["update_available"]:
        raise HTTPException(status_code=409, detail="SeedVR Studio is already up to date.")
    try:
        launch_updater(ROOT, os.getpid())
    except UpdateError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    def stop_for_update() -> None:
        time.sleep(0.75)
        os.kill(os.getpid(), signal.SIGTERM)
    Thread(target=stop_for_update, daemon=True).start()
    return {"ok": True, "message": "Updater started. SeedVR Studio will close and restart automatically."}


@app.post("/api/shutdown")
def shutdown() -> dict[str, object]:
    """Release active inference resources and stop the local JS Studio server."""
    message = cancel_current_render()
    def stop_server() -> None:
        time.sleep(0.35)
        os.kill(os.getpid(), signal.SIGTERM)
    Thread(target=stop_server, daemon=True).start()
    return {"ok": True, "message": message}


@app.post("/api/open-folder")
def open_output_folder(output_path: str = Form(...)) -> dict[str, object]:
    """Open the platform file manager for a workspace video."""
    relative = output_path.removeprefix("/media/").lstrip("/\\")
    candidate = (OUTPUTS / relative).resolve()
    if OUTPUTS.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Selected output was not found")
    command = ["explorer.exe", f"/select,{candidate}"] if os.name == "nt" else ["xdg-open", str(candidate.parent)]
    try:
        subprocess.Popen(command, close_fds=True)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="The desktop file manager could not be opened.") from exc
    return {"ok": True}


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}


def _local_video(path: str) -> Path:
    candidate = Path(path.strip()).expanduser()
    if not path.strip() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Video file was not found: {candidate}")
    if candidate.suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Not a supported video file: {candidate.name}")
    return candidate


@app.post("/api/pick-file")
def pick_file(initial: str = Form("")) -> dict[str, object]:
    """Open the desktop file dialog on this machine and return the chosen video path."""
    try:
        chosen = pick_video_file(initial or None)
    except DialogUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"path": chosen, "cancelled": chosen is None}


@app.get("/api/local-video/info")
def local_video_info(path: str) -> dict[str, object]:
    """Describe a local source video so the browser can preview it before rendering."""
    video = _local_video(path)
    try:
        info = probe(video)
        fps, frames, duration = info.fps, info.frames, info.duration
    except Exception:
        fps, frames, duration = 0.0, 0, 0.0
    return {"name": video.name, "path": str(video), "folder": str(video.parent), "bytes": video.stat().st_size, "fps": fps, "frames": frames, "duration": duration}


@app.get("/api/local-video")
def local_video(path: str) -> FileResponse:
    video = _local_video(path)
    return FileResponse(video, media_type=f"video/{'mp4' if video.suffix.lower() in {'.mp4', '.m4v'} else video.suffix.lstrip('.').lower()}")


@app.get("/api/output-thumbnail")
def output_thumbnail(path: str) -> FileResponse:
    """Return a cached small JPEG made from the output video first frame."""
    relative = path.removeprefix("/media/").lstrip("/\\")
    candidate = (OUTPUTS / relative).resolve()
    if OUTPUTS.resolve() not in candidate.parents or not candidate.is_file() or candidate.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="Selected output was not found")
    thumbnail = candidate.with_name(f".{candidate.stem}-thumbnail.jpg")
    if not thumbnail.is_file() or thumbnail.stat().st_mtime < candidate.stat().st_mtime:
        make_thumbnail(candidate, thumbnail)
    return FileResponse(thumbnail, media_type="image/jpeg")


@app.get("/media/{relative_path:path}")
def media(relative_path: str) -> FileResponse:
    candidate = (OUTPUTS / relative_path).resolve()
    if OUTPUTS.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(candidate)


@app.get("/app.js", include_in_schema=False)
def serve_app_js() -> FileResponse:
    # Windows can register .js as text/plain. Module scripts require a
    # JavaScript MIME type, so do not leave this asset to MIME guessing.
    return FileResponse(WEB / "app.js", media_type="application/javascript")


@app.get("/styles.css", include_in_schema=False)
def serve_styles_css() -> FileResponse:
    return FileResponse(WEB / "styles.css", media_type="text/css")


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
