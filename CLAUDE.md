# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Also read `AGENTS.md` (project structure, style, commit/PR conventions). This file focuses on commands and the cross-file render architecture.

## Commands

Run from the repository root using the project venv. `python` below means `.venv/bin/python` (Linux) or `.\.venv\Scripts\python.exe` (Windows). Never use the system Python: `seedvr_studio/paths.py` hardcodes `VENV_PYTHON` and every render stage is spawned with it.

```bash
# Setup (Linux/CachyOS; Windows uses the .bat / scripts/*.ps1 equivalents)
./scripts/install.sh                 # venv + CUDA 13 torch + deps + models + 4 TensorRT engines
./scripts/install.sh --skip-tensorrt # PyTorch-only path; select "SeedVR2 (Legacy)" in the UI
./scripts/install_sageattention.sh   # optional SageAttention 2 build (needs nvcc)

# Run
./run.sh                     # uvicorn api_server:app on http://127.0.0.1:7870
python -m uvicorn api_server:app --host 127.0.0.1 --port 7870 --reload   # same, with reload
python app.py                        # legacy Gradio UI on :7860 (seedvr_studio/ui.py)

# Checks
python scripts/verify_install.py --allow-sdpa   # deps, CUDA, ffmpeg, engines; no render
python tools/check_sageattention.py             # SageAttention kernel parity check
uv pip check --python .venv/bin/python
shellcheck scripts/install.sh run.sh

# Tests (stdlib unittest, files in tools/test_*.py)
python -m unittest tools.test_frame_rate tools.test_updater tools.test_bootstrap_updater tools.test_linux_support tools.test_source_path tools.test_batch tools.test_face_restore
python -m unittest tools.test_frame_rate.ResolveFrameRateTests.test_override_is_used_when_detection_fails   # single test
python tools/test_vae_encoder_parity.py         # GPU parity; needs models + engines

# Browser viewer test (Studio must be running; deps kept out of the app venv)
uv run --no-project --python 3.12 --with playwright playwright install chromium
uv run --no-project --python 3.12 --with playwright python tools/test_viewer.py
```

Windows-specific tests (PowerShell updater cases) skip on Linux. No linter or formatter is configured. The frontend in `web/` is plain JS with no build step; `api_server.py` sends `no-store` for it, so a browser refresh picks up edits.

## Architecture: how a render flows

Everything is process-based. The FastAPI server never imports the model; it spawns child processes with `VENV_PYTHON` and parses their stdout for progress.

1. **`web/app.js`** POSTs a multipart form to `/api/jobs` and polls `/api/jobs/{id}`. Job state lives only in the in-memory `JOBS` dict in `api_server.py` (keys starting with `_` are private and not returned). Restarting the server forgets jobs, but output folders persist.
2. **`api_server._render_job`** (daemon thread) creates `outputs/js-{preview|full}-{id}/`, copies the source to `source.*` (from the upload, or from the optional `source_path` form field when it names a local file), applies output preset / crop policy, writes `job-manifest.json`, and appends `render.log`. Preview jobs clip the source first (`restored.mp4`); full jobs write `<original stem>-seed.mp4` in the job folder and, when `source_path` gave a file or folder, copy it beside the source as `<stem>-seed.mp4` (never overwriting, `-seed-2` etc.), reported as `saved_path`. Full jobs with `chunked_render` go through `_render_chunked`. Job folders are temporary: `cleanup_workspace` (startup event and `POST /api/workspace/clear`, called by the frontend on every new project) deletes `outputs/js-*` except running jobs and undelivered full renders (a `*-seed.mp4` or legacy `*-restored.mp4` with no `delivered.json` marker).
3. **`seedvr_studio.backend.render`** builds the `vendor/seedvr2/inference_cli.py` command. Backend name decides the path:
   - **"RTX Video Super Resolution"**: no SeedVR2 at all. `backend.render_rtx_vsr` runs `tools/run_rtx_vsr.py` (NVIDIA Video Effects SDK via the `nvidia-vfx` wheel from `https://pypi.nvidia.com/`, progress lines `RTX_PROGRESS n/m`). Without post effects it streams ffmpeg → VSR → ffmpeg straight to the MP4; with any post effect it writes `tensorrt_decoded/decoded_*.pt` batches in the TensorRT layout and hands them to `tensorrt_pipeline.postprocess_and_assemble` (seam mode forced off), so reprocess works too. Factors above 4x chain passes.
   - **"SeedVR2 (Legacy)"**: the vendored CLI does encode, DiT, decode, and writes the MP4 itself.
   - **"SeedVR2 + TensorRT"** (batch size must be 5 or 21): the CLI runs with `--stop_before_vae` and env vars `SEEDVR2_TRT_ENCODER[_LEGACY|_FAST]` and `SEEDVR2_LATENT_CAPTURE_DIR`, which the vendored `src/core/infer.py` and `generation_phases.py` read to use the TensorRT VAE encoder and dump `vae_latents/vae_latent_*.pt`. Control then passes to `tensorrt_pipeline.decode_postprocess_and_assemble`.
4. **`seedvr_studio/tensorrt_pipeline.py`** picks a decoder engine per latent batch (21-frame or 5-frame `.rtxplan`), decodes via `tools/run_tensorrt_persistent.py` (one long-lived process, "optimized"; progress lines `PERSISTENT_PROGRESS n/m`) or `tools/run_tensorrt_tiled.py` (one process per batch, "stable"). Optimized failure falls back to stable automatically and records why in `decoder-manifest.json`. Decoded batches land in `tensorrt_decoded/decoded_*.pt`, then optionally `tools/postprocess_tensor_video.py` (color correction, face restoration, sharpen, grain, microtexture, skin) run **once per render** via `seedvr_studio/postprocess_runner.py` with a `--jobs` manifest so GPU models load a single time (progress lines `POSTPROCESS_PROGRESS n/m`), then `tools/assemble_tensor_video.py` writes the MP4. Color correction never runs inside SeedVR2 on this path (the CLI exits before its phase 4), so the postprocess tool reuses the vendored `src/utils/color_fix.py` functions with source frames read via ffmpeg (`--source`, `--frame-start`) as reference.
5. **Post-only reprocess** (`/api/reprocess` → `backend.reprocess_tensorrt`) reruns step 4's postprocess + assemble on the saved `decoded_*.pt` files without touching SeedVR2. A job is `reprocessable` only if those files exist, so do not delete `tensorrt_decoded/` after a render.

**Face restoration** (`tools/face_restore.py`, called from the postprocess tool with `--face-model codeformer|gfpgan`): RetinaFace detection + IoU tracking with landmark smoothing, FFHQ 512 alignment, restorer from `vendor/faces/` (CodeFormer and GFPGAN archs with `basicsr` replaced by `vendor/faces/_compat.py`), parsed-mask paste-back with a wavelet split (low band from the restorer, `--face-detail` share of high band from SeedVR2). Track state crosses batches via `--face-state` JSON, reset per render. Weights (~900 MB) live in `models/faces/`, downloaded by `scripts/download_models.py` or on first use. `facexlib` is the only new dependency; all blur kernels in the postprocess tool scale with frame height via `_kernel()` (720p reference).

**Batch mode** (`seedvr_studio/batch.py`, endpoints under `/api/batch*`): `BatchQueue` holds an ordered item list persisted to `outputs/batch-queue.json` and a worker thread that calls `api_server._batch_runner` per item. The runner reuses `_prepare_job` + `_render_job` (job_type `full`) with the render settings snapshotted at add time (server fills `RENDER_DEFAULTS` for missing keys) and delivers to `<source folder>/VRUpscale/<stem>-seed.mp4`; items whose output already exists are added as `skipped`. Single renders return 409 while a batch runs, and batch start refuses while a manual job runs. Native pickers for files/folders live in `seedvr_studio/dialogs.py` (kdialog/zenity/yad/tkinter/PowerShell).

**Chunked / resumable renders** (`_render_chunked`): the source is split into fixed-second chunks with a `batch_size - 1` frame overlap, each rendered through `backend.render`, overlap trimmed, and progress written to `chunk-manifest.json`. `/api/jobs/{id}/resume` re-enters `_render_job` with the same values and skips chunks listed as completed. Changing chunk length or overlap invalidates the manifest.

**Progress reporting** is regex parsing of child stdout: `backend._progress_from_log` maps vendored CLI phase lines to fractions (encode 0.12, DiT 0.28, decode 0.78, post 0.90). If you change log strings in `vendor/seedvr2` or the TensorRT tools, update these regexes, the `PERSISTENT_PROGRESS` match in `persistent_decoder.py`, and the `POSTPROCESS_PROGRESS` match in `postprocess_runner.py`.

**Cancellation** (`seedvr_studio/cancellation.py`) sets a threading Event and uses psutil to kill any process whose cmdline contains `inference_cli.py`, one of the four TensorRT tool scripts, or `run_rtx_vsr.py`. New render-stage scripts must be added to that list or Cancel will leave them running.

**Platform split**: `paths.VENV_PYTHON`, `subprocess` `creationflags`, `open-folder` (`explorer` vs `xdg-open`), and the Git-based updater (`seedvr_studio/updater.py`, Windows-only, returns `supported: False` elsewhere) all branch on `os.name`. Keep both branches working; `tools/test_linux_support.py` guards the Linux side.

## TensorRT artifacts

`scripts/prepare_tensorrt.py` exports four fixed-shape ONNX graphs via `tools/export_vae_{encoder,decoder}.py` and builds them with `tools/build_tensorrt_engine.py` into `tensorrt_backend/artifacts/`:

| Engine | Shape |
|---|---|
| `vae_encoder_5f_tile512` | 512×512, 5 frames |
| `vae_encoder_21f_tile512` | 512×512, 21 frames |
| `vae_decoder_tile_512_5f` | 512×512, 2 latent frames |
| `vae_decoder_tile_256_21f` | 256×256, 6 latent frames |

`backend.tensorrt_status` and `scripts/verify_install.py` require all four (each > 1 MB). Engines are GPU- and TensorRT-version-specific and gitignored; a missing or foreign `.rtxplan` means "rerun the installer", not "copy from elsewhere". Model weights live in `models/SEEDVR2/` and are mapped from UI labels by `backend.MODEL_FILES`.

## Vendored SeedVR2

`vendor/seedvr2/` is a runtime subset of ComfyUI-SeedVR2_VideoUpscaler (Apache-2.0) with Studio-specific TensorRT hooks gated behind the `SEEDVR2_TRT_*` env vars above. Keep edits there minimal, keep them env-gated so the upstream CLI behaviour is unchanged by default, and preserve `LICENSE`, `UPSTREAM.md`, and `THIRD_PARTY_NOTICES.md`.
