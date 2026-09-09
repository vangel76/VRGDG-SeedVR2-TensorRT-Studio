# Repository Guidelines

## Project Structure & Module Organization

- `api_server.py` serves the FastAPI API and static browser interface in `web/` (`app.js`, `styles.css`, `index.html`).
- `seedvr_studio/` contains rendering, media handling, jobs, cancellation, updating, and legacy Gradio integration; `app.py` is the legacy entry point.
- `scripts/` contains PowerShell setup/launch/update workflows and Python installation utilities. `bootstrap/` provides the one-time updater.
- `tools/` contains regression tests, TensorRT export/build utilities, and GPU diagnostics. `tensorrt_backend/` holds optional native sources.
- `vendor/seedvr2/` contains the bundled upstream integration. Keep changes there focused and preserve licensing notices.
- `docs/`, `images/`, and `videos/` contain documentation and demonstration assets.

## Build, Test, and Development Commands

Run from the repository root. Windows 11, Python 3.12+, and an NVIDIA RTX GPU are the primary supported setup. Use `.\.venv\Scripts\python.exe` for `python` below.

On CachyOS/Linux, use `scripts/install.sh` and `run.sh`, with `.venv/bin/python`. See `docs/LINUX.md`; preserve both platform paths when editing launch or process code.

- `& '.\Install SeedVR Studio.bat'` (PowerShell): prepare the private environment, dependencies, models, FFmpeg, and GPU-specific engines.
- `& '.\Launch SeedVR Studio Pro.bat'`: launch Studio, checking setup first.
- `python -m uvicorn api_server:app --host 127.0.0.1 --port 7870`: run the API and frontend using the prepared environment. The frontend requires no Node build.
- `python scripts/verify_install.py`: check dependencies, CUDA availability, FFmpeg, and required engines without rendering.
- `python -m unittest tools.test_frame_rate tools.test_updater tools.test_bootstrap_updater`: run regression tests.

## Coding Style & Naming Conventions

Follow nearby code: four-space Python indentation, `snake_case` functions/modules, `PascalCase` classes, and type hints for new interfaces. JavaScript uses two-space indentation, `camelCase`, and semicolons. No repository-wide formatter or linter is configured; avoid unrelated formatting changes.

## Testing Guidelines

Regression tests use standard-library `unittest`, with `tools/test_*.py` files and `test_*` methods. Updater tests require Git; Windows-specific cases require `powershell.exe` and otherwise skip. No coverage threshold is configured. Add focused regression tests for behavior changes. GPU parity and smoke scripts require installed models and matching engines; run relevant diagnostics and a short preview for rendering changes.

Linux checks: `python -m unittest tools.test_linux_support`.

## Commit & Pull Request Guidelines

Recent commits use imperative subjects, such as `Fix first-frame JPEG generation filter`. Keep commits focused. PRs should describe the problem, resulting behavior, validation performed, and relevant issues. Include screenshots for UI changes and GPU/runtime details for rendering changes.

## Local Data & Configuration

Keep `.venv/`, `models/`, `outputs/`, engine artifacts, and caches untracked. Build TensorRT engines locally; do not reuse `.rtxplan` files across different GPUs or TensorRT versions.
