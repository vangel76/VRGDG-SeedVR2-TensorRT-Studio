# Linux setup (CachyOS / Arch)

Use an NVIDIA GPU with a current driver, FFmpeg, and `uv`. The installer creates
a private Python 3.12 environment, independent of the system Python version.
On CachyOS, install missing prerequisites with `sudo pacman -S uv ffmpeg`;
use the distribution's NVIDIA driver setup if `nvidia-smi` does not work.

From the repository root:

```bash
./scripts/install.sh
./run.sh
```

Open <http://127.0.0.1:7870/>. Stop the server with Ctrl+C. No environment
activation is necessary, including when launching from Fish.

Setup installs CUDA 13 PyTorch, the application, vendored SeedVR2 dependencies,
and TensorRT RTX. It validates/downloads the default 3B FP8 model and VAE, then
builds four GPU-specific engines. Allow substantial time and free GPU memory for
the first build. Rerun the installer to resume completed engine profiles.

For the PyTorch render path without engine preparation:

```bash
./scripts/install.sh --skip-tensorrt
```

Select **SeedVR2 (Legacy)** in Studio until all TensorRT engines are ready.
`--skip-models` skips model prefetch; engine preparation can still require weights.
Models belong in `models/SEEDVR2/`; engines belong in `tensorrt_backend/artifacts/`.
Never copy engines from Windows or another GPU/runtime installation.

SageAttention is optional on Linux. Without it, SeedVR2 falls back to PyTorch
SDPA; selecting `sdpa` explicitly avoids the fallback message. The Windows-only
`triton-windows` package must not be installed in this environment.
To build SageAttention 2, run `./scripts/install_sageattention.sh`; see the
[SageAttention guide](SAGEATTENTION.md) for compiler requirements and verification.

The output-folder button uses `xdg-open`. Automatic application updates remain
Windows-only: preserve local modifications, update your Git checkout manually,
and rerun `scripts/install.sh` to refresh dependencies.

## Validation

```bash
.venv/bin/python -m unittest tools.test_frame_rate tools.test_updater tools.test_bootstrap_updater tools.test_linux_support tools.test_source_path tools.test_batch tools.test_face_restore
uv pip check --python .venv/bin/python
shellcheck scripts/install.sh run.sh
.venv/bin/python scripts/verify_install.py --allow-sdpa
```

Windows PowerShell tests skip on Linux. For rendering changes, also run a short
preview with the intended backend. The readiness check requires all four TensorRT
engines; `--allow-sdpa` permits operation without optional SageAttention.

With Studio running, test timeline dragging, clicking, frame stepping, and
single-video playback in Chromium:

```bash
uv run --no-project --python 3.12 --with playwright playwright install chromium
uv run --no-project --python 3.12 --with playwright python tools/test_viewer.py
```

These commands keep browser-test dependencies separate from the app environment.
Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE` to reuse an existing Chromium binary.

Installation references: [PyTorch](https://pytorch.org/get-started/locally/)
and [NVIDIA TensorRT RTX](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/installing.html).
