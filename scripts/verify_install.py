"""Fast, non-rendering readiness check used by the launcher and installer."""

from __future__ import annotations

import importlib
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENGINES = (
    "vae_decoder_tile_256_21f.rtxplan",
    "vae_decoder_tile_512_5f.rtxplan",
    "vae_encoder_5f_tile512.rtxplan",
    "vae_encoder_21f_tile512.rtxplan",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-sdpa", action="store_true", help="Allow PyTorch attention without optional SageAttention")
    args = parser.parse_args()
    failures: list[str] = []
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            failures.append(f"{executable} is not on PATH")
    if not (ROOT / "vendor" / "seedvr2" / "inference_cli.py").exists():
        failures.append("vendored SeedVR2 source is missing")
    modules = ["torch", "fastapi", "uvicorn", "tensorrt_rtx", "onnx"]
    if not args.allow_sdpa:
        modules.append("sageattention")
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"cannot import {module}: {exc}")
    try:
        import torch
        if not torch.cuda.is_available():
            failures.append("PyTorch cannot access an NVIDIA CUDA GPU")
    except Exception:
        pass
    artifacts = ROOT / "tensorrt_backend" / "artifacts"
    for name in REQUIRED_ENGINES:
        path = artifacts / name
        if not path.exists() or path.stat().st_size < 1_000_000:
            failures.append(f"TensorRT engine is missing: {name}")
    if failures:
        print("Installation is incomplete:")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("SeedVR Studio installation is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
