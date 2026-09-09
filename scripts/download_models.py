"""Download and validate the default SeedVR2 model and VAE."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "seedvr2"
sys.path.insert(0, str(VENDOR))

from src.utils.downloads import download_weight  # noqa: E402
from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE  # noqa: E402


def main() -> int:
    model_dir = ROOT / "models" / "SEEDVR2"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading and validating {DEFAULT_DIT} and {DEFAULT_VAE}...")
    if not download_weight(DEFAULT_DIT, DEFAULT_VAE, str(model_dir)):
        print("Model download failed. Run this installer again to resume.", file=sys.stderr)
        return 1
    print(f"Default SeedVR2 models are ready in {model_dir}")
    face_dir = ROOT / "models" / "faces"
    print("Downloading face restoration weights (CodeFormer, GFPGAN, RetinaFace, ParseNet)...")
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        from face_restore import WEIGHTS, ensure_weights  # noqa: E402
        ensure_weights(list(WEIGHTS), face_dir)
        print(f"Face restoration weights are ready in {face_dir}")
    except Exception as exc:  # optional feature: never fail the install for it
        print(f"Face restoration weights could not be downloaded now ({exc}); they download on first use.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
