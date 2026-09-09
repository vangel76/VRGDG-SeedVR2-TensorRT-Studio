from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models" / "SEEDVR2"
VENDOR = ROOT / "vendor" / "seedvr2"
SEEDVR_CLI = VENDOR / "inference_cli.py"
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_workspace() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
