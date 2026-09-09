#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

skip_models=0
skip_tensorrt=0
for argument in "$@"; do
    case "$argument" in
        --skip-models) skip_models=1 ;;
        --skip-tensorrt) skip_tensorrt=1 ;;
        --help)
            echo 'Usage: scripts/install.sh [--skip-models] [--skip-tensorrt]'
            exit 0 ;;
        *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
done

for program in uv ffmpeg ffprobe nvidia-smi; do
    if ! command -v "$program" >/dev/null; then
        echo "Missing $program. On CachyOS, install uv, ffmpeg, and the NVIDIA driver before setup." >&2
        exit 1
    fi
done
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi
python=.venv/bin/python
uv pip install --python "$python" torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$python" --torch-backend cu130 --extra-index-url https://pypi.nvidia.com/ -e . -r vendor/seedvr2/requirements.txt -r requirements-tensorrt.txt
"$python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
if (( ! skip_models )); then
    "$python" scripts/download_models.py
fi
if (( ! skip_tensorrt )); then
    "$python" scripts/prepare_tensorrt.py
    "$python" scripts/verify_install.py --allow-sdpa
fi
echo 'Setup complete. Launch with ./run.sh. Attention uses the built-in SDPA fallback unless SageAttention is installed separately.'
