#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run scripts/install.sh first.' >&2
    exit 1
fi
for program in uv nvcc; do
    if ! command -v "$program" >/dev/null; then
        echo "Missing $program. Install uv and the CUDA toolkit before building SageAttention." >&2
        exit 1
    fi
done

# CUDA's host compiler must be compatible with the installed toolkit.
# CachyOS provides GCC 15 alongside its newer system compiler.
sage_cc="${CC:-gcc}"
sage_cxx="${CXX:-g++}"
if [[ -z ${CC:-} ]] && command -v gcc-15 >/dev/null; then
    sage_cc=gcc-15
fi
if [[ -z ${CXX:-} ]] && command -v g++-15 >/dev/null; then
    sage_cxx=g++-15
fi
sage_arch=$(.venv/bin/python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
uv pip install --python .venv/bin/python packaging wheel ninja
env CC="$sage_cc" CXX="$sage_cxx" CUDAHOSTCXX="$sage_cxx" \
    CXX_APPEND_FLAGS='-std=c++20' NVCC_APPEND_FLAGS='-std=c++20 --threads=2' \
    MAX_JOBS=4 EXT_PARALLEL=1 TORCH_CUDA_ARCH_LIST="$sage_arch" \
    uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
    'sageattention @ git+https://github.com/thu-ml/SageAttention.git@eb615cf6cf4d221338033340ee2de1c37fbdba4a'
.venv/bin/python tools/check_sageattention.py
