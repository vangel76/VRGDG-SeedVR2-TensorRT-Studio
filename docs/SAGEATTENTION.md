# SageAttention guide

## What it does

SageAttention is an optimized attention backend used during SeedVR2's DiT restoration stage. It can reduce attention time and memory traffic. It does not alter the TensorRT VAE engine, and it is not a separate upscaler or quality model.

The Studio defaults to **SageAttention 2** because it works on a wider range of NVIDIA RTX GPUs. The installer installs SageAttention 1.0.6 and Triton for Windows 3.5.1.post24.

## Choosing a mode

- **SageAttention 2:** recommended default for supported NVIDIA RTX GPUs.
- **SageAttention 3:** intended for Blackwell or RTX 50-series hardware. It needs a separate compatible sageattn3 package and may fall back for variable-length attention.
- **Flash Attention 2 or 3:** alternatives when compatible packages are installed.
- **SDPA:** PyTorch's built-in compatibility backend. It is the safest fallback but is usually slower.

Missing accelerated backends do not invalidate a render. SeedVR2 checks what is importable and follows its fallback chain, eventually using SDPA. The render log records the selected or fallback backend.

## Verify SageAttention 2

### Linux / CachyOS

With the main Studio environment installed, run:

```bash
./scripts/install_sageattention.sh
```

This builds the official SageAttention **2.2.0** tag for the installed GPU and
PyTorch environment. It requires the CUDA toolkit (`nvcc`) and a compatible host
compiler. On this CachyOS setup, GCC 15 and C++20 flags are used with PyTorch 2.14
and CUDA 13.3. The script selects `gcc-15`/`g++-15` when available; `CC` and `CXX`
can override that choice. It leaves model downloads and TensorRT engines alone.

Verify the actual SeedVR2 variable-length attention path:

```bash
.venv/bin/python tools/check_sageattention.py
```

The check executes FP16 and BF16 CUDA kernels and compares their output with
SDPA. Select **SageAttention 2** in Studio for subsequent renders. Each render
starts a fresh inference process, so an already-running render keeps its current
attention backend. This accelerates attention, not TensorRT VAE decoding.

The UI label selects `sageattn_varlen`, the package's Triton implementation;
installing version 2.2.0 does not make this integration use every SageAttention2++
CUDA kernel advertised upstream.

### Windows

From the repository folder:

~~~powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'vendor/seedvr2'); from src.optimization.compatibility import SAGE_ATTN_2_AVAILABLE; print('SageAttention 2 ready:', SAGE_ATTN_2_AVAILABLE)"
~~~

The result should be **True**.

For a fuller check:

~~~powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -c "import torch, sageattention; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.version.cuda); print('SageAttention:', sageattention.__file__)"
~~~

## Repair it

Close the Studio and run:

~~~powershell
.\scripts\install.ps1 -Repair
~~~

That reinstalls the known-good Windows CUDA stack and reruns verification. If the exact tested PyTorch nightly is no longer available, the installer uses the newest mutually compatible CUDA 13 nightly and reports that fallback.

## Reading the render log

At model startup, look for the optimization line showing SageAttention as available. If the requested mode cannot load, the log explains why and identifies the fallback. Selecting SageAttention in the UI is a request; the runtime availability check determines what is actually used.

Do not install random SageAttention wheels built for a different Python, PyTorch, CUDA, or GPU combination into this environment. CUDA extensions must match the runtime stack.
