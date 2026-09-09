"""Portable, observable ONNX export helpers for TensorRT engine preparation."""

from __future__ import annotations

import time
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch


def _math_attention_context():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return nullcontext()


@contextmanager
def _portable_convolutions():
    # The VAE workaround calls aten::cudnn_convolution directly, which has no
    # ONNX symbolic. Use ordinary Conv3d during export, retaining cuDNN's
    # memory-efficient implementation instead of allocating an im2col buffer.
    causal = sys.modules.get("src.models.video_vae_v3.modules.causal_inflation_lib")
    original = getattr(causal, "NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND", None)
    if original is not None:
        causal.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = False
    try:
        yield
    finally:
        if original is not None:
            causal.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = original


def _portable_export(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: Path, *, legacy: bool) -> None:
    is_encoder = args[0].ndim == 5 and args[0].shape[1] == 3
    with torch.inference_mode(), _portable_convolutions(), _math_attention_context():
        torch.onnx.export(
            module,
            args,
            str(output),
            input_names=["video"] if is_encoder else ["latent"],
            output_names=["latent_raw"] if is_encoder else ["sample"],
            opset_version=20,
            dynamo=not legacy,
            optimize=False,
            do_constant_folding=False,
        )


def export_portable_onnx(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output: Path,
    *,
    legacy: bool,
) -> None:
    """Export on CUDA with portable operators, falling back to CPU only on OOM."""
    print(f"Exporting ONNX (legacy tracer, portable convs) -> {output}", flush=True)
    started = time.perf_counter()
    try:
        _portable_export(module, args, output, legacy=legacy)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("CUDA ONNX export ran out of memory; falling back to CPU export. This may take a while.", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _portable_export(module.cpu(), tuple(value.cpu() for value in args), output, legacy=legacy)
    print(f"ONNX export finished in {time.perf_counter() - started:.1f}s", flush=True)
