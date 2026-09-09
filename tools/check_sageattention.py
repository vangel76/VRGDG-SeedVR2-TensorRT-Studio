"""Check SeedVR2's SageAttention path against SDPA on small CUDA tensors."""

from importlib.metadata import version
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/seedvr2"))
from src.optimization.compatibility import SAGE_ATTN_2_AVAILABLE, call_sage_attn_2_varlen


def main() -> None:
    if not SAGE_ATTN_2_AVAILABLE:
        raise RuntimeError("SeedVR2 cannot import SageAttention 2")
    print(f"SageAttention {version('sageattention')}; GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)
    lengths = (128, 192)
    cumulative = torch.tensor([0, 128, 320], device="cuda", dtype=torch.int32)
    for dtype in (torch.float16, torch.bfloat16):
        q, k, v = [torch.randn(320, 4, 128, device="cuda", dtype=dtype) for _ in range(3)]
        with torch.inference_mode():
            actual = call_sage_attn_2_varlen(q, k, v, cumulative, cumulative, max(lengths), max(lengths))
            references = []
            offset = 0
            for length in lengths:
                inputs = [tensor[offset:offset + length].transpose(0, 1).unsqueeze(0) for tensor in (q, k, v)]
                references.append(F.scaled_dot_product_attention(*inputs).squeeze(0).transpose(0, 1))
                offset += length
            reference = torch.cat(references)
            torch.cuda.synchronize()
        if actual.shape != reference.shape or not torch.isfinite(actual).all():
            raise RuntimeError("SageAttention returned invalid output")
        similarity = F.cosine_similarity(actual.float().flatten(), reference.float().flatten(), dim=0).item()
        if similarity < 0.99:
            raise RuntimeError(f"SageAttention deviates unexpectedly from SDPA: cosine={similarity:.6f}")
        print(f"{dtype}: variable-length CUDA execution passed; SDPA cosine similarity {similarity:.6f}")


if __name__ == "__main__":
    main()
