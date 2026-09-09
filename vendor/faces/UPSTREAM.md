# Vendored face restoration architectures

- `codeformer/`: `codeformer_arch.py`, `vqgan_arch.py` from https://github.com/sczhou/CodeFormer (S-Lab License 1.0, non-commercial; see LICENSE). Weights `codeformer.pth` are downloaded at install time into `models/faces/`, never redistributed.
- `gfpgan/`: `gfpganv1_clean_arch.py`, `stylegan2_clean_arch.py` from https://github.com/TencentARC/GFPGAN (Apache-2.0; see LICENSE). Weights `GFPGANv1.4.pth` downloaded at install time.

Only change: `basicsr` imports replaced by `vendor/faces/_compat.py` so no BasicSR dependency is needed. Face detection/alignment comes from the `facexlib` package (MIT).
