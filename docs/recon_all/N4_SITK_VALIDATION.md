# Isolated N4 correction replay

FreeSurfer 8.2 runs `AntsN4BiasFieldCorrectionFs -i orig.mgz -o nu0.mgz --dtype uchar` during `mri_nu_correct.mni`. Its wrapper uses the ITK N4 filter with four fitting levels of 50 iterations, convergence threshold 0, shrink factor 4, an all-voxel mask when no mask is supplied, and one thread. These settings were checked against the completed run log and [FreeSurfer source commit d932c45](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/AntsN4BiasFieldCorrectionFs/AntsN4BiasFieldCorrectionFs.cpp). The Python entry [`n4_sitk.py`](../../src/fnit/recon_all/n4_sitk.py) replays this **isolated `orig.mgz` → `nu0.mgz` step** through SimpleITK 2.5.6. It uses FreeSurfer's uchar clipping and `floor(x + 0.5)` rounding.

## Real-input check

The check used the completed sub01 256³ uint8 `orig.mgz` on headcw. The input MRI itself is not included in this repository. The installed official executable was FreeSurfer 8.2.0-1, SHA-256 `9d77c96f6462e0f0f46a42b3ed59d7ed7364318d2006fb31dff96a953929d239`. Both runs used one CPU thread and wrote `nu0.mgz` from the same input. The Python replay converts SimpleITK's division result back to float32 before uchar rounding, matching FreeSurfer's ITK image type.

| Item | Official FreeSurfer | Python/SimpleITK |
| --- | ---: | ---: |
| Wall time | 168.26 s | 96.14 s |
| Output MGZ file SHA-256 | `4db513d79a7f64cc72a86b93d093f332f2fc13f63d4f3602d5ca18ca1a4c5b3c` | `5b4ff78c1629dfb4f20abd18dfda4c4dbfe462f185e5f328552fa9a4d05aa299` |
| Decoded voxel SHA-256 (C-order uint8) | `ef6bd8aff016332d3e317a743358dce8f92bc63d26d82a5498dbd689647cf982` | `ef6bd8aff016332d3e317a743358dce8f92bc63d26d82a5498dbd689647cf982` |
| Decompressed MGH SHA-256 | `8fe2938cf0d94b38019758c1c25c79b27783da4f71ca89fc28a4465c2d6e49aa` | `e4510816373b619b01b4e2f0b3cb94c69a02b8957c771bc06598bbbc1cbe88a4` |

Input MGZ file SHA-256: `d79723f94bfc149ff36c89094a3d734b888a03cecbc32dc57a22992e8a5e817f` (2,038,466 bytes); decoded voxel SHA-256: `84da8a990ef60c6ba30a4ddfbaba597a90e32bca720f96c6d741fcb08c504825`.

The output arrays each have 256³ uint8 voxels and identical affine transforms. **All 16,777,216 decoded voxels match exactly** on this input. The 284-byte MGH headers also match. The complete decompressed files differ: the official footer is 1,345 bytes, while the Python output preserves the input's 1,346-byte footer through `mgh_compat.py`. The N4 footer uses an XFORM tag and does not match the SynthSeg color-table normalization rule used elsewhere in this package. Thus voxel and affine parity here does not imply identical MGH metadata or downstream cortical metric parity. Earlier SimpleITK replays took 124.65–125.58 seconds; the reported 96.14-second final run was faster on the same host, without a controlled explanation for the timing variation.

SimpleITK's downloaded wheel was 52.8 MB and its temporary installation occupied 265 MB. It calls compiled ITK code on CPU, so this result removes the need for the FreeSurfer N4 executable at this step but does **not** complete a GPU N4 implementation. The separate experimental Torch approximation in `n4_gpu.py` did not pass this check (foreground MAE 5.434 and RMSE 6.801 versus official `nu0.mgz`) and is not used by recon-all.

The upstream `mri_nu_correct.mni` continues after `nu0.mgz`: it scales intensities to restore the global mean, converts with the original geometry, and calls `mri_make_uchar` using the Talairach transform. The subsequent Python wrapper and a fresh full-stage byte comparison are documented in [N4_WRAPPER_VALIDATION.md](N4_WRAPPER_VALIDATION.md). Subsequent segmentation and surface measures have not been assessed for this replacement, and it is not connected to the recon-all entry point.

## Replay

With `SimpleITK==2.5.6`, `nibabel`, and NumPy installed:

```bash
python -m fnit.recon_all.n4_sitk --i orig.mgz --o nu0.sitk.mgz
```

Validation logs are retained on headcw in `/tmp/reconall_n4_sub01_20260925/`: `official.log`, `final_sitk.log`, `final_compare.log`, and `pytest_all.log` (five tests passed). The final module source SHA-256 used for the replay was `7d864006979ffe456d1fad8245f9dc1fc3cdaead7f435fd4beef4d853b2eae20`.
