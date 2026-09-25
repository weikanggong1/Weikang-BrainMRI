# White/gray contrast projection: matched frozen subject

`vol2surf_contrast_python.py` implements the fixed `pctsurfcon` sampling path
without a FreeSurfer executable: white normal distance -1 mm, gray at 0.3
times vertex thickness, header registration, trilinear interpolation,
cortex-label masking, and `mri_concat --paired-diff-norm --mul 100`. The image
sampling uses PyTorch CPU or CUDA tensors. Small 4×4 geometry operations use
NumPy float32 in the pinned FreeSurfer/VNL operation order. Summing the
matrix–point products in x, y, z, translation order was necessary for exact
native values; equivalent float64 affine composition was insufficient.

On the frozen `fs_sub01` native `rawavg.mgz`, `orig.mgz`, white surfaces,
thickness maps and cortex labels, fresh native `mri_vol2surf` calls and the
Python implementation matched **every left and right intermediate sample**.
The final percentage maps also matched the archived official values for
**every vertex**:

| Output | Vertices | Unequal float32 values | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| Left white sample | 106,622 | 0 | 0 |
| Left gray sample | 106,622 | 0 | 0 |
| Right white sample | 105,541 | 0 | 0 |
| Right gray sample | 105,541 | 0 | 0 |
| Left white/gray percentage | 106,622 | 0 | 0 |
| Right white/gray percentage | 105,541 | 0 | 0 |

The left and right percentage arrays have SHA-256
`70ad294998e50b67527976db9e8353e097f2c6649f39f58c499362e05ced84be`
and `76389c92e12476849e2beccc095ff62923af9e93f03077d44234f31da3fa1f56`.
The MGH binary headers matched their native counterparts for the percentage
maps. Both the image headers and float32 values matched for all four native
intermediates after preserving the source field of view. The official
percentage files have 611 additional provenance bytes at the end; their
common file prefix is identical to the Python output. This footer does not
affect the image data or geometry. Feeding the two Python-generated maps to
`segstats_surface_snr_python.py` reproduced **35/35 SNR data lines per side**
byte for byte against the archived native tables.

One same-host headcw CLI pair per intermediate, including process startup,
input reads and output writes, took:

| Fixed sample | Native CLI | Python CPU CLI |
| --- | ---: | ---: |
| Left white | 0.587 s | 2.512 s |
| Left gray | 0.615 s | 1.802 s |
| Right white | 0.813 s | 2.348 s |
| Right gray | 0.626 s | 2.056 s |

The warm in-process Python `sample_contrast` API took 0.376/0.447 s for left
white/gray and 0.426/0.430 s for right white/gray. The in-process final
percentage calculation took 0.995 s left and 0.967 s right. These pairs do
not measure the complete reconstruction. The CLI includes PyTorch import time.
On `gpucw1` H100 GPU 1, CUDA CLI generation of the final left/right maps took
5.15/4.49 s, respectively; both maps matched every official float32 value
and the MGH header. Those GPU timings are on a different host from the CPU
pairs above and are not a paired speed comparison.

Input SHA-256 for this comparison:

| `fs_sub01` input | SHA-256 |
| --- | --- |
| `mri/rawavg.mgz` | `2d9fb24d64cbcd9183489ad2ea1e4e93da9d3970e52e645c5280229b98a44e19` |
| `mri/orig.mgz` | `7dde820d02968c9fe18056a9cc5cc776c1a6395c48dc4d3a47eb6ba9c399f518` |
| `surf/lh.white` | `9c88a786c4a571c07d830b83fe647461b09fdacb63ff143670771340ef50b51c` |
| `surf/rh.white` | `5c7fda4367dd3624de0ef265303dab49579aa603aee75f1cce2f9c30bec965b9` |
| `surf/lh.thickness` | `cd29acd6e1c14279fd0d25ee59bd464d88a3861a730d418f57a361cc8529935d` |
| `surf/rh.thickness` | `2eb78e314ee6e230f0ee35086a41e91be2c13268a871ac02bdbdf108207e85a0` |
| `label/lh.cortex.label` | `524245d2c800a7ff3fc8ba1d1fe04c02d09354bcc29e99cf3f64182c2f312506` |
| `label/rh.cortex.label` | `609cce84a6a0488c3333af213639ab55bc0312a95485baddf5f4d58e82fd6630` |

```bash
python -m fnit.recon_all.vol2surf_contrast_python \
  /path/to/subjects/fs_sub01 lh pct /path/to/lh.w-g.pct.mgh --device cpu
```

The upstream white surfaces, thickness maps and cortex labels are frozen
official inputs here. This stage has passed its output gate, while a complete
native-free subject run still depends on porting their earlier producers.
