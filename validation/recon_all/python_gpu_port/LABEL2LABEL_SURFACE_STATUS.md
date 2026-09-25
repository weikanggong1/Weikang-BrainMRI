# Fixed fsaverage surface-label transfer

The frozen FreeSurfer 8.2.0 recon-all log contains 72 calls of `mri_label2label --srcsubject fsaverage --trgsubject fs_sub01 --regmethod surface`: 36 labels per hemisphere. These include BA/V1/V2/MT/entorhinal/perirhinal labels, eight FG/hOc `mpm.vpnl` labels, and the 14 thresholded labels per side. The input contract is the source `fsaverage/label/*.label`, source `fsaverage/surf/?h.sphere.reg`, target `fs_sub01/surf/?h.sphere.reg`, and target `fs_sub01/surf/?h.white`. Each output is the same-named label in `fs_sub01/label`.

`SurfaceLabelMapper` implements the fixed surface call: rescale each sphere to radius 100 using FreeSurfer's bounding-box-centered mean radius; map source label points to nearest target sphere vertices in source-label order; then visit as-yet-unmapped target vertices in vertex order and include those whose nearest source sphere vertex lies in the source label. It retains the first occurrence of duplicate target vertices and carries source statistics as float32 into native label text. The mapper caches the two sphere nearest-vertex arrays for all 36 labels of one hemisphere. Native uses a spatial hash; this SciPy nearest-neighbor replacement was checked against its ordered output on the frozen T1.

## Exact output checks

Freshly replayed native `mri_label2label` on headcw and Python both match the frozen official files byte for byte for representative `lh.FG1.mpm.vpnl.label` (SHA-256 `02df9ccb88212f18f9e5be7e9b8ed2e0351804465a661481686c2c422e3b8054`) and `rh.BA1_exvivo.label` (SHA-256 `0bc04fd8fcdf94d3a9e858421b14fde584664a63541f51414282eb4e442557b3`). The [batch validator](benchmark_label2label_surface.py) then reproduced **72/72** frozen official label files byte for byte, including vertex order, coordinates, statistic text, header, and filename. Its ordered-map regression test passed on headcw (`1 passed`).

The native log's `Writing label file ... 443` for LH FG1 is a pre-write count. FreeSurfer flags duplicate target vertices during writing; the actual native and official files each contain 343 data rows. The Python output has the same 343 rows in the same order.

## Time and external assets

| Measurement | Native FreeSurfer | Python |
|---|---:|---:|
| LH FG1, same headcw CPU, independent call | 3.17 s | 0.53 s |
| RH BA1, same headcw CPU, independent call | 2.94 s | 0.61 s |
| All 72, Python with hemisphere maps reused on headcw | — | 7.90 s wall; 5.711 s of per-label calls, median 0.053 s |
| All 72 native calls in the original gpucw1 log | 236.24 s summed; median 3.245 s | — |

The original 72-call native sum is from gpucw1, a different host from the Python batch run; only the two independent calls above are paired on headcw. Hemisphere mapper setup took 0.264 s LH and 0.287 s RH. These are CPU timings, and this module is not yet connected to the main recon-all entrypoint.

The external fsaverage inputs used by this fixed stage total **21,196,583 bytes** (74 files: two `sphere.reg` surfaces and 72 source labels). They can be supplied as a small separately downloaded template directory; FreeSurfer binaries or its full fsaverage tree are not needed by this Python stage. Target `sphere.reg` and `white` come from the reconstructed subject.

```bash
PYTHONPATH=src python validation/recon_all/python_gpu_port/benchmark_label2label_surface.py \
  /path/to/fsaverage /path/to/fs_sub01 /path/to/scratch/labels
```
