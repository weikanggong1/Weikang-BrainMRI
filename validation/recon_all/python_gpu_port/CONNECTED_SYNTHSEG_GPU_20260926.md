# Connected SynthSeg CUDA comparison against FreeSurfer 8.2

The input was `orig.mgz` generated from the public `sub-01_T1w.nii.gz` by the
native-free Python input chain (SHA-256
`caf74b3a2a7c9929d280ac4db5bd80cc69a59ddbeca54af2884906cd8acf5de6`).
Its 16,777,216 input voxels match the unmodified official subject's `orig.mgz`.
The independently configured SynthSeg 2.0 weight SHA-256 was
`f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e`.
The candidate ran on `gpucw1` H100 GPU 1 with four CPU threads. The official
reference is the archived unmodified FreeSurfer 8.2 `a_official` run on that T1.

The initial candidate used cuDNN TF32. It differed at **166/16,777,216**
segmentation voxels and at up to **89.3775 mm³** among the 33 soft-volume
columns. The other SynthSeg preprocessing, model weights, and postprocessing
were identical to the next trial. This establishes that TF32 cannot be used
for the strict hard-segmentation gate on this fixed input.

With cuDNN TF32 disabled only for SynthSeg, all **16,777,216 hard labels**
matched the official result. The output dtype is `float32`; the affine,
284-byte MGH header, and complete header-plus-voxel payload match. The one
near-tie correction equals the historical hybrid run. All 33 soft-volume
columns were checked individually: 26 differ by at most 0.005 mm³ and all
33 by at most 0.185 mm³ (maximum at CSF; total intracranial differs by
0.175 mm³). The largest relative error is about 1.57 ppm at the left
inferior lateral ventricle. **The CSV fails the existing 0.005 mm³ statistic
gate**; the strict connected SynthSeg stage is therefore not fully accepted.

The official script sums float32 foreground posteriors with NumPy after
restoring the input orientation, then multiplies by the input header's voxel
resolution. The candidate now follows that order. A single additional
FP32 CUDA inference compared four reductions: GPU aligned-grid sum
(464.8 mm³ maximum difference), NumPy aligned-grid sum (41.12125 mm³),
NumPy restored-grid sum with affine determinant (0.09125 mm³), and the
source-correct NumPy restored-grid sum with header resolution (0.185 mm³).
The determinant is 0.99999976158 mm³ while header voxel volume is 1.0 mm³
on this input. Selecting the smaller error from the determinant would change
the official algorithm; the remaining difference needs posterior-level
diagnosis or a stated scientific tolerance.

The two candidate calls took 7.35 s with TF32 and 4.98 s with full float32.
They ran sequentially on a shared GPU, so these observations do not establish
a speed advantage or the cost of disabling TF32. No whole reconstruction was
rerun. The [TF32 report](connected_synthseg_gpu_tf32_official_20260926.json),
[full-FP32 report](connected_synthseg_gpu_fp32_official_20260926.json),
[reduction diagnostic](synthseg_reduction_diagnostic_fp32_20260926.json), and
[replay script](experimental/run_connected_synthseg_gpu_official.sh) preserve
the exact input/output checks. Other subjects remain untested.
