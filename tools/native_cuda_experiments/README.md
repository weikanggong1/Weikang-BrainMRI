# Surface gradient smoothing CUDA experiment

This isolated experiment implements the ordered float32 first-ring averaging in
FreeSurfer 8.2's `MRISaverageGradients`. It does not change the recon-all runtime.
Reference: FreeSurfer commit `d932c45b7941662ea380a05efef580568b98d41a`,
[`utils/mrisurf_metricProperties.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp#L9491)
and [`utils/mrisurf_topology.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_topology.cpp#L2576).

## Run on a CUDA development node

```sh
nvcc -O2 --fmad=false --prec-div=true \
  -Xcompiler=-ffp-contract=off,-fopenmp -lgomp \
  -o smooth tools/native_cuda_experiments/smooth.cu
python tools/native_cuda_experiments/make_mesh_case.py \
  work/reconall_reference_gpucw1/fs_sub01/surf/lh.inflated \
  /tmp/lh_smooth_1024.bin --iterations 1024
./smooth /tmp/lh_smooth_1024.bin
```

The GPU `cuda_total_ms` includes allocation, host-to-device transfer, 1024 kernel
launches, device-to-host transfer, and synchronization. `cuda_kernel_ms` covers
the launches only. `cpu1_ms` and `cpu4_ms` cover the same CPU calculation with
one and four OpenMP threads. The speedup denominator is CPU4, and numerical
differences are against CPU4. This kernel experiment does **not** measure or
predict the whole FreeSurfer stage. Warm up the CUDA context and run multiple
trials on an otherwise idle node for a kernel speed claim.

The generator reads a real FreeSurfer triangular surface and rebuilds the
default `mrisCompleteTopology_old` first-ring order: faces in file order and,
for each incident face, the previous then next corner if not already present.
It uses surface coordinates as representative input vectors. These are **not**
the actual optimizer gradients. Supply `--gradient-bin` with tightly packed
little-endian float32 `[original_vertex, xyz]` values from an instrumented
FreeSurfer run to test those gradients. `--rip-vertices` accepts one original
vertex ID per line and removes ripped vertices and their neighbor entries.

The `.bin` file is little-endian: 8-byte `FSGRAD1\0` magic; four uint32 values
for active vertices, CSR edges, iterations, and original vertices; uint32
original vertex IDs; uint32 CSR offsets; uint32 CSR neighbors; then float32
vectors. Its `.bin.json` sidecar records source and case hashes. No source
surface or subject data is added to Git.

This checks one mathematical kernel. Replacing it in `mris_sphere` or
`mris_register` requires rebuilding those executables with a modified FreeSurfer
`utils` library; FreeSurfer links `utils` statically, so `LD_PRELOAD` cannot
reliably replace this internal call. Stage and full recon-all numerical parity
must be checked after integration.
