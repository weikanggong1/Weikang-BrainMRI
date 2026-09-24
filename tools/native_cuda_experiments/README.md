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
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/native_cuda_experiments -p 'test_*.py'
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

To test the actual `fs_cuda_average_gradients` C ABI with a FSGRAD1 case:

```sh
nvcc -O2 -std=c++11 -arch=sm_90 --fmad=false --prec-div=true \
  -Xcompiler=-ffp-contract=off \
  -o bridge_fixture tools/native_cuda_experiments/bridge_fixture.cpp \
  tools/native_cuda_experiments/fs_cuda_average_gradients.cu
./bridge_fixture /tmp/lh_smooth_1024.bin
```

`bridge_fixture` prints one JSON object with float32 bitwise mismatch count,
maximum and mean absolute error, one-thread ordered CPU time, CUDA context
initialization time, and the warm CUDA call time including allocation and both
transfers. It also reports their sum as first-call time. This checks the bridge
against a CPU reference using the same supplied vectors and neighbor order;
it does not measure a FreeSurfer stage. Supply `-ccbin /path/to/g++` to `nvcc`
when its default host compiler is incompatible with the CUDA Toolkit.

This checks one mathematical kernel. Replacing it in `mris_sphere` or
`mris_register` requires rebuilding those executables with a modified FreeSurfer
`utils` library; FreeSurfer links `utils` statically, so `LD_PRELOAD` cannot
reliably replace this internal call. Stage and full recon-all numerical parity
must be checked after integration.

## Opt-in `mris_sphere` bridge

`apply_mris_sphere_cuda.py` applies `mris_sphere_cuda.patch` to a checkout at
the exact FreeSurfer source commit above and copies
`fs_cuda_average_gradients.cu` into `mris_sphere/`. It never edits an installed
FreeSurfer tree unless that tree is explicitly supplied as its source argument.

```sh
python3 tools/native_cuda_experiments/apply_mris_sphere_cuda.py \
  /path/to/freesurfer-source --check
python3 tools/native_cuda_experiments/apply_mris_sphere_cuda.py \
  /path/to/freesurfer-source
# Configure the patched FreeSurfer source with its normal build dependencies,
# CMake >= 3.18, CUDA Toolkit, and a CUDA-compatible host compiler:
cmake -S /path/to/freesurfer-source -B /path/to/fs-build \
  -DFS_SPHERE_CUDA_GRADIENTS=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_CUDA_HOST_COMPILER=/path/to/cuda-compatible-g++
cmake --build /path/to/fs-build --target mris_sphere -j4
```

The CMake option builds the CUDA object into `mris_sphere` only; the other
FreeSurfer executables keep their CPU implementations. At runtime, the rebuilt
`mris_sphere` uses CUDA only with `FS_SPHERE_CUDA_GRADIENTS=1`. With that variable
unset, or after a CUDA call fails, it uses the original CPU loop. The CUDA
runtime is linked statically so missing `libcudart.so` does not prevent CPU
fallback. A successful CUDA call writes `MRISaverageGradients: CUDA active` to
the process log. This has not yet been integrated into the recon-all bundle or
validated through a complete recon-all run. On one left-hemisphere
`inflated.nofix` to `qsphere.nofix` stage, both the clean rebuilt and
CUDA-enabled binaries produced the official ordered coordinates and faces
exactly. Their measured stage times were about 76.0 and 76.7 seconds,
respectively, so that run did not show a stage-level gain.

The patch also fixes the bounding-box `abs` calls to retain the official 8.2
binary's integer truncation when compiling with C++17 and ITK 5.3. Without
that compatibility fix, the same `lh.inflated.nofix` scaled by 0.347 instead
of 0.349 and its sphere differed by up to 3.055 mm. On CentOS 7 targets,
compile `utils/chklc.cpp` against the target's `crypt.h`: linking an object
compiled against newer `libxcrypt` headers caused a crash in `crypt_r` during
license checking. Rebuild and relink both clean and CUDA binaries on the target
before comparing outputs.

The bridge consumes FreeSurfer's in-process active-vertex order, neighbor
order, and actual float32 gradients, including its `num_avgs > 150` storage
reordering. It copies those arrays to GPU on each call; topology caching is
deliberately deferred until in-process parity is verified. Compile with the
same OpenMP setting as the reference: without OpenMP, the original code enters
`MRISaverageGradientsFast` before this bridge. Compare an unmodified rebuilt
binary to the official binary first, then compare CUDA to that rebuilt CPU
binary for every `mris_sphere` output and downstream per-vertex and ROI metrics.

For that stage comparison, freeze one subject's `surf/lh.inflated` and run
`benchmark_mris_sphere_stage.py` with the official, clean rebuilt, and patched
binaries, the FreeSurfer home/license, and the target GPU UUID. The script
runs the patched binary once with CUDA disabled and once enabled. It saves each
log, wall time, output sphere, and ordered vertex/face comparison in a fresh
`report.json`; it fails if the CUDA-active marker is absent or if the default
1e-5 mm coordinate bound is exceeded. Run the right hemisphere separately.
Pass `--existing-sphere` from the completed official subject to check that
the stage-only official command reproduces the original recon-all result.
For the earlier `inflated.nofix` to `qsphere.nofix` call, use `--mode nofix`;
the default `final` mode matches the later `inflated` to `sphere` call.
The stage output alone does not establish downstream recon-all parity.
The first full left-hemisphere final-stage replay failed the official gate:
the clean rebuilt and patched CUDA outputs matched each other exactly but
differed from the official sphere by up to 7.968 mm. The CUDA binary is not
eligible for the validated bundle. See
[`gpu_native_sphere_pilot_2026-09-24.md`](../../validation/recon_all/gpu_native_sphere_pilot_2026-09-24.md).
