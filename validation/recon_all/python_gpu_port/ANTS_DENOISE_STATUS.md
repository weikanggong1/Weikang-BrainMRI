# `AntsDenoiseImageFs`: whole-stage Python parity

The fixed FreeSurfer 8.2 `recon-all` call is
`AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz`. The pinned source
`AntsDenoiseImageFs/AntsDenoiseImageFs.cpp` converts each MGH frame to an ITK
float image, fixes ITK to one thread and applies adaptive non-local means with
Gaussian noise, patch radius 1, search radius 2, local mean/variance radius 1,
epsilon 1e-5, mean threshold 0.95, variance threshold 0.5, smoothing factor 1,
and smoothing variance 2. The reconstruction call does **not** pass `--rician`.

[`ants_denoise_python.py`](../../../src/fnit/recon_all/ants_denoise_python.py)
uses `antspyx==0.6.3` with `noise_model="Gaussian"`, `p=1`, `r=2`, no mask and
no shrink. It passes FreeSurfer voxel spacing, then uses the source-defined
`MRIsetVoxVal` conversion: clip to `[0, 255]` and round positive half values
up. It writes the new voxel bytes while preserving the input MGH header and
trailing tags. This is a **CPU Python API backed by ANTs/ITK compiled code**;
it requires `pip install antspyx==0.6.3` and no FreeSurfer installation at
runtime.

On `headcw`, the fresh native command ran on the frozen `fs_sub01/mri/brain.mgz`
in **27.58 seconds** and reproduced the saved official output at all
**16,777,216/16,777,216** voxels. The Python wrapper ran end to end in
**28.25 seconds**, and [fresh native versus Python](ants_denoise_stage_headcw.json)
also matched **16,777,216/16,777,216** uint8 voxels, with zero maximum voxel
difference, identical affine and identical first 284 MGH header bytes. The
standalone ANTsPy filter call took **25.98 seconds** in the same isolated
environment; these are single observed wall times, not repeated-run speed
claims. The Python wrapper is about 0.67 seconds slower than fresh native in
these observations.

The [rounding experiment](ants_denoise_rounding_headcw.json) explains the
remaining apparent mismatch before conversion was corrected: truncation
matched 16,135,004 voxels and NumPy's ties-to-even rounding matched
16,777,213; FreeSurfer's half-up `nint` matched all 16,777,216. The focused
half-up regression test passed **1/1** on headcw.

Compressed `.mgz` file hashes differ. Native `MRIwrite` added a 2,421-byte
trailer whereas the input and Python output have 2,420-byte trailers. The
trailer is not image data; the voxel array, geometry and fixed MGH header
matched exactly. This is whole-stage **numerical image parity** on the frozen
T1, not compressed-file byte parity. The wrapper is ready for use as the
denoising stage once the runner declares the ANTsPy dependency.
