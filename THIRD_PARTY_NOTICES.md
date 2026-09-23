# Third-party notices

This package is an independent adaptation; it is not an official FreeSurfer release.

- `synthstrip/` adapts FreeSurfer's `mri_synthstrip`, by Andrew Hoopes, Jocelyn S. Mora, Adrian V. Dalca, Bruce Fischl, Malte Hoffmann and collaborators. The original implementation already uses PyTorch. Preserve the accompanying FreeSurfer license (`licenses/FreeSurfer.txt`).
- `synthmorph/pipeline.py` and the image-space workflow adapt FreeSurfer's SynthMorph registration code by Malte Hoffmann and collaborators. Preserve the accompanying FreeSurfer license.
- `synthmorph/models.py` and `synthmorph/spatial.py` implement VoxelMorph/Neurite algorithms, originally distributed under Apache License 2.0 (`licenses/Apache-2.0.txt`). Modified for PyTorch, channels-first tensors, direct HDF5 loading, and reusable inference.
- Surfa is installed as a dependency; its own license and notices apply.
- Pretrained weights are separate data files and are not included in the wheel. Their upstream terms remain applicable. See https://synthstrip.io and https://synthmorph.io for models, licenses, and scientific citations.

Reference installation: FreeSurfer 8.2.0, build `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`. Exact source and weight SHA-256 values are recorded in `docs/provenance.json`.
