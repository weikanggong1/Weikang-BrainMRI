# Third-party notices

All or portions of this licensed product (such portions are the "Software") have been obtained under license from The General Hospital Corporation "MGH" and are subject to the following terms and conditions:

The complete FreeSurfer Software License terms appear in [`licenses/FreeSurfer.txt`](licenses/FreeSurfer.txt).

This package is an independent adaptation; it is not an official FreeSurfer release.

- `synthstrip/` adapts FreeSurfer's `mri_synthstrip`, by Andrew Hoopes, Jocelyn S. Mora, Adrian V. Dalca, Bruce Fischl, Malte Hoffmann and collaborators. The original implementation already uses PyTorch. Preserve the accompanying FreeSurfer license (`licenses/FreeSurfer.txt`).
- `synthmorph/pipeline.py` and the image-space workflow adapt FreeSurfer's SynthMorph registration code by Malte Hoffmann and collaborators. Preserve the accompanying FreeSurfer license.
- `synthmorph/models.py` and `synthmorph/spatial.py` implement VoxelMorph/Neurite algorithms, originally distributed under Apache License 2.0 (`licenses/Apache-2.0.txt`). Modified for PyTorch, channels-first tensors, direct HDF5 loading, and reusable inference.
- `wmh_synthseg/` adapts the inference workflow and 3D U-Net of FreeSurfer's `mri_WMHsynthseg` (FreeSurfer 8.2.0-1) under the FreeSurfer Software License (`licenses/FreeSurfer.txt`).
- The FreeSurfer WMH-SynthSeg `unet3d` model/building-block implementation derives from Adrian Wolny's [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), distributed under the MIT License (Copyright 2018 Adrian Wolny). The corresponding model architecture in this package retains that attribution and license; the complete license is in [`licenses/pytorch-3dunet-MIT.txt`](licenses/pytorch-3dunet-MIT.txt), copied from the [upstream license](https://github.com/wolny/pytorch-3dunet/blob/master/LICENSE).
- Surfa is installed as a dependency; its own license and notices apply.
- Pretrained weights are separate data files and are not included in the wheel. Their upstream terms remain applicable. See https://synthstrip.io, https://synthmorph.io, and https://surfer.nmr.mgh.harvard.edu/fswiki/WMH-SynthSeg for models, licenses, and scientific citations.

SynthStrip and SynthMorph reference FreeSurfer 8.2.0, build `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`; WMH-SynthSeg references the separately audited FreeSurfer 8.2.0-1 installation. Exact available source and weight provenance is recorded in `docs/provenance.json` and the feature documentation.
