# Continuous T1 import through Talairach affine

`run_input_talairach_chain` connects the package's NIfTI import, single-run
copy, PyTorch conform, PyTorch SynthStrip, and PyTorch SynthMorph affine plus
Python LTA-to-MNI-XFM conversion in one process. It invokes no FreeSurfer or
FSL executable. The template and the two model weights are external files.
This is an early-stage chain, not complete recon-all; 33-class SynthSeg, N4,
surfaces and statistics are outside this call.

```python
from fnit.recon_all.input_talairach_chain import run_input_talairach_chain

report = run_input_talairach_chain(
    "sub-01_T1w.nii.gz", "/empty/sub01",
    "/external/weights", "/external/recon_all_assets",
    device="cpu", threads=4,
)
```

The same function has a module CLI (`python -m
fnit.recon_all.input_talairach_chain T1 SUBJECT_DIR --weights-dir WEIGHTS
--assets-dir ASSETS`). It requires an empty subject directory. Its source
SHA-256 is `8043810b543078220a876c1490b7b00d6b43560c02e248dca4e4af4222bd2284`.

A headcw CPU run used the original T1 SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`,
SynthStrip weight `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33`,
SynthMorph affine weight `1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6`,
and MNI305 stripped template
`fff93f13255a8d393c0e787fbfcfaf5eb379e88e955bda7e04e31552568878a4`.
The latter was copied to an external asset directory and checked against the
[verified asset manifest](ASSETS_VALIDATION.md). The asset downloader's
separate network attempt failed in this headcw session; the local checked
copy provided the exact required bytes.

| Output | Archived unmodified official | Archived hybrid |
| --- | ---: | ---: |
| `orig/001.mgz`, `rawavg.mgz`, `orig.mgz`, `synthstrip.mgz` | 0 voxel differences each; 284-byte MGH header, payload, dtype and affine exact | Same |
| `talairach.xfm` maximum 4×4 element difference | 1.526×10⁻⁵ | 3.052×10⁻⁵ |
| XFM displacement at eight input-grid corners, maximum | 0.0001569 mm | 0.0001355 mm |

The [comparison report](connected_input_talairach_comparison_20260926.json)
contains the four image hashes and all per-file gates. The
[run report](connected_input_talairach_cpu_20260926.json) records 0.499 s
import, 0.035 s copy, 2.117 s conform/tag, 6.039 s SynthStrip, and 16.051 s
Talairach. These are one CPU run's stage timings, not paired native timing or
an end-to-end speed ratio. The new transform has not yet been propagated
through N4 or surface generation. SynthMorph's shared model currently
disables TF32 for its checkpoint parity; this CPU run cannot verify its
GPU/TF32 behavior.
