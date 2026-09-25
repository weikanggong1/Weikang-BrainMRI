# Connected T1-to-N4 comparison

The input was the same `sub-01_T1w.nii.gz` used for the preserved, unmodified
FreeSurfer 8.2 reconstruction. The candidate `orig.mgz` and
`talairach.xfm` came from the Python `run_input_talairach_chain` run, which
started with an empty output folder. The candidate then ran
`fnit.recon_all.n4_sitk.correct_volume` and
`fnit.recon_all.n4_wrapper.make_nu` on headcw CPU. No FreeSurfer executable
was called in the candidate chain. SimpleITK was version 2.5.6.

| `nu.mgz` reference | Differing voxels / 16,777,216 | Maximum intensity difference | MGH header and voxel payload |
| --- | ---: | ---: | --- |
| Fresh official N4 on headcw, same source T1 | 0 | 0 | identical |
| Archived unmodified official full reconstruction | 34 | 2 | different |
| Archived hybrid full reconstruction | 34 | 2 | different |

The archived differences have the same positions and values relative to the
candidate and the fresh official run: 25 voxels are lower by 2 and 9 are
lower by 1. All three comparisons have identical geometry and 284-byte MGH
headers. The fresh official and candidate files differ only in trailing
metadata. The old archive removed `nu0.mgz`, so its N4 source of difference
cannot be resolved from those files; the current Python output is **not**
voxel-identical to the archived official full reconstruction. This is a
measured boundary for strict end-to-end acceptance, not an allowed exception.

The candidate N4 correction took 106.20 s and the Python wrapper 1.27 s in
this single headcw CPU run. These times are unpaired and do not establish a
speed ratio. The exact input hashes, output hashes, and comparisons are in
[the report](connected_n4_cpu_20260926.json); the
[replay script](experimental/run_connected_n4_headcw.py) documents the
file-oriented procedure. The earlier
[paired N4 investigation](../../../docs/recon_all/N4_WRAPPER_VALIDATION.md)
contains the fresh native run and isolated postprocessing comparisons.

## Single-call Python API replay

[`run_input_n4_chain`](../../../src/fnit/recon_all/input_n4_chain.py)
was also run from the original T1 in a new empty subject directory on headcw.
It calls the validated input, PyTorch SynthStrip, PyTorch SynthMorph affine,
SimpleITK N4, and Python uchar wrapper in sequence. The four MRI files before
N4 again had zero voxel mismatches and identical MGH headers/payloads against
the archived official run; the Talairach corner displacement was
0.00015691 mm ([input report](connected_input_n4_api_comparison_20260926.json)).
Its `nu.mgz` matched the previous Python result and fresh headcw official N4
at every voxel, while differing from the archived official output at the same
34 voxels ([N4 report](connected_input_n4_api_nu_comparison_20260926.json)).

This one-call CPU run spent 0.33 s importing the T1, 0.013 s copying rawavg,
2.05 s conforming/tagging, 5.18 s on SynthStrip, 9.14 s on Talairach affine,
105.49 s on N4, and 0.93 s on the N4 wrapper. These are one-run stage times,
without a paired end-to-end native timing. The call stops at `nu.mgz`; it
has not generated aseg, surfaces, annotations, or cortical metrics.
