# External-model recon-all candidate and independent SynthSeg

On 2026-09-25, a **candidate** runtime was derived from a copy of the
previously certified FreeSurfer 8.2 bundle. Its 13 model and lookup files were
moved to the repository's shared weight directory. The original certified
bundle and its manifest were left unchanged. The candidate still requires its
native programs and a user-provided license. It is designed to run without a
system FreeSurfer or FSL installation; complete runtime closure was not checked.
The Python package source used in the checks has
tree SHA-256 `17858230ba8e10302f4052be2299def0abaa0f84597fb2c4f0e5a607f03353e4`.

| Check on the candidate | Result |
|---|---|
| External model inventory | 13 fixed files, 3,653,914,059 bytes; each size and SHA-256 matched the certified manifest |
| Static runtime preflight | 295 bundled files and 13 external files passed hash, ELF linkage and script resource checks; no errors |
| Bundle directory size | 697,670,460 bytes versus 4,351,587,774 bytes for the original copy |
| Independent `fnit synthseg` on the existing sub-01 `orig.mgz` | Hard-label voxels and soft-volume CSV identical to the v0.6 recon-all SynthSeg output; 0 changed voxels |
| Recon-all `mri_synthseg` with external weights on the same `orig.mgz` | Hard-label voxels and soft-volume CSV identical to that output; 0 changed voxels |
| Automated Python tests on headcw | 258 passed, 8 skipped |

The two SynthSeg checks used `cuda:1` on `gpucw1`. They compare decompressed
labels and CSV content; the MGZ files have different binary hashes because of
metadata. The independently callable 33-class SynthSeg uses the same inference
core as the recon-all neural launcher and needs four files totalling 53,087,016
bytes. It does not use the separate WMH-SynthSeg model.

The complete sub-01 candidate reconstruction was started with a clean
`env -i` launch and file/exec tracing. At the user's request it was stopped
during intensity correction, before surface reconstruction and the numerical
comparison. There is **no completed 52+2+19 comparison or trace certification
for this external-model candidate**. Its manifest remains
`standalone_verified=false`; the default `run_recon_all` and
`fnit-recon-all` entry rejects it. An explicit development invocation can
use this candidate, but its complete outputs and run time are unverified.
Previous [v0.6 numerical results](gpucw1_batch_integration_2026-09-24.md)
and [v0.7 source-equivalence attestation](v06_to_v07_runtime_equivalence.json)
apply to the original bundled-model layout, not to this candidate.

Raw evidence is retained on shared project storage under
`work/reconall_external_models_20260925/evidence/`. The principal file SHA-256
values are:

| Evidence | SHA-256 |
|---|---|
| Static preflight JSON | `737dbd24fae45d2037907cdd40831ee93456f0279cc315ae3cf2bc1f22b8abb4` |
| SynthSeg smoke log | `fccd51037906d56c7e7ecbec95cc075a80853bf4382a4dcc7f40293e924b4afa` |
| Frozen Python code manifest | `ada91a535524f2968c9091c79508f86bb39fd292f0e584b95923fd1959d290d2` |
| Unverified candidate bundle manifest | `f1201528e27e76973b8acdd2594562454fa33a8a7dcf159b1888bd1a31d7f098` |
