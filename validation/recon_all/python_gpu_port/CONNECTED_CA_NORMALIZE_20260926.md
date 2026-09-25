# Connected Python GCA registration and normalization

The Python-generated `nu.mgz` and `brainmask.mgz` from the
[one-call T1 input chain](CONNECTED_BRAINMASK_20260926.md) were passed to
[`register_t1`](../../../src/fnit/recon_all/mri_em_register_python.py)
and then [`run_ca_normalize`](../../../src/fnit/recon_all/ca_normalize_python.py)
on headcw CPU. These two calls use no FreeSurfer executable. The GCA atlas
was copied to the external assets directory and checked against its official
SHA-256 `2fcd276a39800f01f93a4c8828ae6d0a8cea3d8b8b9fe1599d4ee54e806be93e`.
The source data asset is therefore external to the runtime package.

The generated `talairach.lta` differs from the archived unmodified official
matrix by at most `1.49e-8`; none of its 16 elements exceeds the preset
`2e-5` tolerance. All 315,638 atlas samples map to identical source voxels.
The 23 nonmatrix metadata lines agree after comparing atlas filename
basenames: the single literal difference is the external GCA path. This is
an archived-official comparison, not a fresh native run on the connected
Python input. The stage previously passed a separate native same-input test
on the frozen subject. See the [structured LTA report](connected_em_registration_comparison_20260926.json)
and [comparison script](experimental/compare_connected_em_registration.py).

| Downstream volume | Candidate vs archived official | Affine and MGH header |
| --- | ---: | --- |
| `norm.mgz` | 25 / 16,777,216 uint8 voxels differ; maximum 2 | exact |
| `ctrl_pts.mgz` | 0 / 100,663,296 float32 values differ across six frames | exact |

The six control-point frames are unchanged despite the earlier 34-voxel
historical N4 difference and 49-voxel brainmask difference. All 25
`norm.mgz` differing positions are among the 34 `nu.mgz` differing positions;
none occurs elsewhere. These connected outputs have not been compared to a
fresh native run with the same Python-generated inputs. [Volume report](connected_ca_normalize_comparison_20260926.json)
and [comparison script](experimental/compare_connected_ca_normalize.py)
retain the framewise checks. The isolated GCA normalization stage previously
matched a fresh official same-input run exactly on the frozen inputs.

One Python registration call took 246.77 s; the following normalization
call took 18.70 s on headcw CPU. These are sequential observations, with no
paired native timing on the connected inputs. The
[`run_input_ca_normalize_chain`](../../../src/fnit/recon_all/input_ca_normalize_chain.py)
API joins these steps with the T1-to-brainmask chain. It requires the
external MNI305 template and GCA atlas as well as external model weights.
A separate **single Python call from the original T1** in a new empty
subject directory completed in 433.99 s on headcw CPU. Its early `nu.mgz`,
`T1.mgz`, and `brainmask.mgz` matched the segmented Python replay in all
voxels and MGH header/payload bytes. Its final `talairach.lta`, `norm.mgz`,
and `ctrl_pts.mgz` have the same complete file SHA-256 values as the segmented
Python replay, so the archived-official 1.49e-8 matrix, 0/315,638 atlas
mapping, 25/16,777,216 `norm` and 0/100,663,296 control-point comparisons
apply to the single-call output as well. The [single-call report](connected_input_ca_chain_api_cpu_20260926.json),
[early comparison](connected_input_ca_chain_early_comparison_20260926.json),
[LTA comparison](connected_input_ca_chain_lta_vs_segmented_20260926.json),
and [volume comparison](connected_input_ca_chain_vs_segmented_20260926.json)
record these checks. Its principal CPU stage times were N4 100.20 s, first
normalization 62.28 s, EM registration 238.43 s, and GCA normalization
17.24 s; these numbers are one unpaired run.

This connected volume chain stops at `norm.mgz` and `ctrl_pts.mgz`; downstream
aseg, topology repair, pial and spherical registration are still outside it.
