# Archived official FreeSurfer versus published hybrid output

The existing full FreeSurfer 8.2 subject
`reconall_benchmark_pair_ac_20260924/official_subjects/a_official` and the
published hybrid subject `reconall_main_20260924/single_subjects/fs_sub01`
used the same `examples/data/sub-01_T1w.nii.gz`. The official log contains no
`FS_TORCH_NEURAL_TOOL` call; the hybrid log records its PyTorch neural calls.
This is a comparison of **two existing complete subjects**, not a new
native-free reconstruction or a new official run. It prevents the hybrid
archive from being mistaken for an unmodified FreeSurfer numerical reference.

The [138-file comparison](hybrid_vs_true_official_138_geometry_normalized_20260926.json)
passed 110 files. The checker compares all 39 archived MRI images, two surface
MGH maps, 18 meshes, 44 morph maps, 12 annotations, and 23 statistics files.
It ignores only the absolute `filename` string in surface volume geometry;
all numeric geometry fields, ordered coordinates, faces and annotations are
still checked. The metadata diagnosis is [here](hybrid_vs_official_metadata_diffs_20260926.json).

| Output group | Passed / checked | Remaining difference |
| --- | ---: | --- |
| MRI images | 29 / 39 | Seven label volumes have identical voxel values and affines but official float32 versus hybrid int32 storage; three SynthMorph outputs differ numerically. |
| Surface MGH maps and morph maps | 46 / 46 | Every scalar value is bitwise equal after loading, including thickness, area, volume and curvature. |
| Ordered meshes | 18 / 18 | Every coordinate and face is exact; their absolute source-filename metadata points to different subject directories. |
| Annotations | 12 / 12 | None. |
| Statistics | 5 / 23 | eTIV and SynthSeg soft volumes differ; downstream rows inherit those differences. |

The hard 33-class SynthSeg labels matched in all 16,777,216 voxels, but the
hybrid stored them as int32 while the official file used float32. A storage-only
conversion of the archived hybrid result to float32 reproduced the complete
decompressed official `synthseg.rca.mgz`, including header, payload and
trailer ([probe](synthseg_float32_storage_probe_20260926.json)). The independent
PyTorch SynthSeg now saves float32 labels; their integer-valued IDs are unchanged.
The connected GPU inference from the Python-generated `orig.mgz` has not yet
been rerun after this storage change.

Despite exact hard labels, the soft-volume CSVs differ across individual
regions ([all 33 columns](official_vs_hybrid_synthseg_soft_volumes_20260926.json)).
The hybrid total intracranial estimate is 464.8 mm³ higher than official
(0.0363%); the largest per-region relative difference is 0.1663%. The eTIV
estimate differs by 0.315165 mm³. All 18 statistics failures remain open;
matching hard labels or printed cortical thickness does not resolve them.
The SynthMorph forward warp has 8,970,023 values above the current 1e-6
absolute voxel threshold (maximum 0.0001907 mm in its stored coordinates),
and its inverse warp also differs; these transforms require their own paired
validation.

The [full acceptance gates](RELEASE_GATES.md) still require a connected
native-free run on the original T1, comparison to the unmodified official
subject, and same-host stage timing after numerical acceptance.
