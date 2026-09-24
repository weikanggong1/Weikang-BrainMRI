# Serial A-C recon-all comparison on gpucw1

On 2026-09-24, FreeSurfer 8.2.0-1 (`A_official`) and the frozen standalone
GPU/native candidate (`C_candidate`) completed sequentially on `gpucw1` using
the same public T1, `examples/data/sub-01_T1w.nii.gz` (SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`).
Both exited 0 with no missing required outputs. They used four OpenMP threads;
the candidate selected `cuda:1`. The candidate profile and bundle are described
in [the standalone validation](gpucw1_sub01_2026-09-24.md).

| Run | Elapsed wall time |
|---|---:|
| Official FreeSurfer | 6,805.180 s |
| Standalone candidate | 5,827.217 s |

The candidate's observed wall time was 977.963 s (14.37%) lower. This is a
single A-C observation, **not a controlled speedup estimate**: the shared
node's median one-minute load was 136.04 during A and 119.66 during C on 128
logical CPUs. GPU visibility also differed between the commands. The
`mris_sphere` stage used the same binary in both runs, yet the left final
sphere took 0.1729 hours during A versus 0.1320 hours during C, further
showing the effect of run conditions. A
fixed-affinity, fixed-GPU, repeated comparison is still needed for a speed
claim.

| Numerical comparison | Result |
|---|---:|
| Surface, per-vertex, atlas, volume and stats checks | 52/52 pass |
| Aggregate criteria | 19/19 pass |
| Ribbon and wmparc voxel checks | 2/2 pass |
| Aseg, aparc+aseg, ribbon and wmparc voxel mismatches | 0 each |
| White, pial, inflated and white.preaparc ordered geometry | identical |
| Per-vertex thickness, area, TH3 volume, curvature and sulc maximum error | 0 |

These checks support output agreement for this T1 and profile. They do not
establish agreement for other subjects. A separate held-out SynthSeg check
still fails the frozen CSF soft-volume threshold, as recorded in the
standalone validation.

Raw records are retained on shared storage under
`work/reconall_benchmark_pair_ac_20260924/`. The comparison JSON SHA-256
digests are `3b18a4e20231d4b33dc7bbc18d868e5e0afb61b960be155da01f86a31f7d0383`
(52 checks), `9a272a71ac4a1880ca61fd62b6297f028dd12454545af87c09911285f70a28fd`
(19 checks), and `29472515f746c30993757ab90be67f0c119c73392ad6329b68acfb7fc7ff0a9a`
(2 checks).
