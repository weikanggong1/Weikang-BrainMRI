# Four-model batch inference study, 24 September 2026

The tested two-image tensor batch (`B2`) did **not** beat two independent, persistent single-image Python programs (`P2`) for any of SynthStrip, WMH-SynthSeg, SynthSR or SynthMorph on this shared H100. WMH-SynthSeg and SynthSR improved over one serial `B1` program, but P2 was faster. SynthMorph's B2 versus B1 order changed between the two runs. The Python `predict_batch()` API therefore keeps each network inference at B1 and offers `workers=2` to run two processes; the tensor B2 path studied here was experimental and is not part of the release. The P2 results below came from two independent persistent programs and do not measure `predict_batch(..., workers=2)`. These findings apply to the measured B2 cohorts and machine, not every batch size or deployment.

[Machine-readable aggregate results](batch_modes_2026-09-24.json) contain all 24 event timings, output byte counts and hashes, model hashes, shape counts, and parity aggregates. They omit subject identifiers and private paths.

## Protocol and timing scope

- Hardware: `gpucw1`, one shared NVIDIA H100 PCIe (81,559 MiB). `CUDA_VISIBLE_DEVICES=1` mapped Python `cuda:0` to physical GPU 1. PyTorch 2.5.1, CUDA 11.8, cuDNN 9.1, surfa 0.6.3, nibabel 5.4.0 and pandas 2.2.3. The GPU had other allocations during this study; its sampled used-memory baseline varied between 11,837 and 15,336 MiB. Reported GPU peaks are **total sampled device use**, not this job's isolated peak.
- `B1`: one persistent Python process, one loaded model, one image per inference. `B2`: one persistent Python process, one loaded model, two shape-compatible images stacked for network inference. `P2`: two independent persistent Python child processes, each with its own loaded model and half the cases, each inferring at B1. P2 used two aggregate Torch CPU threads versus one for B1/B2; each process had `torch.set_num_threads(1)` and one OMP/MKL/OpenBLAS thread. No DataLoader workers were used. P2 is the relevant existing parallel-program baseline; it does not reload weights for every case.
- Each mode handled the **same ordered cohort**, checkpoint, default numerical settings, model configuration, preprocessing, output prefix contract and all routine output writes. We ran two blocks in opposite order: `B1 → B2 → P2`, then `P2 → B2 → B1`. Each event started a fresh process, loaded the model, processed one **cold** complete cohort and then one **warm** complete cohort using the same model. Each cold and warm number uses synchronized CUDA wall time around input reads, preprocessing, inference, postprocessing and all requested file writes. The cold number excludes process startup and model load. The **external** number is the entire Python command for both cohorts, including startup, imports, model load, writes and report generation. The cold cohort also warmed all realized B1 or B2 network shapes before the measured warm cohort; SynthStrip's cuDNN autotuning is therefore included in cold. Warm outputs were removed after measurement; cold outputs were retained for parity checks.
- All 24 formal events exited successfully, with the reported B2 batch size actually reaching the network for every case. P2 assigned 6/6 cases to its two worker PIDs for N=12 and 2/2 for N=4. No batch fallback or OOM occurred. The study measured one warm repetition per fresh process in each order block. Shared GPU and storage contention remain possible, especially for a small 7–8% B2/B1 difference.

## Inputs and shape compatibility

The cohort fingerprint is SHA-256 of the **ordered sequence of compressed input-file SHA-256 digests**, with no paths or subject IDs. We kept the original preprocessing rather than forcing every original volume onto an arbitrary voxel grid.

| Function | Cases | Original input shapes | Network input shapes | Actual B2 cases / two-image groups | Cohort fingerprint |
| --- | ---: | --- | --- | ---: | --- |
| SynthStrip | 12 T1w | 5 × 208×320×320; 1 × 223×320×320; 6 × 224×320×320, 0.8 mm isotropic | 12 × 192×256×256 | 12 / 6 | `11a70e0b4469e2b2de41450ebfeaa6ead9251e4f526fc821b98ccd863a356907` |
| WMH-SynthSeg | 12 FLAIR | 12 × 256×256×42, 0.8438×0.8438×3 mm | 12 × 224×224×128 | 12 / 6 | `202918287017eedd921a938b0d23796845bea7090206dd82a3109365a8ab2942` |
| SynthSR | the same 12 FLAIR | 12 × 256×256×42 | 12 × 224×224×128 | 12 / 6 | same FLAIR fingerprint |
| SynthMorph `joint` | 4 T1w, shared 2 mm fixed template | 1 × 208×320×320; 1 × 223×320×320; 2 × 224×320×320, 0.8 mm isotropic moving images | 4 × 256×256×256 | 4 / 2 | `92817c697265949c94c62d512c17bd0e4f6d278da5db928f154ce71d4eb43e86` |

Both SynthMorph pairs deliberately mixed different original moving-image shapes, and all four images still entered true B2 network batches. SynthStrip likewise mixed original dimensions within one normalized network bucket. For WMH-SynthSeg and SynthSR, the cohort's original dimensions happened to match. If normalized network tensors have different dimensions, the experimental code used separate shape buckets; a user-requested B2 can then realize B1 for a leftover case. Resampling all originals solely to obtain a single pixel count would alter physical sampling and potentially the field of view and output geometry; it was not used as an equivalent substitute for each model's existing spatial preprocessing. With only one input image, a requested B2 cannot form a two-subject batch and gives no B2 single-subject speed comparison.

## Complete cohort wall time

Seconds below include **all outputs**: SynthStrip brain/mask/signed-distance (3 files per case), WMH-SynthSeg segmentation/probability/volumes CSV (3), SynthSR restored image (1), and SynthMorph moved/fixed-moved images plus forward/inverse transforms (4). For every row the last column is one cold plus one warm cohort and also includes startup and model load; compare it only with the same column and same model. The cold/warm columns exclude startup/model load. GPU peak is total sampled used memory on physical GPU 1 in MiB. It was sampled independently for each event; several mode repeats reached the same observed peak.

| Function | Block | Mode | Cold cohort, s | Warm cohort, s | External two-cohort wall, s | Files / cold cohort | GPU peak, MiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| SynthStrip | 0 | B1 | 176.33 | 139.40 | 324.50 | 36 | 19,353 |
| SynthStrip | 0 | B2 | 155.82 | 152.37 | 319.09 | 36 | 29,810 |
| SynthStrip | 0 | P2 | 76.39 | 71.29 | 169.93 | 36 | 30,367 |
| SynthStrip | 1 | P2 | 88.95 | 76.35 | 180.95 | 36 | 30,367 |
| SynthStrip | 1 | B2 | 150.04 | 138.84 | 297.00 | 36 | 29,810 |
| SynthStrip | 1 | B1 | 148.99 | 131.35 | 290.71 | 36 | 22,852 |
| WMH-SynthSeg | 0 | B1 | 35.80 | 32.24 | 83.82 | 36 | 43,978 |
| WMH-SynthSeg | 0 | B2 | 38.17 | 29.98 | 78.68 | 36 | 64,756 |
| WMH-SynthSeg | 0 | P2 | 20.24 | 19.34 | 57.70 | 36 | 72,619 |
| WMH-SynthSeg | 1 | P2 | 19.57 | 19.07 | 56.31 | 36 | 72,619 |
| WMH-SynthSeg | 1 | B2 | 31.90 | 26.36 | 66.01 | 36 | 64,756 |
| WMH-SynthSeg | 1 | B1 | 36.51 | 36.50 | 82.13 | 36 | 43,978 |
| SynthSR | 0 | B1 | 51.75 | 56.48 | 115.23 | 12 | 22,424 |
| SynthSR | 0 | B2 | 54.38 | 52.03 | 116.26 | 12 | 28,916 |
| SynthSR | 0 | P2 | 32.47 | 30.50 | 79.79 | 12 | 29,511 |
| SynthSR | 1 | P2 | 27.57 | 31.09 | 73.56 | 12 | 29,511 |
| SynthSR | 1 | B2 | 53.97 | 55.41 | 117.87 | 12 | 28,916 |
| SynthSR | 1 | B1 | 57.19 | 59.42 | 124.70 | 12 | 22,424 |
| SynthMorph | 0 | B1 | 257.93 | 293.89 | 579.96 | 16 | 28,672 |
| SynthMorph | 0 | B2 | 254.78 | 261.63 | 532.70 | 16 | 39,106 |
| SynthMorph | 0 | P2 | 129.00 | 128.13 | 278.71 | 16 | 42,013 |
| SynthMorph | 1 | P2 | 129.10 | 126.18 | 282.06 | 16 | 42,013 |
| SynthMorph | 1 | B2 | 248.82 | 239.26 | 501.11 | 16 | 39,106 |
| SynthMorph | 1 | B1 | 240.94 | 217.96 | 475.02 | 16 | 28,672 |

Median warm complete-cohort time across the two blocks was SynthStrip **135.38 / 145.61 / 73.82 s**, WMH-SynthSeg **34.37 / 28.17 / 19.20 s**, SynthSR **57.95 / 53.72 / 30.79 s**, and SynthMorph **255.93 / 250.45 / 127.16 s** for B1 / B2 / P2 respectively. Thus B2 versus serial B1 was 0.93×, 1.22×, 1.08× and 1.02× respectively, while P2 was faster than B2 for all four. SynthMorph B2 beat B1 in block 0 but lost in block 1, so its 1.02× median is not evidence of a stable advantage. WMH P2 sampled 72,619 MiB = 70.92 GiB of the 81,559 MiB device; this is a substantial shared-device memory cost. P2's aggregate host threads also exceed either single-process arm.

## Full-cohort output comparison

All shapes and spatial affines matched B1 for every case and saved image in the B2 and P2 cold cohorts. Checks compared decompressed voxel data and, for SynthMorph, forward/inverse transforms in RAS millimeters. CSV comparisons used numeric columns. Exact cohort output file counts, byte counts by output type, and ordered file-content SHA-256 hashes for all three modes are in the [JSON report](batch_modes_2026-09-24.json). The per-cohort bytes were 789,562,730 for SynthStrip, about 275 million for WMH-SynthSeg, 37,522,398 for SynthSR and about 1.029 billion for SynthMorph. WMH CSVs embed mode-dependent output paths, so whole-file hashes/bytes differ even when P2 numeric columns equal B1. SynthMorph P2 MGZ bytes/hashes differed while decoded images and transform arrays were numerically exact.

| Function | P2 versus B1 | B2 versus B1 |
| --- | --- | --- |
| SynthStrip | All 12 brain images, masks and signed-distance maps exactly equal at every voxel; image files have identical content hashes. | Same exact result. |
| SynthSR | All 12 uint8 images exactly equal, with identical file-content hashes. | Same exact result. |
| WMH-SynthSeg | All 12 segmentations and lesion-probability arrays exactly equal; all numeric CSV columns equal. | 783 changed segmentation voxels out of 70,543,872; 20 changed label-77 voxels, worst case label-77 Dice 0.9997600. Lesion-probability maximum absolute difference 0.0020893; maximum per-case mean absolute difference 4.21×10⁻⁷. Maximum per-case relative difference in label-77 probability sum was 0.01223%; the CSV label-77 volume maximum was 0.01224%. |
| SynthMorph `joint` | All four outputs and forward/inverse transform arrays exactly equal at every voxel/element. | Moved-image maximum absolute difference 0.03601, maximum per-case mean absolute difference 0.000634; forward/inverse displacement-field maximum RAS errors 0.000413/0.000474 mm. The `fixed_moved` image had 13 voxels with absolute intensity difference >10 among 90,009,600 voxels, maximum 7,332.92. These 13 voxels were **inside** the images, not within two voxels of an edge. Maximum per-case range-normalized RMSE was 0.000174. |

WMH-SynthSeg's formal runs used the environment default `torch.backends.cudnn.allow_tf32=True` and `torch.backends.cuda.matmul.allow_tf32=False`. In a separate two-case numerical diagnostic, turning cuDNN TF32 off for **both** B1 and B2 made their segmentation and probability arrays exactly equal; it also changes the B1 numerical baseline and was not used for the formal timing comparison. That small diagnostic does not establish TF32-off performance on the full cohort.

SynthMorph's sparse `fixed_moved` intensity outliers remain a real numerical difference despite sub-0.001 mm transform errors. They were 0, 1, 8 and 4 voxels above 10 for the four cases, with no moved-image voxel above 1. Their precise cause was not established. We therefore do not claim full B2 output equivalence. In a separate two-case preflight, the unreleased prototype accepted a row-aligned list of different fixed images and saved readable joint/affine outputs; the earlier B1-only candidate also passed a two-case fixed-list smoke test.

## Checkpoints and release validation

| Checkpoint | SHA-256 |
| --- | --- |
| SynthStrip `synthstrip.1.pt` | `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33` |
| WMH-SynthSeg `WMH-SynthSeg_v10_231110.pth` | `0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988` |
| SynthMorph affine `synthmorph.affine.2.h5` | `1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6` |
| SynthMorph deform `synthmorph.deform.3.h5` | `95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536` |
| SynthSR `synthsr_v20_230130.h5` | `a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b` |

The exact experimental B2 source archive has SHA-256 `d10420ed4eabb3f209a78eb102ce0e7527296ca8e838c5954f68ec03831064e5`; the executed all-output benchmark driver has SHA-256 `0ef205c216c0387c441f3aca2aca7c4d17b5d1fb9ea8eb95e8046cd52ccd1721`. These identify the archived, **unpublished** prototype; the earlier merged B1 interface could not run its B2 arm, nor can the current two-process API. The prototype's remote suite passed 64 tests with 3 skips before formal benchmarking. A pre-merge, four-model B1 candidate archive (`dca897b2c92b8d3cc19338f3af07104f5930c7bb404e14fd9ddb774b3eacb6d5`) separately passed 60 tests with 3 skips and four two-case output smokes. That archive is not the final merged source.

The validated merged code snapshot was commit `06a7c3ba5c25652fdd32076fed49e27b59a3fa14` (Git tree `12696c9e9cbc01aca2c21da1723d52c3f4c73b4f`); its exact archive SHA-256 is `380581b8057c971ce8a860cd600b1a57035a5c2e5e738756a411c153de858d98`. In a fresh isolated gpucw1 directory, this tree passed the complete suite (`83 passed, 3 skipped`, exit 0), a focused FAST/BatchRunner/VBM suite (`33 passed`, exit 0), and real-checkpoint two-case full-output smokes for each of the four models (6 SynthStrip files, 6 WMH-SynthSeg including CSV, 2 SynthSR, 8 SynthMorph with a row-aligned fixed-image list; exit 0). An offline `pip wheel --no-deps` build passed; the wheel SHA-256 is `f14e5b6ba7d551c8fa610325952e07f4b2e562ffb090c0101d7a35b5a1bbeacc`. Its contents include `fast/README.md` and all eight `fast/upstream_fast4/*` files. Installed into an isolated target, its `fs-torch --help` exposed `fast`, the four single-image model commands, and `apply`, with no `batch` command. The subsequent report edit changed no library code. These are release validation checks; the performance table above remains the measurement of the pre-merge B2 prototype with the stated full-output contract.

The decision not to release tensor B2 is based on the stronger two-program comparator and the measured output differences. The experiment did not establish a B2 single-subject latency benefit, a B4 result, a final-contract threaded baseline, or a DataLoader worker benefit.

## Python table API with two processes

The released four-model `predict_batch(table, workers=2)` keeps B=1 network inference in each process. The caller's model handles alternating rows while one spawned Python process handles the other rows on the same GPU. This differs from the independent, persistent P2 programs above: the API starts and closes its child on **every call**, so the following call times include child startup and model loading. The parent's one-time model construction is outside the call times. `workers=1` and `workers=2` both saved every routine output for the same ordered cohort.

We tested the exact source tree `34373a7c7347a98ebb9c523d741f78b8f824049a` (archive SHA-256 `27b4a4d0692a99ef03944013585506cf277ed3f2e1bc916d0dfe0c8321ed53db`) on gpucw1. Physical H100 GPU 1 was exposed as `cuda:0` with `CUDA_VISIBLE_DEVICES=1`; each inference process used one Torch CPU thread. CUDA was synchronized around each full table call, which included reading images, preprocessing, inference, postprocessing and writing all outputs. Each run loaded a parent model, then called the API twice per mode. Strip, SR and WMH used two fresh runs in reversed order (`1→2`, then `2→1`); the more expensive Morph joint registration used one `1→2` run with two calls per mode. Times are whole-cohort seconds. The GPU was shared, and sampled device memory includes other users' allocations.

| Model | Cases, outputs per call | `workers=1` calls, s | `workers=2` calls, s | Median, 1 / 2, s | Observed ratio |
| --- | ---: | --- | --- | ---: | ---: |
| SynthStrip | 12 T1w, 36 files | 153.79, 151.05, 107.34, 105.86 | 97.40, 81.02, 71.61, 74.95 | 129.19 / 77.98 | 1.66× |
| SynthSR | 12 FLAIR, 12 files | 53.53, 53.36, 57.39, 51.36 | 34.30, 36.42, 37.26, 40.90 | 53.45 / 36.84 | 1.45× |
| SynthMorph `joint` | 4 T1w, 16 files | 238.20, 225.78 | 130.78, 141.43 | 231.99 / 136.10 | 1.70× |
| WMH-SynthSeg | 12 FLAIR, 36 files | 32.97, 28.77, 28.17, 28.25 | 28.76, 26.40, 27.67, 23.17 | 28.51 / 27.04 | 1.05× |

Every paired call for Strip, SR and Morph favored two processes. WMH's roughly 5% median difference is small relative to shared-device variation and the extra memory cost, so it does not establish a stable practical speed advantage. In the WMH runs the highest sampled **total device** use was 61,205 of 81,559 MiB, with no out-of-memory failure. The Morph total-device sample reached 50,986 MiB. These are not model-exclusive allocations. For only two cases, child startup could dominate: SynthSR took 9.15 s with one process and 18.22 s with two; the two-case Morph fixed-list smoke took 237.90 and 144.96 s because registration per case was much heavier. A single image does not create two-subject parallelism.

Across all formal repeats, Strip's 144 image-file pairs and SR's 48 pairs were byte-identical, with identical decoded voxels and affines. Morph's moved and fixed-moved NIfTI outputs were byte-identical; its forward and inverse MGZ fields had different container bytes but identical decoded displacements (maximum 0 mm) and geometry. WMH segmentation and lesion-probability NIfTIs were byte-identical. All WMH CSV headers and numeric fields matched exactly; their `Input-file` columns differ because the compared modes wrote to different absolute prefixes. A separate two-case rerun overwrote the same prefixes and confirmed the WMH CSV files were byte-identical. The two-case Morph smoke also verified a row-aligned `fixed` path list. Every reported formal call and output audit exited successfully. The GPU-visible test suite for the frozen source passed `90 passed, 3 skipped` (the skips require optional FreeSurfer checkpoints); the wheel built and exposed all four single-subject CLI commands with no multi-subject CLI. A two-case IPython call with `workers=2` saved both outputs successfully on retry. Its first attempt had a CUDA out-of-memory error while constructing the model, before entering `predict_batch`; the cause was not established. [Anonymous machine-readable results](python_table_workers_2026-09-24.json) provide individual calls, file counts, parity aggregates and sampled GPU memory conditions without subject paths.
