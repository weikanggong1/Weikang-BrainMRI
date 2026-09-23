The benchmark includes 12 T1w cases and 96 requested fresh-process runs: 96 completed, 0 pending, and 0 failed.

Wall-clock time includes process/framework startup, model and input loading, inference, and output writing. Filesystem caches were not reset and other queues may share host resources. Paired elapsed-time ratios are descriptive; they do not establish performance with exclusive hardware access.

SynthStrip's **Official source CUDA** arm runs the unchanged FreeSurfer script in the project's CUDA-enabled Python environment. It is distinct from the bundled FreeSurfer command runtime.

All brackets below contain the 25th and 75th percentiles. Missing and failed observations remain listed in the CSV files.

| Function | Arm | Completed / requested | Pending | Failed | Seconds, median [IQR] |
|---|---|---:|---:|---:|---:|
| SynthStrip | FreeSurfer CPU | 12 / 12 | 0 | 0 | 16.92 [16.56, 17.27] |
| SynthStrip | Official source CUDA | 12 / 12 | 0 | 0 | 16.92 [16.74, 17.29] |
| SynthStrip | PyTorch CPU | 12 / 12 | 0 | 0 | 16.96 [16.62, 17.34] |
| SynthStrip | PyTorch GPU | 12 / 12 | 0 | 0 | 18.27 [17.29, 19.48] |
| SynthMorph | FreeSurfer CPU | 12 / 12 | 0 | 0 | 164.5 [125.4, 207.1] |
| SynthMorph | FreeSurfer GPU | 12 / 12 | 0 | 0 | 116.6 [114.6, 231.1] |
| SynthMorph | PyTorch CPU | 12 / 12 | 0 | 0 | 122.3 [120.6, 124.3] |
| SynthMorph | PyTorch GPU | 12 / 12 | 0 | 0 | 17.62 [17.3, 18.6] |

| Function | Paired elapsed-time ratio (first / second) | Available / requested | Pending | Failed | Median [IQR] |
|---|---|---:|---:|---:|---:|
| SynthStrip | FreeSurfer CPU / PyTorch CPU | 12 / 12 | 0 | 0 | 1.003 [0.9827, 1.015] |
| SynthStrip | Official source CUDA / PyTorch GPU | 12 / 12 | 0 | 0 | 0.9269 [0.8631, 0.9871] |
| SynthStrip | FreeSurfer CPU / Official source CUDA | 12 / 12 | 0 | 0 | 0.9928 [0.9851, 1.014] |
| SynthStrip | PyTorch CPU / PyTorch GPU | 12 / 12 | 0 | 0 | 0.9401 [0.8515, 0.977] |
| SynthMorph | FreeSurfer CPU / PyTorch CPU | 12 / 12 | 0 | 0 | 1.377 [1.005, 1.686] |
| SynthMorph | FreeSurfer GPU / PyTorch GPU | 12 / 12 | 0 | 0 | 6.786 [6.553, 11.94] |
| SynthMorph | FreeSurfer CPU / FreeSurfer GPU | 12 / 12 | 0 | 0 | 1.068 [0.9901, 1.268] |
| SynthMorph | PyTorch CPU / PyTorch GPU | 12 / 12 | 0 | 0 | 6.974 [6.634, 7.139] |

NRMSE is the image RMSE divided by the reference image RMS. Warp errors use physical RAS displacement in millimetres. Jacobian fractions cover the entire forward-warp target grid.

The detailed accuracy table contains 1008 available, 0 pending, and 0 failed metric observations.

| Function | Comparison | Metric | Available / requested | Pending | Failed | Median [IQR] | Min–max |
|---|---|---|---:|---:|---:|---:|---:|
| SynthStrip | FreeSurfer CPU vs PyTorch CPU | mask_dice | 12 / 12 | 0 | 0 | 1 [1, 1] | 1–1 |
| SynthStrip | FreeSurfer CPU vs PyTorch CPU | mask_disagreeing_voxels | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthStrip | FreeSurfer CPU vs PyTorch CPU | sdt_max_error_mm | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthStrip | FreeSurfer CPU vs PyTorch CPU | sdt_mae_mm | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthStrip | FreeSurfer CPU vs PyTorch CPU | stripped_image_nrmse | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthStrip | Official source CUDA vs PyTorch GPU | mask_dice | 12 / 12 | 0 | 0 | 1 [1, 1] | 1–1 |
| SynthStrip | Official source CUDA vs PyTorch GPU | mask_disagreeing_voxels | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthStrip | Official source CUDA vs PyTorch GPU | sdt_max_error_mm | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthStrip | Official source CUDA vs PyTorch GPU | sdt_mae_mm | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthStrip | Official source CUDA vs PyTorch GPU | stripped_image_nrmse | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthStrip | FreeSurfer CPU vs Official source CUDA | mask_dice | 12 / 12 | 0 | 0 | 1 [1, 1] | 0.9999998586–1 |
| SynthStrip | FreeSurfer CPU vs Official source CUDA | mask_disagreeing_voxels | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–1 |
| SynthStrip | FreeSurfer CPU vs Official source CUDA | sdt_max_error_mm | 12 / 12 | 0 | 0 | 2.319e-05 [2.009e-05, 2.48e-05] | 1.788e-05–2.718e-05 |
| SynthStrip | FreeSurfer CPU vs Official source CUDA | sdt_mae_mm | 12 / 12 | 0 | 0 | 1.004e-06 [1.001e-06, 1.006e-06] | 9.945e-07–1.02e-06 |
| SynthStrip | FreeSurfer CPU vs Official source CUDA | stripped_image_nrmse | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0.0001395 |
| SynthStrip | PyTorch CPU vs PyTorch GPU | mask_dice | 12 / 12 | 0 | 0 | 1 [1, 1] | 0.9999998586–1 |
| SynthStrip | PyTorch CPU vs PyTorch GPU | mask_disagreeing_voxels | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–1 |
| SynthStrip | PyTorch CPU vs PyTorch GPU | sdt_max_error_mm | 12 / 12 | 0 | 0 | 2.319e-05 [2.009e-05, 2.48e-05] | 1.788e-05–2.718e-05 |
| SynthStrip | PyTorch CPU vs PyTorch GPU | sdt_mae_mm | 12 / 12 | 0 | 0 | 1.004e-06 [1.001e-06, 1.006e-06] | 9.945e-07–1.02e-06 |
| SynthStrip | PyTorch CPU vs PyTorch GPU | stripped_image_nrmse | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0.0001395 |
| SynthMorph | FreeSurfer CPU vs PyTorch CPU | warp_vector_mean_error_mm | 12 / 12 | 0 | 0 | 7.446e-05 [5.031e-05, 7.885e-05] | 3.024e-05–9.472e-05 |
| SynthMorph | FreeSurfer CPU vs PyTorch CPU | warp_vector_max_error_mm | 12 / 12 | 0 | 0 | 0.0003145 [0.0002139, 0.0003597] | 0.0001639–0.0005015 |
| SynthMorph | FreeSurfer CPU vs PyTorch CPU | moved_image_nrmse | 12 / 12 | 0 | 0 | 1.758e-05 [1.229e-05, 2.027e-05] | 8.633e-06–2.989e-05 |
| SynthMorph | FreeSurfer GPU vs PyTorch GPU | warp_vector_mean_error_mm | 12 / 12 | 0 | 0 | 8.215e-05 [6.642e-05, 9.327e-05] | 5.043e-05–0.0001143 |
| SynthMorph | FreeSurfer GPU vs PyTorch GPU | warp_vector_max_error_mm | 12 / 12 | 0 | 0 | 0.0003969 [0.0002951, 0.0004645] | 0.0002385–0.0007896 |
| SynthMorph | FreeSurfer GPU vs PyTorch GPU | moved_image_nrmse | 12 / 12 | 0 | 0 | 2.22e-05 [1.785e-05, 2.834e-05] | 1.378e-05–4.05e-05 |
| SynthMorph | FreeSurfer CPU vs FreeSurfer GPU | warp_vector_mean_error_mm | 12 / 12 | 0 | 0 | 8.104e-05 [7.043e-05, 0.0001044] | 4.822e-05–0.0001421 |
| SynthMorph | FreeSurfer CPU vs FreeSurfer GPU | warp_vector_max_error_mm | 12 / 12 | 0 | 0 | 0.0003133 [0.0002806, 0.0003522] | 0.0001907–0.0004914 |
| SynthMorph | FreeSurfer CPU vs FreeSurfer GPU | moved_image_nrmse | 12 / 12 | 0 | 0 | 2.503e-05 [1.916e-05, 3.219e-05] | 1.043e-05–4.252e-05 |
| SynthMorph | PyTorch CPU vs PyTorch GPU | warp_vector_mean_error_mm | 12 / 12 | 0 | 0 | 3.161e-05 [2.586e-05, 4.373e-05] | 2.04e-05–0.0001165 |
| SynthMorph | PyTorch CPU vs PyTorch GPU | warp_vector_max_error_mm | 12 / 12 | 0 | 0 | 0.0002035 [0.0001284, 0.0002846] | 9.77e-05–0.0007041 |
| SynthMorph | PyTorch CPU vs PyTorch GPU | moved_image_nrmse | 12 / 12 | 0 | 0 | 9.489e-06 [7.218e-06, 1.121e-05] | 5.755e-06–2.146e-05 |
| SynthMorph | FreeSurfer CPU | jacobian_fraction_nonpositive | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthMorph | FreeSurfer GPU | jacobian_fraction_nonpositive | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthMorph | PyTorch CPU | jacobian_fraction_nonpositive | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
| SynthMorph | PyTorch GPU | jacobian_fraction_nonpositive | 12 / 12 | 0 | 0 | 0 [0, 0] | 0–0 |
