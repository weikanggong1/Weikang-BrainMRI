# Python cortical statistics file

`anatomical_stats_file.py` writes a stats-style file from existing Python
segmentation/surface outputs: it combines the ten-column ROI table with the
numeric global measures, using the Python `ComputeBrainVolumeStats2` and eTIV
ports. Metadata identifies `fnit`; timestamps, hostname and
historical native command lines are intentionally not copied.

On frozen official `fs_sub01` inputs, the independent Python writer's left
aparc white table had **34/34 data lines exactly identical** to the official
file. Nine of ten numeric global fields were exact at the displayed precision;
`CortexVol` differed by **0.000295 mm³** because the surface volume integral
uses a slightly different floating-point accumulation. See the
[report](anatomical_stats_file_cpu_report.json) and
[probe](experimental/anatomical_stats_file_probe.py).

The same writer now handles the four bilateral BA/exvivo white tables without
a cortex-label restriction. With the official brain-volume cache as input,
all **56/56 data lines and 36/36 numeric Measure lines** matched the native
files exactly. This is still an isolated downstream replay.

This writer currently requires the final aseg, white/pial meshes, morphometry
maps and atlas annotations as inputs. Most of those upstream files are still
produced by native stages in the frozen reference, so this is a validated
downstream Python stage, not an end-to-end reconstruction result.
