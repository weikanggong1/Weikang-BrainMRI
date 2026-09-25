# Cortical statistics numeric headers

`anatomical_stats_global.py` reads the 16 values in `brainvol.stats`, computes
the cortex vertex count, area and mean thickness, and uses the Python eTIV
port to format the 11 numeric/global header lines of
`mris_anatomical_stats -no-th3`. For the calls with `-cortex`, FreeSurfer
accumulates cortex area and thickness in **vertex order with float32
additions**; summing in float64 shifted the one-decimal area header by as
much as 0.5 mm² on this subject.

On frozen `fs_sub01` inputs, all **88/88 lines** matched native text exactly
across bilateral aparc, aparc.a2009s and aparc.DKTatlas white tables and
bilateral aparc pial tables. See the
[report](anatomical_stats_global_cpu_report.json) and
[probe](experimental/anatomical_stats_global_probe.py).

The four bilateral BA/exvivo calls have no `-cortex` option. Their all-vertex
area uses the total surface area and their headers omit mean thickness. With
the frozen `brainvol.stats` cache, the Python writer matched all **36/36
Measure lines** across those four tables, alongside 56/56 data lines.

This exact-text replay reads the official `brainvol.stats` cache. A separate
Python [`ComputeBrainVolumeStats2` port](BRAIN_VOLUME_STATS.md) now reproduces
its 16 numerical measures within 0.000295 mm³ on this subject; supplying its
result to the header formatter therefore gives numerical agreement, with
surface-volume differences in the sixth-decimal text. The required final
segmentation and surfaces still have to come from the Python reconstruction.
Runtime metadata such as timestamps and hostname are not compared.
