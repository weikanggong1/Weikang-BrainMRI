# `mris_label2annot` fixed label-to-annotation calls

`label2annot_python.py` implements the six fixed `--maxstatwinner` calls for
bilateral `mpm.vpnl`, `BA_exvivo`, and `BA_exvivo.thresh` annotations. It reads
the ordered vertex count from each `orig` surface, the projected label files,
and the corresponding downloaded FreeSurfer 8.2 color table. Equal maximum
stats select the later label, and vertices with no winning label receive the
color table's unknown value. The output includes the binary color table and
the same per-vertex annotation order as the native `.annot` format.

On `fs_sub01`, all **six files matched the archived official files byte for
byte**. A fresh native replay of `lh.BA_exvivo.annot` also matched the archived
file and the Python output. Replacing all 72 projected input labels with the
Python-generated, previously byte-verified label files preserved exact output
for all six annotations. The original surfaces and `sphere.reg` files remain
frozen official inputs; this is a chained label/annotation check, not an
end-to-end reconstruction.

| Output | Vertices | Native/Python byte differences |
| --- | ---: | ---: |
| `lh.mpm.vpnl.annot` | 106,622 | 0 |
| `lh.BA_exvivo.annot` | 106,622 | 0 |
| `lh.BA_exvivo.thresh.annot` | 106,622 | 0 |
| `rh.mpm.vpnl.annot` | 105,541 | 0 |
| `rh.BA_exvivo.annot` | 105,541 | 0 |
| `rh.BA_exvivo.thresh.annot` | 105,541 | 0 |

One same-host headcw fresh CLI comparison for the left BA call took **0.564 s
native and 0.263 s Python**. Both used the same subject surface, 14 labels,
and color table; this single short-stage pair does not establish an
end-to-end speedup. The focused tie/unknown test passed.

```bash
python -m fnit.recon_all.label2annot_python \
  /path/to/surf/lh.orig /path/to/assets/average/colortable_BA.txt \
  /path/to/label/lh.BA_exvivo.annot \
  /path/to/label/lh.BA1_exvivo.label ...
```
