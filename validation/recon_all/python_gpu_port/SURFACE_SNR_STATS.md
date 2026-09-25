# Cortical white/gray contrast SNR statistics

`segstats_surface_snr_python.py` replaces the two fixed recon-all
`mri_segstats --in surf/<hemi>.w-g.pct.mgh --annot <subject> <hemi> aparc
--snr` calls. It consumes the frozen annotation, intensity map and white
surface area map, and runs on CPU without a FreeSurfer executable. It follows
the pinned FreeSurfer 8.2 sample-standard-deviation expression and its
ordered float32 accumulation of vertex area.

For `fs_sub01`, the 35 left and 35 right data lines matched both the archived
official tables and fresh native replays **byte for byte**. This includes all
11 fields per row and the `-nan` SNR of the unknown region. SHA-256 of the
35-line data sections was `f3b7966d05453861c51686675323db1ac224f84f82b34fb17dfc68809b88fd6d`
on the left and `8ac56b29d55420dd92d8c8fabc53726d2c83a0ad08e2edb2d44c23abf30194a8`
on the right, identical for native and Python. Command, temporary color-table,
host and timestamp headers are not reproduced.

| Same-input headcw cold CLI | Native | Python CPU |
| --- | ---: | ---: |
| Left hemisphere | 0.37 s | 0.20 s |
| Right hemisphere | 0.35 s | 0.23 s |

These are one same-host call per side with input reads and output writes. They
do not measure the preceding `pctsurfcon` map generation or complete
reconstruction. The frozen input hashes were `lh/rh.w-g.pct.mgh`
`74f5abf4be497c8d125e9c62df689fe532c8ed8e3c7b442e4228127946033e13`/
`5d40683eed5738c1eef7b1ee8270be1ed9cdc89e2c4e8edef399e97351908597`,
`lh/rh.aparc.annot`
`9b4b6dce47037a623ca4bbab6564cd3bd31369c22c2e7c55d305a85c46517536`/
`aafe54c8b3ae84fb90ce38f766637ee2b80f0a2549cfde2d80698eacc89646de`,
and `lh/rh.area`
`f5a8545e6656938f7019308453b31e7a964abc9ae0730fbe40172aa8e2365a65`/
`5d91003ad60b11aa1f5b0c47825edf4c70675e0f54e1eb70faf4caf159ebfa0b`.

```bash
python -m fnit.recon_all.segstats_surface_snr_python \
  /path/to/subjects/fs_sub01 lh \
  /path/to/subjects/fs_sub01/stats/lh.w-g.pct.stats
```

An additional replay fed Python-generated contrast maps into this writer;
all 70 numeric lines still matched. The annotations, white surfaces,
thickness maps and cortex labels remain frozen official upstream inputs,
so this comparison does not establish end-to-end subject parity.
