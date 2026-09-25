# `mris_jacobian` isolated replacement

The fixed recon-all call maps `lh/rh.white.preaparc` to `lh/rh.sphere.reg` and
writes `lh/rh.jacobian_white`. `surface_jacobian_gpu.py` computes the ratio of
mapped and original per-vertex triangle areas, divided by the ratio of total
areas, as in pinned FreeSurfer 8.2 source
`mris_jacobian/mris_jacobian.cpp`. The inputs are ordered surface pairs with
identical faces. No FreeSurfer executable is called by the Python step.

On the frozen `fs_sub01` inputs, a fresh official headcw run reproduced each
archived native morphometry array exactly. The Python CPU and H100 CUDA 1
outputs met the per-vertex acceptance bound of `1e-5` for **212,163/212,163**
vertices. The largest CPU difference was `1.43e-6`; the largest CUDA
difference was `2.86e-6`. The outputs are numerically consistent, not byte
identical. Two synthetic unit tests also passed on headcw.

| Hemisphere | Vertices | Native headcw CLI | Python CPU headcw CLI | Python H100 CLI | Max CPU error | Max H100 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Left | 106,622 | 0.53 s | 2.08 s | 5.53 s | 1.43e-6 | 1.43e-6 |
| Right | 105,541 | 0.51 s | 1.79 s | 5.74 s | 1.43e-6 | 2.86e-6 |

Each CLI time includes process startup, surface I/O, computation, and output
write. The native/CPU pairs ran on headcw; CUDA ran on gpucw1 with a shared
H100, so the table is not a same-host native/CUDA speed comparison. GPU
offloading is unnecessary for this short stage unless the data remain resident
for surrounding steps.

Reproduce from the repository checkout with:

```bash
python -m fnit.recon_all.surface_jacobian_gpu \
  /path/to/lh.white.preaparc /path/to/lh.sphere.reg \
  /path/to/lh.jacobian_white.python --device cpu
```

The original and mapped input SHA-256 values for the left hemisphere are
`0d5699ce744114c0e6f39449bab394d1926cc18fcb9fe8c48bb10d5b88fbf797` and
`801d345eb11c4b02e64aa453bc388f18e8cd47e8bb0beed7d75868d92276b8f8`;
for the right hemisphere they are
`363f9a0124802a33e41475bd0aec9296da85709719002d8e70712a703cd453be` and
`0e2a10093721f2f75f7460c559aa982560abd44662a9dac16865d42cb419fa8b`.
This is an isolated replay using official upstream surfaces. It does not
establish an end-to-end native-free recon-all run.
