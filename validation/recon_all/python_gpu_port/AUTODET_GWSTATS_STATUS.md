# Gray/white surface threshold statistics

The Python/Numba [stage](../../../src/fnit/recon_all/autodet_gwstats_python.py) translates the default `mris_autodet_gwstats` call in pinned FreeSurfer 8.2.0 source `d932c45b7941662ea380a05efef580568b98d41a`. It reads `brain.finalsurfs.mgz`, `wm.mgz`, and either hemisphere's `orig.premesh` surface directly. It clips bright labeled white matter, identifies the 26-neighbor white/gray border, computes class standard deviations, samples intensity 1 mm inside/outside the original surface, and writes the 40-field threshold file. It does not read a native threshold file during inference.

On the fixed completed `fs_sub01`, both hemispheres produced **40/40 identical fields and byte-identical complete files** against the official recon-all output. The SHA-256 pairs are in the [validation report](autodet_gwstats_fs_sub01_report.json); [validator](validate_autodet_gwstats.py) requires exact bytes. Independent internal counts also match the official log: 13,615 clipped white voxels, 192,038 white border voxels and 216,128 gray border voxels. Two small unit checks pass in the same environment.

Run the Python stage with:

```bash
python -m fnit.recon_all.autodet_gwstats_python \
  --i SUBJECT/mri/brain.finalsurfs.mgz --wm SUBJECT/mri/wm.mgz \
  --surf SUBJECT/surf/lh.orig.premesh --o OUTPUT/autodet.gw.stats.lh.dat
```

The implementation uses NumPy, SciPy, Numba and nibabel on **CPU**; it requires no FreeSurfer executable, license or runtime files. This is a short stage: the historical official log records 3.81 s LH and 3.72 s RH. Python validation took 3.54 s LH and 2.62 s RH including I/O and JIT cache lookup. Other workloads were running, so these are observations rather than a controlled paired speed comparison. Its parity has been tested on one subject only; no end-to-end Python recon-all runner invokes it yet.
