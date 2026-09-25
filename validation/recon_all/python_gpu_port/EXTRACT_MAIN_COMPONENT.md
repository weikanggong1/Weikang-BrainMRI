# Main surface component

`extract_main_component_python.py` translates the fixed recon-all
`mris_extract_main_component` call and its FreeSurfer 8.2.0 quad split rule.
The default native reader chooses a quad diagonal from both leading vertex
indices; nibabel's old-quad reader uses a different rule. The Python stage
then retains the largest connected component, preserving source vertex and
face order. This stage runs on NumPy/SciPy CPU.

| Hemisphere | Vertices | Triangles | Native median | Python median |
| --- | ---: | ---: | ---: | ---: |
| Left | 102,764 | 205,560 | 0.686 s | 0.135 s |
| Right | 101,454 | 202,936 | 0.413 s | 0.059 s |

The [three-trial same-host report](extract_main_component_cpu_report.json)
uses freshly replayed native quad surfaces as identical input. After the
variable `created by ...` timestamp, **every output byte** matches native:
counts, vertices, triangle indices, volume geometry and surface coordinate
tags. FreeSurfer's `mris_info` loads both Python outputs. The two real
hemispheres each have one component; a separate synthetic surface with two
equal-size disconnected components matched native byte for byte after the
timestamp and retained the first component. Two
[focused tests](test_extract_main_component_python.py) also cover the quad
split and tie rule.

The chained Python `mri_pretess → mri_tessellate →
mris_extract_main_component` geometry matches the native triangular surface
for both hemispheres: 0 differing vertex or face bytes. The later bilateral
[six-stage replay](SMOOTH_SURFACE.md) uses the native relative input name,
so its volume-geometry tags match through `qsphere.nofix` as well. Full
surface hashes still differ in provenance text. These are isolated replays
from retained `filled.mgz` and `norm.mgz`; upstream filling and later
topology/surface stages remain separate gates.
