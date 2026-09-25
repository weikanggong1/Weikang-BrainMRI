"""Verify the fixed cortical labels against one completed FreeSurfer subject."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from fnit.recon_all.label_cortex_fix_ga_python import label_cortex_fix_ga
from fnit.recon_all.label_cortex_python import label_cortex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for hemi in ("lh", "rh"):
        surface = args.subject / "surf" / f"{hemi}.white.preaparc"
        aseg = args.subject / "mri" / "aseg.presurf.mgz"
        entowm = args.subject / "mri" / "entowm.mgz"
        for name, generate in (
            ("cortex", lambda path: label_cortex_fix_ga(surface, aseg, entowm,
                                                          hemi, path)),
            ("cortex+hipamyg", lambda path: label_cortex(surface, aseg, path,
                                                          keep_hip_amyg=True)),
        ):
            output = args.output_dir / f"{hemi}.{name}.label"
            start = perf_counter()
            generate(output)
            seconds = perf_counter() - start
            reference = args.subject / "label" / output.name
            exact = output.read_bytes() == reference.read_bytes()
            print(f"{hemi}.{name}: {seconds:.2f} s, exact={exact}")
            if not exact:
                raise SystemExit(f"label differs: {output} vs {reference}")


if __name__ == "__main__":
    main()
