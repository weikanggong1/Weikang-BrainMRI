"""Check all 72 fixed fsaverage surface-label transfers against a subject."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from fnit.recon_all.label2label_surface_python import SurfaceLabelMapper


BA = ("BA1_exvivo", "BA2_exvivo", "BA3a_exvivo", "BA3b_exvivo",
      "BA4a_exvivo", "BA4p_exvivo", "BA6_exvivo", "BA44_exvivo",
      "BA45_exvivo", "V1_exvivo", "V2_exvivo", "MT_exvivo",
      "entorhinal_exvivo", "perirhinal_exvivo")
VP = ("FG1", "FG2", "FG3", "FG4", "hOc1", "hOc2", "hOc3v", "hOc4v")
NAMES = (tuple(f"{name}.label" for name in BA)
         + tuple(f"{name}.mpm.vpnl.label" for name in VP)
         + tuple(f"{name}.thresh.label" for name in BA))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fsaverage", type=Path)
    parser.add_argument("subject", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    durations = []
    failures = []
    for hemi in ("lh", "rh"):
        start = perf_counter()
        mapper = SurfaceLabelMapper(args.fsaverage / "surf" / f"{hemi}.sphere.reg",
                                    args.subject / "surf" / f"{hemi}.sphere.reg",
                                    args.subject / "surf" / f"{hemi}.white",
                                    args.subject.name)
        print(f"{hemi} setup {perf_counter() - start:.3f} s", flush=True)
        for name in NAMES:
            label_name = f"{hemi}.{name}"
            output = args.output_dir / label_name
            start = perf_counter()
            mapper.map_label(args.fsaverage / "label" / label_name, output)
            seconds = perf_counter() - start
            durations.append(seconds)
            reference = args.subject / "label" / label_name
            exact = output.read_bytes() == reference.read_bytes()
            print(f"{label_name}\t{seconds:.3f}\t{exact}", flush=True)
            if not exact:
                failures.append(label_name)
    print(f"exact={len(durations) - len(failures)}/{len(durations)} "
          f"map_total={sum(durations):.3f}s map_median={median(durations):.3f}s "
          f"map_max={max(durations):.3f}s", flush=True)
    if failures:
        raise SystemExit(f"mismatched labels: {', '.join(failures)}")


if __name__ == "__main__":
    main()
