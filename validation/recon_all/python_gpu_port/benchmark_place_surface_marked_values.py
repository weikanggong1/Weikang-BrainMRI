"""Compare instrumented native border-value averaging against PyTorch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.place_surface_smoothing import average_marked_values


FIELDS = np.dtype([("value", "<f4"), ("marked", "<i4"), ("ripped", "<i4")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--native-dumps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = {"device": args.device, "iterations": 5, "checks": {}}
    for hemi in ("lh", "rh"):
        for surface in ("white", "pial"):
            prefix = args.native_dumps / f"{hemi + '_' if hemi == 'rh' else ''}{surface}_marked"
            before = np.fromfile(f"{prefix}.before", dtype=FIELDS)
            after = np.fromfile(f"{prefix}.after", dtype=FIELDS)
            input_name = "orig" if surface == "white" else "white"
            _, faces = fs.read_geometry(args.subject / "surf" / f"{hemi}.{input_name}")
            started = time.perf_counter()
            predicted = average_marked_values(
                before["value"], before["marked"], before["ripped"], faces, 5, device=args.device
            )
            seconds = time.perf_counter() - started
            delta = np.abs(predicted - after["value"])
            report["checks"][f"{hemi}.{surface}"] = {
                "vertices": len(before),
                "native_changed_values": int(np.count_nonzero(before["value"] != after["value"])),
                "mismatched_values": int(np.count_nonzero(delta)),
                "max_abs": float(delta.max()),
                "metadata_equal": bool(
                    np.array_equal(before["marked"], after["marked"])
                    and np.array_equal(before["ripped"], after["ripped"])
                ),
                "python_seconds": seconds,
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
