"""Audit a complete native sphere capture against frozen inputs and official output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from audit_standard_sphere_file import _sections


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hemisphere", choices=("lh", "rh"))
    parser.add_argument("prefix_surf", type=Path)
    parser.add_argument("complete_surf", type=Path)
    parser.add_argument("official_sphere", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    hemi = args.hemisphere
    input_hashes = {}
    for name in ("inflated", "smoothwm"):
        suffix = f"{hemi}.{name}"
        earlier = _sha(args.prefix_surf / suffix)
        complete = _sha(args.complete_surf / suffix)
        if earlier != complete:
            raise ValueError(f"{suffix} changed between captures")
        input_hashes[name] = earlier
    snapshots = sorted(args.complete_surf.glob(f"{hemi}.sphere[0-9][0-9][0-9][0-9]"))
    old = sorted(args.prefix_surf.glob(f"{hemi}.sphere[0-9][0-9][0-9][0-9]"))
    if not snapshots or len(old) > len(snapshots):
        raise ValueError("missing complete-capture checkpoints")
    first_difference = None
    for index, old_path in enumerate(old):
        new_path = args.complete_surf / old_path.name
        earlier, complete = _sections(old_path), _sections(new_path)
        if (earlier["xyz_sha256"] != complete["xyz_sha256"] or
                earlier["faces_sha256"] != complete["faces_sha256"]):
            first_difference = index
            break
    final = _sections(args.complete_surf / f"{hemi}.sphere")
    official = _sections(args.official_sphere)
    log = (args.complete_surf.parent / "native_complete.log").read_text()
    timing = re.search(r"native_wall_seconds=([0-9.]+)", log)
    result = {"hemisphere": hemi, "input_sha256": input_hashes,
              "native_checkpoint_count": len(snapshots),
              "frozen_prefix_checkpoint_count": len(old),
              "first_prefix_geometry_difference": first_difference,
              "native_wall_seconds_including_snapshot_io": (
                  float(timing.group(1)) if timing else None),
              "complete_final": final, "official_final": official,
              "final_comparison": {
                  "ordered_vertices_exact": final["xyz_sha256"] == official["xyz_sha256"],
                  "ordered_faces_exact": final["faces_sha256"] == official["faces_sha256"],
                  "volume_geometry_exact": (final["volume_info_sha256"]
                                            == official["volume_info_sha256"]),
              }}
    output = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
