"""Create a linked official subject with one altered LH pial area vertex."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from nibabel.freesurfer import read_morph_data, write_morph_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("existing_thickness_negative", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    shutil.copytree(args.existing_thickness_negative, args.output, symlinks=True)
    thickness = args.output / "surf/lh.thickness"
    thickness.unlink()
    thickness.symlink_to(args.reference / "surf/lh.thickness")
    area = args.output / "surf/lh.area.pial"
    area.unlink()
    values = read_morph_data(str(args.reference / "surf/lh.area.pial"))
    values[1234] += 0.02
    write_morph_data(str(area), values)
    print(f"altered vertex 1234 by +0.02 mm^2: {area}")


if __name__ == "__main__":
    main()
