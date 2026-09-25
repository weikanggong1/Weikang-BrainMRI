"""Link a reference subject and change one surface metric vertex."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

from nibabel.freesurfer import read_morph_data, write_morph_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metric", choices=("thickness", "area.pial"), required=True)
    parser.add_argument("--vertex", type=int, default=1234)
    parser.add_argument("--delta", type=float, default=0.02)
    args = parser.parse_args()
    for folder in ("mri", "surf", "label", "stats"):
        shutil.copytree(args.reference / folder, args.output / folder,
                        symlinks=True, copy_function=os.symlink)
    target = args.output / "surf" / f"lh.{args.metric}"
    target.unlink()
    values = read_morph_data(str(args.reference / "surf" / target.name))
    values[args.vertex] += args.delta
    write_morph_data(str(target), values)
    print(target)


if __name__ == "__main__":
    main()
