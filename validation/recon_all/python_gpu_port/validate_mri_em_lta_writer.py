"""Check Python LTA serialization using the independently captured native matrix."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all.mri_em_register import read_gca, read_masked_input
from fnit.recon_all.mri_em_register_lta import write_voxel_lta
from validate_mri_em_optimizer import read_matrix


def canonical(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("type "))
    return [" ".join(line.split()) for line in lines[index:] if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("nu", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("reference_lta", type=Path)
    args = parser.parse_args()
    atlas = read_gca(args.atlas)
    masked = read_masked_input(args.nu, args.mask)
    matrix = read_matrix(args.reference_lta)
    with TemporaryDirectory() as directory:
        candidate = Path(directory) / "talairach.lta"
        write_voxel_lta(candidate, matrix, args.nu, args.atlas, atlas, masked)
        expected, observed = canonical(args.reference_lta), canonical(candidate)
    differing = [i for i, (a, b) in enumerate(zip(expected, observed)) if a != b]
    differing.extend(range(min(len(expected), len(observed)), max(len(expected), len(observed))))
    print(json.dumps({"source_lines": len(expected), "candidate_lines": len(observed),
                      "differing_numeric_or_geometry_lines": differing,
                      "matrix": np.asarray(matrix).tolist()}, indent=2))
    return int(bool(differing))


if __name__ == "__main__":
    raise SystemExit(main())
