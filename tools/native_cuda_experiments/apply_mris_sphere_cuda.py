#!/usr/bin/env python3
"""Apply the opt-in CUDA smoother patch to FreeSurfer d932c45 source."""

import argparse
import shutil
import subprocess
from pathlib import Path


SOURCE_REVISION = "d932c45b7941662ea380a05efef580568b98d41a"
HERE = Path(__file__).resolve().parent


def git(source, *args):
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="FreeSurfer source checkout")
    parser.add_argument("--check", action="store_true", help="verify applicability without editing")
    args = parser.parse_args()
    source = args.source.resolve()
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        parser.error(f"source must be at FreeSurfer commit {SOURCE_REVISION}")

    destination = source / "mris_sphere/fs_cuda_average_gradients.cu"
    if destination.exists():
        parser.error(f"CUDA source already exists: {destination}")
    patch = HERE / "mris_sphere_cuda.patch"
    git(source, "apply", "--check", str(patch))
    if not args.check:
        git(source, "apply", str(patch))
        shutil.copyfile(HERE / "fs_cuda_average_gradients.cu", destination)
    print("CUDA mris_sphere patch applicable" if args.check else "CUDA mris_sphere patch applied")


if __name__ == "__main__":
    main()
