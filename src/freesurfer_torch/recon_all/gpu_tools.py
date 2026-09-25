"""CUDA replacements for neural commands used by the pinned recon-all 8.2 run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

from ..synthseg_parc.synthseg import (
    SYNTHSEG_TIE_EPSILON, _synthseg_index_with_numerical_ties,
)


def _models() -> Path:
    external = os.environ.get("FS_TORCH_MODEL_DIR")
    root = os.environ.get("FREESURFER_HOME")
    if not external and not root:
        raise RuntimeError("Set FS_TORCH_MODEL_DIR or FREESURFER_HOME")
    models = Path(external) if external else Path(root) / "models"
    if not models.is_dir():
        raise FileNotFoundError(models)
    return models


def _device() -> str:
    device = os.environ.get("FS_TORCH_DEVICE", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{device} requested but CUDA is unavailable")
    return device


def _strip(argv: list[str]) -> None:
    from ..synthstrip import SynthStrip

    parser = argparse.ArgumentParser(prog="mri_synthstrip")
    parser.add_argument("-i", required=True)
    parser.add_argument("-o")
    parser.add_argument("-m")
    parser.add_argument("-d")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-csf", action="store_true")
    parser.add_argument("-b", type=float, default=1)
    parser.add_argument("-f", type=float)
    parser.add_argument("-g", action="store_true")
    args = parser.parse_args(argv)
    if not any((args.o, args.m, args.d)):
        parser.error("provide -o, -m, or -d")
    result = SynthStrip(weights=_models(), device=_device(), no_csf=args.no_csf,
                        threads=args.threads)(args.i, border=args.b, fill=args.f)
    for image, path in ((result.image, args.o), (result.mask, args.m),
                        (result.distance, args.d)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            image.save(path)


def _morph(argv: list[str]) -> None:
    from ..synthmorph import SynthMorph

    parser = argparse.ArgumentParser(prog="mri_synthmorph")
    parser.add_argument("-m", choices=("rigid", "affine", "deform", "joint"), default="joint")
    parser.add_argument("-t")
    parser.add_argument("-T")
    parser.add_argument("-o")
    parser.add_argument("-O")
    parser.add_argument("-i")
    parser.add_argument("-j", type=int, default=4)
    parser.add_argument("-r", type=float, default=0.5)
    parser.add_argument("-n", type=int, default=7)
    parser.add_argument("-e", type=int, choices=(192, 256), default=256)
    parser.add_argument("-g", action="store_true")
    parser.add_argument("moving")
    parser.add_argument("fixed")
    args = parser.parse_args(argv)
    if not any((args.t, args.T, args.o, args.O)):
        parser.error("provide at least one output")
    torch.set_num_threads(args.j)
    result = SynthMorph(weights=_models(), device=_device(), model=args.m,
                        extent=args.e, hyper=args.r, steps=args.n)(
                            args.moving, args.fixed, init=args.i)
    for value, path in ((result.transform, args.t), (result.inverse, args.T),
                        (result.moved, args.o), (result.fixed_moved, args.O)):
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            value.save(str(target))


def _segment(argv: list[str]) -> None:
    from ..synthseg_parc import SynthSeg

    parser = argparse.ArgumentParser(prog="mri_synthseg")
    parser.add_argument("--i", required=True)
    parser.add_argument("--o", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vol")
    parser.add_argument("--keepgeom", action="store_true")
    parser.add_argument("--addctab", action="store_true")
    parser.add_argument("--noaddctab", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--parc", action="store_true")
    args = parser.parse_args(argv)
    if args.parc:
        parser.error("the pinned recon-all call uses 33-class SynthSeg without --parc")

    device = _device()
    models = _models()
    lookup = None
    if not args.noaddctab:
        lookup = Path(os.environ["FREESURFER_HOME"]) / "FreeSurferColorLUT.txt"
        if not lookup.is_file():
            raise FileNotFoundError(lookup)
    result = SynthSeg(weights=models, device=device, threads=args.threads)(
        args.i, keep_geometry=args.keepgeom, color_lut=lookup)
    print("FS_TORCH_SYNTHSEG_NEAR_TIE"
          f" epsilon={SYNTHSEG_TIE_EPSILON}"
          f" changed_voxels={result.near_tie_voxels}",
          flush=True)
    output = Path(args.o)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.segmentation.save(str(output))

    if args.vol:
        result.write_volumes_csv(args.i, args.vol)


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        raise SystemExit("expected mri_synthstrip, mri_synthmorph or mri_synthseg")
    tool, *options = arguments
    commands = {
        "mri_synthstrip": _strip,
        "mri_synthmorph": _morph,
        "mri_synthseg": _segment,
    }
    try:
        print(f"FS_TORCH_NEURAL_TOOL {tool} device={_device()}", flush=True)
        commands[tool](options)
    except KeyError as error:
        raise SystemExit(f"unknown neural tool: {tool}") from error


if __name__ == "__main__":
    main()
