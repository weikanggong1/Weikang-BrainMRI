"""CUDA replacements for neural commands used by the pinned recon-all 8.2 run."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np
import surfa as sf
import torch


# Cross-framework FP32 convolutions can reverse an almost exact SynthSeg tie.
# Extend argmax's lower-channel tie rule by 16 float32 unit roundoffs.
SYNTHSEG_TIE_EPSILON = 2 ** -20


def _synthseg_index_with_numerical_ties(posterior: torch.Tensor) -> torch.Tensor:
    peak = posterior.amax(dim=0, keepdim=True)
    return (posterior >= peak - SYNTHSEG_TIE_EPSILON).to(torch.uint8).argmax(dim=0)


def _models() -> Path:
    root = os.environ.get("FREESURFER_HOME")
    if not root:
        raise RuntimeError("FREESURFER_HOME must point to the packaged runtime")
    models = Path(root) / "models"
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
    from ..synthseg_parc.postprocess import postprocess_segmentation
    from ..synthseg_parc.preprocess import preprocess_t1
    from ..synthseg_parc.segment import SynthSegSegmenter

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

    torch.set_num_threads(args.threads)
    device = _device()
    models = _models()
    prepared = preprocess_t1(args.i, device=device)
    raw_labels = np.load(models / "synthseg_segmentation_labels_2.0.npy")
    unique_labels, unique_indices = np.unique(raw_labels, return_index=True)
    topology = torch.as_tensor(
        np.load(models / "synthseg_topological_classes_2.0.npy")[unique_indices],
        device=device,
    )
    segmenter = SynthSegSegmenter(
        models / "synthseg_2.0.h5",
        models / "synthseg_segmentation_labels_2.0.npy", device=device,
    )
    posterior = segmenter.posterior(prepared.image)
    ordinary_labels, posterior = postprocess_segmentation(
        posterior, segmenter.labels, topology, prepared.content_slices)
    labels = segmenter.labels[_synthseg_index_with_numerical_ties(posterior)]
    print("FS_TORCH_SYNTHSEG_NEAR_TIE"
          f" epsilon={SYNTHSEG_TIE_EPSILON}"
          f" changed_voxels={int(torch.count_nonzero(labels != ordinary_labels))}",
          flush=True)
    aligned_affine = prepared.aligned_affine.copy()
    aligned_affine[:3, 3] += aligned_affine[:3, :3] @ np.asarray(
        [part.start for part in prepared.content_slices])

    data = labels.to(torch.int32).cpu().numpy()
    source = sf.Volume(
        data, geometry=sf.ImageGeometry(shape=data.shape, vox2world=aligned_affine))
    if args.keepgeom:
        source = source.resample_like(sf.load_volume(args.i), method="nearest")
    if not args.noaddctab:
        lookup = Path(os.environ["FREESURFER_HOME"]) / "FreeSurferColorLUT.txt"
        if not lookup.is_file():
            raise FileNotFoundError(lookup)
        source.labels = sf.load_label_lookup(str(lookup))
    output = Path(args.o)
    output.parent.mkdir(parents=True, exist_ok=True)
    source.save(str(output))

    if args.vol:
        # FreeSurfer's non-parcellated 2.0 CSV places total intracranial
        # volume first, then 32 foreground soft volumes in label order.
        soft = posterior[1:].sum(dim=(1, 2, 3)).cpu().numpy()
        voxel_volume = abs(np.linalg.det(aligned_affine[:3, :3]))
        values = np.around(np.concatenate(([soft.sum()], soft)) * voxel_volume, 3)
        raw_names = np.load(models / "synthseg_segmentation_names_2.0.npy")
        names = raw_names[unique_indices]
        assert len(unique_labels) == len(names) == len(soft) + 1
        csv_path = Path(args.vol)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["subject", "total intracranial", *names[1:]])
            writer.writerow([Path(args.i).name.replace(".nii.gz", ""),
                             *(str(float(value)) for value in values)])


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
