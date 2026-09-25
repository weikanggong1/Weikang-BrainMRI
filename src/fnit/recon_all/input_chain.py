"""Connected single-T1 import, single-run copy, and conform stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

from .conform_gpu import add_xform_to_header, conform_volume
from .nifti_import import import_t1


def run_input_chain(t1: str | Path, subject_dir: str | Path,
                    *, device: str = "cpu") -> dict:
    """Create the first three recon-all MRI files from one float32 T1 NIfTI.

    The XFORM tag records the future ``talairach.xfm`` path. SynthMorph must
    create that transform before downstream stages use it.
    """
    root = Path(subject_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("subject_dir must be empty")
    original = root / "mri" / "orig" / "001.mgz"
    rawavg = root / "mri" / "rawavg.mgz"
    conformed = root / "mri" / "orig.mgz"
    xform = root / "mri" / "transforms" / "talairach.xfm"
    original.parent.mkdir(parents=True, exist_ok=True)
    xform.parent.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    started = time.perf_counter()
    import_t1(t1, original)
    timings["import_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    shutil.copyfile(original, rawavg)
    timings["single_run_copy_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    conform_volume(rawavg, conformed, device=device)
    add_xform_to_header(conformed, conformed, xform)
    timings["conform_and_xform_tag_seconds"] = time.perf_counter() - started
    return {
        "subject_dir": str(root),
        "original": str(original),
        "rawavg": str(rawavg),
        "conformed": str(conformed),
        "talairach_xfm_path": str(xform),
        "device": device,
        **timings,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("t1")
    parser.add_argument("subject_dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    print(run_input_chain(args.t1, args.subject_dir, device=args.device))


if __name__ == "__main__":
    main()
