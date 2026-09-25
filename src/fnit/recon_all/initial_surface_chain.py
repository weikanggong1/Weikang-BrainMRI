"""Run the validated pretess-to-quick-sphere part of recon-all in Python."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel as nib

from .extract_main_component_python import extract_main_component
from .inflate_python import inflate_surface
from .pretess_python import pretess_mgh
from .smooth_surface_python import smooth_surface
from .sphere_quick_python import write_quick_sphere
from .tessellate_gpu import tessellate_mgh, write_quad_surface


def run_initial_surface_chain(
    filled: str | Path,
    norm: str | Path,
    output_root: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    """Build both nofix quick spheres from the same filled and norm volumes.

    The CPU path matches the pinned single-subject FreeSurfer surface geometry.
    CUDA tessellation is validated separately; the connected CUDA path has not
    passed a bilateral comparison yet. The output directory must be empty.
    """
    filled, norm, root = Path(filled).resolve(), Path(norm).resolve(), Path(output_root).resolve()
    for source in (filled, norm):
        if not source.is_file():
            raise FileNotFoundError(source)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}")
    mri, surf = root / "mri", root / "surf"
    mri.mkdir(parents=True, exist_ok=True)
    surf.mkdir(parents=True, exist_ok=True)
    report: dict = {"filled": str(filled), "norm": str(norm), "device": device,
                    "hemispheres": {}}

    for hemi, label in (("lh", 255), ("rh", 127)):
        stages: list[dict] = []

        def run(name: str, function, *args, **kwargs):
            start = time.perf_counter()
            value = function(*args, **kwargs)
            stages.append({"stage": name, "seconds": time.perf_counter() - start})
            return value

        pretess = mri / f"filled-pretess{label}.mgz"
        raw = surf / f"{hemi}.orig.raw.quad"
        orig = surf / f"{hemi}.orig.nofix"
        smooth = surf / f"{hemi}.smoothwm.nofix"
        inflated = surf / f"{hemi}.inflated.nofix"
        qsphere = surf / f"{hemi}.qsphere.nofix"
        run("mri_pretess", pretess_mgh, filled, label, norm, pretess)
        def tessellate_and_write() -> None:
            vertices, quads = tessellate_mgh(pretess, label, device)
            relative_pretess = Path("../mri") / pretess.name
            write_quad_surface(raw, vertices, quads, nib.load(str(pretess)), relative_pretess)

        run("mri_tessellate", tessellate_and_write)
        run("mris_extract_main_component", extract_main_component, raw, orig)
        run("mris_smooth", smooth_surface, orig, smooth, device=device)
        run("mris_inflate", inflate_surface, smooth, inflated)
        run("mris_sphere -q", write_quick_sphere, inflated, qsphere)
        report["hemispheres"][hemi] = {
            "stages": stages,
            "outputs": {"pretess": str(pretess), "orig": str(orig),
                        "smoothwm": str(smooth), "inflated": str(inflated),
                        "qsphere": str(qsphere)},
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filled", type=Path)
    parser.add_argument("norm", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = run_initial_surface_chain(args.filled, args.norm, args.output_root,
                                       device=args.device)
    content = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
