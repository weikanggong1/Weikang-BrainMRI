"""Compare isolated PyTorch Talairach output with the archived hybrid result."""

import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import surfa as sf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_xfm(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = lines.index("Linear_Transform =") + 1
    rows = [[float(x) for x in line.strip().rstrip(";").split()]
            for line in lines[start:start + 3]]
    return np.array(rows + [[0, 0, 0, 1]], np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moving", type=Path)
    parser.add_argument("--module-file", required=True, type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("reference_aff", type=Path)
    parser.add_argument("reference_xfm", type=Path)
    parser.add_argument("candidate_aff", type=Path)
    parser.add_argument("candidate_xfm", type=Path)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("talairach_synthmorph", args.module_file)
    talairach_synthmorph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(talairach_synthmorph)
    ref_aff = sf.load_affine(str(args.reference_aff))
    got_aff = sf.load_affine(str(args.candidate_aff))
    ref_xfm = read_xfm(args.reference_xfm)
    got_xfm = read_xfm(args.candidate_xfm)
    ref_reconverted = talairach_synthmorph.talairach_matrix(ref_aff)
    corners = np.array(np.meshgrid(*[[0, 255]] * 3, indexing="ij")).reshape(3, -1)
    source = sf.load_volume(str(args.moving)).geom.vox2world.matrix
    points = source[:3, :3] @ corners + source[:3, 3, None]
    hom = np.vstack((points, np.ones(points.shape[1])))
    displacement = (got_xfm[:3] @ hom) - (ref_xfm[:3] @ hom)
    timings = {}
    if args.run_log:
        log = args.run_log.read_text()
        for key, pattern in {
            "elapsed": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)",
            "peak_rss_kib": r"Maximum resident set size \(kbytes\): (\d+)",
        }.items():
            match = re.search(pattern, log)
            if match:
                timings[key] = int(match.group(1)) if key == "peak_rss_kib" else match.group(1)
    report = {
        "scope": "published hybrid rca-talairach: FreeSurfer script and lta_convert, PyTorch SynthMorph neural tool",
        "device": "cpu",
        "input_sha256": {"synthstrip": sha256(args.moving),
                         "mni305_stripped": sha256(args.template),
                         "affine_weights": sha256(args.weights)},
        "reference_sha256": {"aff_lta": sha256(args.reference_aff),
                              "talairach_xfm": sha256(args.reference_xfm)},
        "candidate_sha256": {"aff_lta": sha256(args.candidate_aff),
                              "talairach_xfm": sha256(args.candidate_xfm)},
        "implementation_sha256": {"talairach": sha256(Path(talairach_synthmorph.__file__))},
        "reference_affine_matrix": ref_aff.matrix.tolist(),
        "candidate_affine_matrix": got_aff.matrix.tolist(),
        "reference_xfm_matrix": ref_xfm.tolist(),
        "candidate_xfm_matrix": got_xfm.tolist(),
        "affine_matrix_max_abs_error": float(np.max(np.abs(ref_aff.matrix - got_aff.matrix))),
        "xfm_matrix_max_abs_error": float(np.max(np.abs(ref_xfm[:3] - got_xfm[:3]))),
        "xfm_corner_rms_mm": float(np.sqrt(np.mean(np.sum(displacement ** 2, axis=0)))),
        "xfm_corner_max_mm": float(np.max(np.linalg.norm(displacement, axis=0))),
        "archived_affine_reconverted_xfm_max_abs_error": float(np.max(np.abs(ref_reconverted[:3] - ref_xfm[:3]))),
        "timing": timings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key.endswith("error") or key in ("timing", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
