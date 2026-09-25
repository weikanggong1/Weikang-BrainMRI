"""Three paired FreeSurfer 8.2 mri_pretess calls on frozen subject inputs."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.pretess_python import pretess_mgh


def compare(native: Path, candidate: Path) -> dict:
    original = nib.load(str(native))
    actual = nib.load(str(candidate))
    voxels = np.asanyarray(original.dataobj)
    differences = int(np.count_nonzero(voxels != np.asanyarray(actual.dataobj)))
    original_raw = gzip.decompress(native.read_bytes())
    candidate_raw = gzip.decompress(candidate.read_bytes())
    core_end = int(original.header.get_data_offset()) + voxels.size * voxels.dtype.itemsize
    return {"voxels": int(voxels.size), "voxel_differences": differences,
            "header_and_voxel_bytes_identical": original_raw[:core_end] == candidate_raw[:core_end],
            "decompressed_mgh_identical": original_raw == candidate_raw,
            "max_affine_error": float(np.max(np.abs(original.affine - actual.affine)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-mri", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {name: args.subject_mri / name for name in
              ("wm.asegedit.mgz", "filled.mgz", "norm.mgz")}
    calls = (("wm", inputs["wm.asegedit.mgz"], "wm"),
             ("lh", inputs["filled.mgz"], "255"),
             ("rh", inputs["filled.mgz"], "127"))
    rows = []
    for name, segmentation, label in calls:
        timings = {"native": [], "python": []}
        comparison = None
        edits = None
        for repeat in range(args.repeats):
            outputs = {side: args.output_dir / f"{name}.{side}.{repeat}.mgz"
                       for side in timings}
            for side in (("native", "python") if repeat % 2 == 0
                         else ("python", "native")):
                start = time.perf_counter()
                if side == "native":
                    subprocess.run([str(args.native_binary), str(segmentation),
                                    label, str(inputs["norm.mgz"]), str(outputs[side])],
                                   capture_output=True, check=True)
                else:
                    edits = pretess_mgh(segmentation, label, inputs["norm.mgz"],
                                        outputs[side])
                timings[side].append(time.perf_counter() - start)
            comparison = compare(outputs["native"], outputs["python"])
            if comparison["voxel_differences"] or not comparison["header_and_voxel_bytes_identical"]:
                raise ValueError(f"{name} repeat {repeat} differs: {comparison}")
        rows.append({"call": name, "label": label, "python_topology_edits": edits,
                     "comparison": comparison, "wall_seconds": timings,
                     "median_seconds": {side: statistics.median(values)
                                        for side, values in timings.items()}})
    report = {"host": platform.node(),
              "native_binary": str(args.native_binary),
              "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in inputs.items()},
              "repeats": args.repeats,
              "timing_scope": "same-host file I/O; native process startup included; Python imports excluded",
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
