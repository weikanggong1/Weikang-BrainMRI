"""Collect six-atlas mris_ca_label parity and timing evidence on frozen inputs."""

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path


CASES = (
    ("lh", "DKaparc", "aparc.annot", "full_lh"),
    ("rh", "DKaparc", "aparc.annot", "full_rh"),
    ("lh", "CDaparc", "aparc.a2009s.annot", "full_lh_cd"),
    ("rh", "CDaparc", "aparc.a2009s.annot", "full_rh_cd"),
    ("lh", "DKTaparc", "aparc.DKTatlas.annot", "full_lh_dkt"),
    ("rh", "DKTaparc", "aparc.DKTatlas.annot", "full_rh_dkt"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("subject", type=Path)
    parser.add_argument("probe", type=Path)
    args = parser.parse_args()
    inputs = {"aseg.presurf.mgz": sha256(args.subject / "mri" / "aseg.presurf.mgz"),
              "ic4.tri": sha256(args.bundle / "lib/bem/ic4.tri"),
              "ic7.tri": sha256(args.bundle / "lib/bem/ic7.tri")}
    results = []
    for hemi, atlas, annot, directory in CASES:
        atlas_path = args.bundle / "average" / f"{hemi}.{atlas}.atlas.acfb40.noaparc.i12.2016-08-02.gcs"
        output_name = f"{hemi}.{annot}"
        python_path = args.probe / directory / f"{hemi}.{annot[:-6]}.python.annot"
        if not python_path.exists():
            raise FileNotFoundError(python_path)
        official_path = args.subject / "label" / output_name
        native_path = args.probe / "native_no_snapshot" / output_name
        native_log = (args.probe / "native_no_snapshot" / f"{hemi}.{atlas}.log").read_text()
        native_wall = float(re.search(r"^real ([0-9.]+)$", native_log, re.MULTILINE).group(1))
        python_run = json.loads((args.probe / directory / "end_to_end.python.json").read_text())
        hashes = {"python": sha256(python_path), "native": sha256(native_path),
                  "official": sha256(official_path)}
        results.append({"hemi": hemi, "atlas": atlas, "annotation": output_name,
                        "vertices": python_run["vertices"],
                        "input_sha256": {
                            "smoothwm": sha256(args.subject / "surf" / f"{hemi}.smoothwm"),
                            "sphere.reg": sha256(args.subject / "surf" / f"{hemi}.sphere.reg"),
                            "cortex.label": sha256(args.subject / "label" / f"{hemi}.cortex.label"),
                            "atlas.gcs": sha256(atlas_path)},
                        "output_sha256": hashes,
                        "byte_identical": len(set(hashes.values())) == 1,
                        "python_seconds": python_run["seconds"],
                        "native_seconds_no_snapshots": native_wall,
                        "gibbs_history": python_run["gibbs_history"],
                        "islands_history": python_run["islands_history"]})
    print(json.dumps({"hostname": platform.node(), "build": "8.2.0-20260314-d932c45",
                      "fixed_subject": args.subject.name, "common_input_sha256": inputs,
                      "cases": results}, indent=2))


if __name__ == "__main__":
    main()
