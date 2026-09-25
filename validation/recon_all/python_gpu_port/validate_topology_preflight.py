"""Compare Python pre-genetic topology diagnostics with pinned FreeSurfer 8.2.

The native reference directory is produced by ``mris_fix_topology -diagonly``
with ``DIAG=0x8 DIAG_VERBOSE=1`` on a separate subject copy.  This validator
only reads existing data; it never runs reconstruction or native executables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_preflight_python import (
    center_sphere,
    defect_border_labels,
    defect_component_labels,
    defect_hull_labels,
    defect_retention_status,
    genetic_base_translation,
    genetic_candidate_base_prune,
    project_and_smooth_sphere,
    topology_counts,
)


SOURCE_COMMIT = "d932c45b7941662ea380a05efef580568b98d41a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(subject: Path, diagnostics: Path) -> dict:
    surf = subject / "surf"
    report = {"source_commit": SOURCE_COMMIT,
              "scope": "spherical preprocessing and pre-genetic defect diagnostics only",
              "hemispheres": {}}
    for hemi in ("lh", "rh"):
        start = perf_counter()
        q_path = surf / f"{hemi}.qsphere.nofix"
        orig_path = surf / f"{hemi}.orig.nofix"
        input_xyz, faces = fsio.read_geometry(str(q_path))
        original_xyz, original_faces = fsio.read_geometry(str(orig_path))
        if not np.array_equal(faces, original_faces):
            raise ValueError(f"{hemi}: qsphere and orig face order differ")
        before = topology_counts(faces, len(input_xyz))
        spherical, iterations = center_sphere(project_and_smooth_sphere(input_xyz, faces))
        reference_sphere = diagnostics / f"{hemi}.smooth5.sphere.native"
        expected_xyz, expected_faces = fsio.read_geometry(str(reference_sphere))
        if not np.array_equal(faces, expected_faces):
            raise ValueError(f"{hemi}: diagnostic sphere face order differs")
        vertex_distance = np.linalg.norm(spherical.astype(np.float64)
                                         - expected_xyz.astype(np.float64), axis=1)
        sphere_seconds = perf_counter() - start

        start = perf_counter()
        components = defect_component_labels(spherical, faces)
        maps = {"defect_labels": components,
                "defect_borders": defect_border_labels(components, faces),
                "defect_chull": defect_hull_labels(components, faces),
                "defect_status": defect_retention_status(spherical, original_xyz,
                                                          faces, components)}
        map_results = {}
        for name, actual in maps.items():
            reference = diagnostics / subject.name / "surf" / f"{hemi}.{name}"
            expected = fsio.read_morph_data(str(reference))
            map_results[name] = {"different_vertices": int(np.count_nonzero(actual != expected)),
                                 "reference_sha256": _sha256(reference)}
        translations = {}
        for name, actual in zip(("vtrans", "ftrans"),
                                genetic_base_translation(components, faces)):
            reference = diagnostics / f"{hemi}.{subject.name}.{name}.log"
            pairs = re.findall(r"\s*(\d+) -->\s*(-?\d+)", reference.read_text())
            if len(pairs) != len(actual) or any(int(index) != i for i, (index, _) in enumerate(pairs)):
                raise ValueError(f"{hemi}: malformed native {name} mapping")
            expected = np.fromiter((int(value) for _, value in pairs), np.int32,
                                   count=len(actual))
            translations[name] = {
                "different_indices": int(np.count_nonzero(actual != expected)),
                "reference_sha256": _sha256(reference),
            }
        candidate_total, candidate_discarded = genetic_candidate_base_prune(
            spherical, faces, components, maps["defect_status"], 0)
        first_defect_log = diagnostics / f"{hemi}.first_defect_native.log"
        edge_match = re.search(r"(\d+) of (\d+) overlapping edges discarded",
                               first_defect_log.read_text())
        if edge_match is None:
            raise ValueError(f"{hemi}: native first-defect candidate count missing")
        native_discarded, native_remaining = map(int, edge_match.groups())
        candidate_result = {
            "total": candidate_total,
            "discarded": candidate_discarded,
            "remaining": candidate_total - candidate_discarded,
            "native_total": native_discarded + native_remaining,
            "native_discarded": native_discarded,
            "native_remaining": native_remaining,
            "native_log_sha256": _sha256(first_defect_log),
        }
        defects_seconds = perf_counter() - start

        final_path = surf / f"{hemi}.orig.premesh"
        final_xyz, final_faces = fsio.read_geometry(str(final_path))
        after = topology_counts(final_faces, len(final_xyz))
        report["hemispheres"][hemi] = {
            "input_sha256": {"qsphere.nofix": _sha256(q_path),
                             "orig.nofix": _sha256(orig_path)},
            "native_diagnostic_sphere_sha256": _sha256(reference_sphere),
            "native_final_surface_sha256": _sha256(final_path),
            "input_topology": vars(before),
            "native_final_topology": vars(after),
            "sphere_iterations_python": iterations,
            "sphere_max_vertex_distance_mm": float(vertex_distance.max()),
            "sphere_vertices_over_0.0001_mm": int(np.count_nonzero(vertex_distance > 1e-4)),
            "defects": int(components.max()),
            "maps": map_results,
            "pre_patch_translation": translations,
            "first_defect_candidate_edges": candidate_result,
            "python_sphere_seconds": sphere_seconds,
            "python_defects_seconds": defects_seconds,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.subject, args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for hemi, values in result["hemispheres"].items():
        print(hemi, values["input_topology"], values["native_final_topology"],
              values["defects"], values["sphere_max_vertex_distance_mm"],
              {name: row["different_vertices"] for name, row in values["maps"].items()},
              {name: row["different_indices"] for name, row in values["pre_patch_translation"].items()},
              values["first_defect_candidate_edges"])
    if any(
        values["sphere_vertices_over_0.0001_mm"]
        or any(row["different_vertices"] for row in values["maps"].values())
        or any(row["different_indices"] for row in values["pre_patch_translation"].values())
        or values["first_defect_candidate_edges"]["total"] != values["first_defect_candidate_edges"]["native_total"]
        or values["first_defect_candidate_edges"]["discarded"] != values["first_defect_candidate_edges"]["native_discarded"]
        for values in result["hemispheres"].values()
    ):
        raise SystemExit("topology preflight mismatch; inspect output report")


if __name__ == "__main__":
    main()
