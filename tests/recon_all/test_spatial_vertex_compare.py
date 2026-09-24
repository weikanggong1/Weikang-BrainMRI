"""Spatial diagnostics tolerate remeshing but never claim native index identity."""

import json

import nibabel.freesurfer as fs
import numpy as np
import pytest

from freesurfer_torch.recon_all.compare_subject import MAP_SURFACES
from freesurfer_torch.recon_all.spatial_vertex_compare import compare_spatial_vertices, main


VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)


def _write_subject(path, vertices, faces, values):
    (path / "surf").mkdir(parents=True, exist_ok=True)
    for hemi in ("lh", "rh"):
        fs.write_geometry(path / "surf" / f"{hemi}.white", vertices, faces)
        fs.write_geometry(path / "surf" / f"{hemi}.pial", vertices + [0, 0, 0.3], faces)
        for name in MAP_SURFACES:
            fs.write_morph_data(path / "surf" / f"{hemi}.{name}", values)


def test_reindexed_mesh_matches_spatially_without_claiming_native_identity(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    permutation = np.array([2, 0, 3, 1])
    _write_subject(reference, VERTICES, FACES, values)
    _write_subject(candidate, VERTICES[permutation], np.argsort(permutation)[FACES], values[permutation])
    report = compare_spatial_vertices(reference, candidate, max_distance_mm=0.01)
    assert report["spatial_thresholds_met"]
    assert report["native_index_consistency_proven"] is False
    hemi = report["hemispheres"]["lh"]
    assert hemi["shape"]["white"]["reference_to_candidate"]["reciprocal_fraction"] == 1
    assert hemi["shape"]["pial_under_white_match"]["reference_to_candidate"]["distance_mm_max"] == 0
    assert hemi["metrics"]["thickness"]["reference_to_candidate"]["mae"] == 0
    assert json.loads(json.dumps(report, allow_nan=False))["spatial_thresholds_met"]


def test_different_vertex_count_and_unmatched_distant_vertices(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_subject(reference, VERTICES, FACES, np.ones(4, dtype=np.float32))
    extra = np.array([[0.001, 0.001, 0]], dtype=np.float32)
    vertices = np.vstack([VERTICES, extra])
    faces = np.vstack([FACES[1:], [[0, 4, 1], [4, 2, 1], [0, 2, 4]]])
    _write_subject(candidate, vertices, faces, np.ones(5, dtype=np.float32))
    report = compare_spatial_vertices(reference, candidate, max_distance_mm=0.01)
    assert report["spatial_thresholds_met"]
    hemi = report["hemispheres"]["lh"]
    assert hemi["shape"]["white"]["candidate_to_reference"]["source_vertices"] == 5
    assert hemi["metrics"]["area"]["candidate_to_reference"]["coverage_fraction"] == 1
    vertices[4] += [10, 0, 0]
    _write_subject(candidate, vertices, faces, np.ones(5, dtype=np.float32))
    report = compare_spatial_vertices(reference, candidate, max_distance_mm=0.01)
    hemi = report["hemispheres"]["lh"]
    assert not report["spatial_thresholds_met"]
    assert hemi["shape"]["white"]["candidate_to_reference"]["coverage_fraction"] == 0.8
    assert hemi["metrics"]["area"]["candidate_to_reference"]["matched_count"] == 4


def test_pial_distance_blocks_metric_pairs_and_outlier_fraction_is_directional(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    values = np.ones(4, dtype=np.float32)
    _write_subject(reference, VERTICES, FACES, values)
    _write_subject(candidate, VERTICES, FACES, values)
    path = candidate / "surf/lh.thickness"
    values[0] += 0.2
    fs.write_morph_data(path, values)
    report = compare_spatial_vertices(reference, candidate, max_distance_mm=0.1,
                                      tolerances={"thickness": {"atol": 0.1}})
    check = report["hemispheres"]["lh"]["metrics"]["thickness"]["reference_to_candidate"]
    assert check["mae"] == pytest.approx(0.05)
    assert check["outlier_fraction"] == 0.25
    assert check["p99_abs_error"] <= check["max_abs_error"]
    for hemi in ("lh", "rh"):
        fs.write_geometry(candidate / "surf" / f"{hemi}.pial", VERTICES + [0, 0, 3], FACES)
    report = compare_spatial_vertices(reference, candidate, max_distance_mm=0.1)
    hemi = report["hemispheres"]["lh"]
    assert hemi["shape"]["pial_under_white_match"]["reference_to_candidate"]["joint_coverage_fraction"] == 0
    assert hemi["metrics"]["thickness"]["reference_to_candidate"]["mae"] is None
    assert hemi["metrics"]["thickness"]["reference_to_candidate"]["coverage_fraction"] == 0


def test_missing_map_is_reported_and_cli_writes_json(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_subject(reference, VERTICES, FACES, np.ones(4, dtype=np.float32))
    _write_subject(candidate, VERTICES, FACES, np.ones(4, dtype=np.float32))
    (candidate / "surf/rh.curv").unlink()
    output = tmp_path / "report.json"
    assert main([str(reference), str(candidate), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["hemispheres"]["rh"]["metrics"]["curv"]["status"] == "error"
    with pytest.raises(ValueError, match="max_distance_mm"):
        compare_spatial_vertices(reference, candidate, max_distance_mm=0)
