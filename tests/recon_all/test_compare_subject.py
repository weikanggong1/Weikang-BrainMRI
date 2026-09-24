"""End-to-end artifact checks must reject local errors hidden by averages."""

import json
from pathlib import Path
import shutil

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest

from freesurfer_torch.recon_all.compare_subject import ANNOTATIONS, MAP_SURFACES, SURFACES, compare_subject, main


def _subject(path):
    for directory in ("surf", "label", "stats", "mri"):
        (path / directory).mkdir(parents=True)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    for hemi in ("lh", "rh"):
        for surface in SURFACES:
            fs.write_geometry(path / "surf" / f"{hemi}.{surface}", vertices, faces)
        for metric in MAP_SURFACES:
            fs.write_morph_data(path / "surf" / f"{hemi}.{metric}", np.array([1, 2, 3, 0], dtype=np.float32))
        for annotation in ANNOTATIONS:
            fs.write_annot(path / "label" / f"{hemi}.{annotation}.annot", np.array([0, 0, 1, 1]),
                           np.array([[10, 20, 30, 0], [40, 50, 60, 0]]), ["region1", "region2"])
        (path / "label" / f"{hemi}.cortex.label").write_text("#!ascii label\n3\n0 0 0 0 0\n1 0 0 0 0\n2 0 0 0 0\n")
        for suffix in ("aparc.stats", "aparc.pial.stats", "aparc.DKTatlas.stats", "aparc.a2009s.stats"):
            (path / "stats" / f"{hemi}.{suffix}").write_text(
                "# cmdline mris_anatomical_stats -no-th3 -a aparc.annot subject lh white\n"
                "# Measure Cortex, CortexVol, Cortex volume, 42, mm^3\n"
                "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd\n"
                "region1 2 3 4 2.500 0.300 0.1 0.2 1 1\n"
                "region2 2 6 8 3.000 0.400 0.2 0.3 2 2\n")
    data = np.zeros((3, 3, 3), dtype=np.int16)
    data[0] = 2
    data[1] = 3
    for name in ("aseg.mgz", "aparc+aseg.mgz"):
        nib.save(nib.MGHImage(data, np.eye(4)), path / "mri" / name)
    (path / "stats" / "aseg.stats").write_text(
        "# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange\n"
        "1 2 9 9 Left-WM 100 3 90 110 20\n2 3 9 9 Left-Cortex 90 2 85 95 10\n")
    (path / "stats" / "synthseg.vol.csv").write_text(
        "subject,total intracranial,csf,left lateral ventricle\n"
        "orig.mgz,1000000,1000,500\n")


@pytest.fixture
def subjects(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _subject(reference)
    shutil.copytree(reference, candidate)
    return reference, candidate


def test_complete_identical_subjects_pass_and_json_roundtrips(subjects):
    report = compare_subject(*subjects)
    assert report["passed"]
    assert not report["failed_checks"]
    assert len(report["checks"]) == 52
    assert json.loads(json.dumps(report, allow_nan=False))["passed"]
    assert report["checks"]["surf/lh.white"]["reference_topology"]["euler"] == 2


def test_single_vertex_violation_is_not_hidden_by_mean(subjects):
    path = subjects[1] / "surf/lh.thickness"
    values = fs.read_morph_data(path)
    values[2] += 0.02
    fs.write_morph_data(path, values)
    report = compare_subject(*subjects, tolerances={"thickness": {"atol": 0.01}})
    check = report["checks"]["surf/lh.thickness"]
    assert not report["passed"]
    assert check["mae"] < 0.01
    assert check["outlier_ids"] == [2]
    assert check["max_error_id"] == 2
    assert compare_subject(*subjects, tolerances={"thickness": {"atol": 0.03}})["passed"]


def test_surface_displacement_reports_vertex_and_does_not_block_valid_indexing(subjects):
    path = subjects[1] / "surf/lh.pial"
    vertices, faces = fs.read_geometry(path)
    vertices[1, 0] += 0.2
    fs.write_geometry(path, vertices, faces)
    report = compare_subject(*subjects, tolerances={"coordinates": {"atol": 0.01}})
    check = report["checks"]["surf/lh.pial"]
    assert not report["passed"]
    assert check["displacement_mm"]["outlier_ids"] == [1]
    assert check["vertex_correspondence_compatible"]
    assert report["checks"]["surf/lh.thickness"]["status"] == "passed"


def test_reindexing_mesh_blocks_native_vertex_metrics(subjects):
    path = subjects[1] / "surf/lh.white"
    vertices, faces = fs.read_geometry(path)
    permutation = np.array([1, 0, 2, 3])
    fs.write_geometry(path, vertices[permutation], permutation[faces])
    report = compare_subject(*subjects)
    check = report["checks"]["surf/lh.white"]
    assert check["candidate_topology"]["euler"] == 2
    assert not check["ordered_faces_equal"]
    assert report["checks"]["surf/lh.thickness"]["status"] == "blocked"
    assert report["checks"]["label/lh.aparc.annot"]["status"] == "blocked"


def test_nonfinite_and_wrong_length_maps_fail(subjects):
    fs.write_morph_data(subjects[1] / "surf/lh.volume", np.array([1, np.nan, 3, 0], dtype=np.float32))
    fs.write_morph_data(subjects[1] / "surf/rh.area", np.ones(3, dtype=np.float32))
    report = compare_subject(*subjects)
    assert report["checks"]["surf/lh.volume"]["nonfinite_count"] == 1
    assert report["checks"]["surf/lh.volume"]["outlier_ids"] == [1]
    assert report["checks"]["surf/rh.area"]["status"] == "error"
    json.dumps(report, allow_nan=False)


def test_missing_in_both_is_not_treated_as_agreement(subjects):
    for path in subjects:
        (path / "surf/rh.white.K").unlink()
    report = compare_subject(*subjects)
    assert not report["passed"]
    assert report["checks"]["surf/rh.white.K"] == {"status": "missing", "missing_in": ["reference", "candidate"]}


def test_corrupt_volume_is_reported_without_abandoning_other_checks(subjects):
    (subjects[1] / "mri/aseg.mgz").write_bytes(b"")
    report = compare_subject(*subjects)
    assert not report["passed"]
    assert report["checks"]["mri/aseg.mgz"]["status"] == "error"
    assert report["checks"]["mri/aparc+aseg.mgz"]["status"] == "passed"


def test_annotation_labels_and_color_names_are_checked(subjects):
    path = subjects[1] / "label/lh.aparc.annot"
    labels, colors, names = fs.read_annot(path)
    labels[0] = 1
    names[0] = b"renamed"
    fs.write_annot(path, labels, colors, names)
    report = compare_subject(*subjects)
    check = report["checks"]["label/lh.aparc.annot"]
    assert check["outlier_ids"] == [0]
    assert not check["color_table_equal"]
    assert not report["passed"]


def test_cortex_label_accepts_matching_duplicates_regardless_of_entry_order(subjects):
    for subject, ids in zip(subjects, ([0, 1, 1, 2], [1, 2, 1, 0])):
        (subject / "label/lh.cortex.label").write_text(
            "#!ascii label\n4\n" + "".join(f"{i} 0 0 0 0\n" for i in ids))
    report = compare_subject(*subjects)
    assert report["passed"]
    check = report["checks"]["label/lh.cortex.label"]
    assert check["membership"]["status"] == "passed"
    assert check["reference"]["duplicate_entries"] == 1
    assert check["candidate"]["unique_vertices"] == 3


def test_cortex_label_duplicate_multiplicity_change_fails_with_same_membership(subjects):
    (subjects[0] / "label/lh.cortex.label").write_text(
        "#!ascii label\n4\n0 0 0 0 0\n1 0 0 0 0\n1 0 0 0 0\n2 0 0 0 0\n")
    report = compare_subject(*subjects)
    check = report["checks"]["label/lh.cortex.label"]
    assert not report["passed"]
    assert check["status"] == "failed"
    assert check["membership"]["status"] == "passed"
    assert check["outlier_ids"] == [1]


@pytest.mark.parametrize("invalid_id", [-1, 4])
def test_cortex_label_rejects_out_of_range_ids_even_in_self_comparison(subjects, invalid_id):
    (subjects[0] / "label/lh.cortex.label").write_text(
        f"#!ascii label\n3\n0 0 0 0 0\n1 0 0 0 0\n{invalid_id} 0 0 0 0\n")
    check = compare_subject(subjects[0], subjects[0])["checks"]["label/lh.cortex.label"]
    assert check["status"] == "error"
    assert "outside [0, nvertices)" in check["reason"]


def test_all_stats_columns_global_measures_and_rows_are_checked(subjects):
    path = subjects[1] / "stats/lh.aparc.stats"
    path.write_text(path.read_text().replace("42, mm^3", "43, mm^3").replace("2 3 4 2.500", "2 3 5 2.500"))
    path = subjects[1] / "stats/aseg.stats"
    path.write_text(path.read_text().replace("2 3 9 9 Left-Cortex 90 2 85 95 10\n", ""))
    report = compare_subject(*subjects)
    check = report["checks"]["stats/lh.aparc.stats"]
    assert check["rows"]["region1"]["GrayVol"]["max_abs_error"] == 1
    assert check["measures"]["Cortex.CortexVol"]["status"] == "failed"
    assert report["checks"]["stats/aseg.stats"]["missing_candidate_rows"] == ["Left-Cortex"]


def test_synthseg_soft_volumes_check_each_column_and_etiv(subjects):
    tolerance = json.loads(Path(__file__).with_name("tolerances_numeric.json").read_text())
    path = subjects[1] / "stats/synthseg.vol.csv"
    path.write_text("subject,total intracranial,csf,left lateral ventricle\n"
                    "orig.mgz,1000500,1009,509\n")
    check = compare_subject(*subjects, tolerances=tolerance)["checks"]["stats/synthseg.vol.csv"]
    assert check["status"] == "passed"
    assert check["eTIV_column"] == "total intracranial"
    assert check["columns"]["total intracranial"]["signed_error_mm3"] == 500
    assert check["columns"]["total intracranial"]["allowed_error_mm3"] == 1010
    assert check["columns"]["left lateral ventricle"]["allowed_error_mm3"] == 10.5
    path.write_text(path.read_text().replace("509\n", "511\n"))
    report = compare_subject(*subjects, tolerances=tolerance)
    check = report["checks"]["stats/synthseg.vol.csv"]
    assert check["status"] == "failed"
    assert check["columns"]["total intracranial"]["status"] == "passed"
    assert check["columns"]["left lateral ventricle"]["abs_error_mm3"] == 11
    assert check["columns"]["left lateral ventricle"]["status"] == "failed"


def test_synthseg_csv_missing_column_and_nonfinite_value_fail(subjects):
    path = subjects[1] / "stats/synthseg.vol.csv"
    path.write_text("subject,total intracranial,csf\norig.mgz,1000000,1000\n")
    check = compare_subject(*subjects)["checks"]["stats/synthseg.vol.csv"]
    assert check["status"] == "failed"
    assert check["missing_candidate_columns"] == ["left lateral ventricle"]
    path.write_text("subject,total intracranial,csf,left lateral ventricle\n"
                    "orig.mgz,1000000,nan,500\n")
    check = compare_subject(*subjects)["checks"]["stats/synthseg.vol.csv"]
    assert check["status"] == "error"
    assert "Nonfinite" in check["reason"]


def test_synthseg_stiv_measure_uses_same_bound_as_soft_total(subjects):
    tolerances = json.loads(Path(__file__).with_name("tolerances_numeric.json").read_text())
    measure = "# Measure SegmentedTotalIntraCranialVol, sTIV, Segmented Total Intracranial Volume, "
    for subject, value in zip(subjects, ("1000000", "1000500")):
        path = subject / "stats/aseg.stats"
        path.write_text(measure + value + ", mm^3\n" + path.read_text())
    report = compare_subject(*subjects, tolerances=tolerances)
    assert report["passed"]
    check = report["checks"]["stats/aseg.stats"]["measures"]["SegmentedTotalIntraCranialVol.sTIV"]
    assert check["tolerance"] == tolerances["stats.measure.SegmentedTotalIntraCranialVol.sTIV"]
    assert check["max_abs_error"] == 500
    path = subjects[1] / "stats/aseg.stats"
    path.write_text(path.read_text().replace(measure + "1000500", measure + "1001500"))
    assert not compare_subject(*subjects, tolerances=tolerances)["passed"]


def test_explicit_stats_volume_algorithm_difference_fails_even_if_numbers_match(subjects):
    path = subjects[1] / "stats/lh.aparc.stats"
    path.write_text(path.read_text().replace("-no-th3", "-th3"))
    check = compare_subject(*subjects)["checks"]["stats/lh.aparc.stats"]
    assert check["status"] == "failed"
    assert check["reference_volume_mode"] == "-no-th3"
    assert check["candidate_volume_mode"] == "-th3"


def test_dkt_and_destrieux_are_mandatory_and_compared(subjects):
    (subjects[1] / "stats/rh.aparc.DKTatlas.stats").unlink()
    path = subjects[1] / "stats/lh.aparc.a2009s.stats"
    path.write_text(path.read_text().replace("2 3 4 2.500", "2 3 4 2.750"))
    path = subjects[1] / "label/lh.aparc.DKTatlas.annot"
    labels, colors, names = fs.read_annot(path)
    labels[2] = 0
    fs.write_annot(path, labels, colors, names)
    report = compare_subject(*subjects)
    assert report["checks"]["stats/rh.aparc.DKTatlas.stats"]["status"] == "missing"
    assert report["checks"]["stats/lh.aparc.a2009s.stats"]["rows"]["region1"]["ThickAvg"]["status"] == "failed"
    assert report["checks"]["label/lh.aparc.DKTatlas.annot"]["outlier_ids"] == [2]


def test_provisional_profile_applies_vertex_bounds_without_skipping_files(subjects):
    tolerances = json.loads(Path(__file__).with_name("tolerances_numeric.json").read_text())
    path = subjects[1] / "surf/lh.area"
    values = fs.read_morph_data(path)
    values[0] += 0.001
    fs.write_morph_data(path, values)
    assert compare_subject(*subjects, tolerances=tolerances)["passed"]
    values[3] = 0.002  # Reference zero: the absolute bound, not rtol, controls this vertex.
    fs.write_morph_data(path, values)
    (subjects[1] / "label/rh.aparc.a2009s.annot").unlink()
    report = compare_subject(*subjects, tolerances=tolerances)
    assert report["checks"]["surf/lh.area"]["outlier_ids"] == [3]
    assert report["checks"]["label/rh.aparc.a2009s.annot"]["status"] == "missing"


def test_aseg_reports_label_and_voxel_without_resampling(subjects):
    path = subjects[1] / "mri/aseg.mgz"
    image = nib.load(path)
    data = np.asarray(image.dataobj).copy()
    data[0, 1, 2] = 3
    nib.save(nib.MGHImage(data, image.affine), path)
    check = compare_subject(*subjects)["checks"]["mri/aseg.mgz"]
    assert check["outlier_ids"] == [[0, 1, 2]]
    assert check["labels"]["2"]["candidate_count"] == 8
    affine = image.affine.copy()
    affine[0, 3] += 0.5
    nib.save(nib.MGHImage(data, affine), path)
    check = compare_subject(*subjects)["checks"]["mri/aseg.mgz"]
    assert check["status"] == "failed"
    assert "no resampling" in check["reason"]


def test_invalid_tolerances_and_cli_failure_code(subjects, tmp_path):
    with pytest.raises(ValueError, match="nonnegative"):
        compare_subject(*subjects, tolerances={"thickness": {"atol": -1}})
    with pytest.raises(ValueError, match="absolute displacement"):
        compare_subject(*subjects, tolerances={"coordinates": {"rtol": 0.1}})
    output = tmp_path / "report.json"
    assert main([str(p) for p in subjects] + ["--output", str(output)]) == 0
    (subjects[1] / "surf/rh.curv").unlink()
    assert main([str(p) for p in subjects] + ["--output", str(output)]) == 1
    assert not json.loads(output.read_text())["passed"]
