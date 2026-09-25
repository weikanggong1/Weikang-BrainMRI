"""Regression checks for the native-free GCSA annotation boundary."""

from dataclasses import replace

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.gcsa_aseg import relabel_with_aseg
from fnit.recon_all.gcsa_initial import InitialAtlas
from fnit.recon_all.gcsa_label_python import write_annotation


def _atlas() -> InitialAtlas:
    return InitialAtlas(
        classifier_nodes=[((111, 10, 0.0, 1.0),)],
        prior_nodes=[((111, 1.0),)],
        color_table={0: ("unknown", 0, 0, 0, 0),
                     1: ("Medial_wall", 10, 20, 30, 0)},
        source_name="test.ctab", average_variance=1.0,
        minimum_determinant=0.01, singular_count=0, regularized_count=0)


def test_cc_voxel_uses_medial_wall_when_atlas_has_no_callosum():
    atlas = _atlas()
    result = relabel_with_aseg(
        np.array([111], np.int32), atlas, np.array([0]), np.array([0]),
        np.array([0.0]), np.zeros((1, 3), np.float32),
        np.array([[[251]]], np.int16), np.eye(4))
    np.testing.assert_array_equal(result, [10 | (20 << 8) | (30 << 16)])

    with_callosum = replace(atlas, color_table={**atlas.color_table,
                                               2: ("corpuscallosum", 1, 2, 3, 0)})
    result = relabel_with_aseg(
        np.array([111], np.int32), with_callosum, np.array([0]), np.array([0]),
        np.array([0.0]), np.zeros((1, 3), np.float32),
        np.array([[[251]]], np.int16), np.eye(4))
    np.testing.assert_array_equal(result, [1 | (2 << 8) | (3 << 16)])


def test_native_annotation_writer_roundtrip(tmp_path):
    atlas = _atlas()
    expected = np.array([0, 10 | (20 << 8) | (30 << 16)], np.int32)
    output = tmp_path / "lh.aparc.annot"
    write_annotation(output, expected, atlas)
    actual, _, names = fsio.read_annot(str(output), orig_ids=True)
    np.testing.assert_array_equal(actual, expected)
    assert names == [b"unknown", b"Medial_wall"]
