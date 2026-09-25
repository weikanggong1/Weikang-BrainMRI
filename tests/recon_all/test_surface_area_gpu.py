from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fnit.recon_all.surface_area_gpu import area_map, mid_area_map, vertex_area


def test_shared_triangle_area_and_morph_output(tmp_path: Path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    surface, output = tmp_path / "lh.white", tmp_path / "lh.area"
    fsio.write_geometry(str(surface), vertices, faces)

    area_map(surface, output, device="cpu")
    expected = np.array([1 / 3, 1 / 6, 1 / 3, 1 / 6], dtype=np.float32)
    np.testing.assert_allclose(fsio.read_morph_data(str(output)), expected,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(vertex_area(vertices, faces, device="cpu"), expected,
                               rtol=0, atol=1e-7)


def test_mid_area_matches_sequential_float32_add_and_divide(tmp_path: Path):
    white = np.array([1.0, 2.0, 3.25], dtype=np.float32)
    pial = np.array([2.0, 4.5, 4.75], dtype=np.float32)
    first, second, result = (tmp_path / name for name in ("white.area", "pial.area", "mid.area"))
    fsio.write_morph_data(str(first), white)
    fsio.write_morph_data(str(second), pial)
    mid_area_map(first, second, result, device="cpu")
    np.testing.assert_array_equal(fsio.read_morph_data(str(result)), (white + pial) / 2)
