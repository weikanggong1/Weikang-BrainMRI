import nibabel.freesurfer as fs
import numpy as np
import pytest
import torch

from fnit.recon_all import surface_thickness_gpu as stage
from fnit.recon_all.surface_thickness_gpu import thickness_map


def test_parallel_surfaces_have_unit_thickness(tmp_path):
    white = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    pial = white.copy()
    pial[:, 2] = 1
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    white_file, pial_file, output = (tmp_path / name for name in
                                     ("white", "pial", "thickness"))
    fs.write_geometry(str(white_file), white, faces)
    fs.write_geometry(str(pial_file), pial, faces)
    report = thickness_map(white_file, pial_file, output, device="cpu")
    np.testing.assert_allclose(fs.read_morph_data(str(output)), 1, atol=1e-6)
    assert report["vertices"] == 4


def test_rejects_different_white_pial_topology(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    white_file, pial_file = tmp_path / "white", tmp_path / "pial"
    fs.write_geometry(str(white_file), vertices, np.array([[0, 1, 2]], dtype=np.int32))
    fs.write_geometry(str(pial_file), vertices, np.array([[0, 2, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match="identical topology"):
        thickness_map(white_file, pial_file, tmp_path / "thickness", device="cpu")


def test_expands_search_when_256_nearest_are_rejected(tmp_path, monkeypatch):
    white = np.zeros((259, 3), dtype=np.float32)
    white[1:, 0] = np.arange(1001, 1259)
    pial = np.zeros_like(white)
    pial[0, 2] = 4
    pial[1:258, 2] = -.5
    pial[258, 2] = 1
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    white_file, pial_file, output = (tmp_path / name for name in
                                     ("white", "pial", "thickness"))
    fs.write_geometry(str(white_file), white, faces)
    fs.write_geometry(str(pial_file), pial, faces)
    monkeypatch.setattr(stage, "_normals", lambda vertices, _: torch.tensor(
        [0., 0., 1.], device=vertices.device).expand_as(vertices))
    monkeypatch.setattr(stage, "_adjacency", lambda _, n: [list(range(n)) for _ in range(n)])
    report = thickness_map(white_file, pial_file, output, device="cpu")
    assert fs.read_morph_data(str(output))[0] == pytest.approx(2.5)
    assert report["expanded_searches"] >= 1
