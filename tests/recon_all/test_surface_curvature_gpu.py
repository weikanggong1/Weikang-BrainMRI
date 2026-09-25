import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.surface_curvature_gpu import curvature_map


def test_planar_surface_has_zero_curvature(tmp_path):
    vertices = np.array([(x, y, 0) for y in range(4) for x in range(4)],
                        dtype=np.float32)
    faces = []
    for y in range(3):
        for x in range(3):
            first = 4 * y + x
            faces.extend(((first, first + 1, first + 4),
                          (first + 1, first + 5, first + 4)))
    surface, output = tmp_path / "white", tmp_path / "curv"
    fs.write_geometry(str(surface), vertices, np.asarray(faces, dtype=np.int32))
    report = curvature_map(surface, output, device="cpu")
    np.testing.assert_allclose(fs.read_morph_data(str(output)), 0, atol=1e-7)
    assert report["vertices"] == 16
