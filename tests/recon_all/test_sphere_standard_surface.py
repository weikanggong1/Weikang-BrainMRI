"""Final conventional sphere file preserves ordered geometry and volume info."""

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.sphere_standard_python import write_standard_sphere_surface


def test_standard_sphere_writer_preserves_volume_geometry_bytes(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], np.int32)
    volume_info = {
        "head": np.array([2, 0, 20], np.int32), "valid": "1  # volume info valid",
        "filename": "wm.mgz", "volume": np.array([256, 256, 256]),
        "voxelsize": np.ones(3), "xras": np.array([1, 0, 0]),
        "yras": np.array([0, 1, 0]), "zras": np.array([0, 0, 1]),
        "cras": np.array([0.1995697021484375, 20.30433654785156, -41.288330078125]),
    }
    inflated, output = tmp_path / "inflated", tmp_path / "sphere"
    fsio.write_geometry(str(inflated), vertices, faces, volume_info=volume_info)
    predicted = vertices + np.float32(0.25)
    write_standard_sphere_surface(output, predicted, faces, inflated)

    written_xyz, written_faces, written_info = fsio.read_geometry(
        str(output), read_metadata=True)
    _, _, source_info = fsio.read_geometry(str(inflated), read_metadata=True)
    assert np.array_equal(written_xyz, predicted)
    assert np.array_equal(written_faces, faces)
    assert all(np.array_equal(np.asarray(written_info[key]), np.asarray(value))
               for key, value in source_info.items())
    for path in (inflated, output):
        data = path.read_bytes()
        begin = data.index(b"valid =") - 12
        end = data.index(b"\n", data.index(b"cras   =")) + 1
        if path == inflated:
            expected = data[begin:end]
        else:
            assert data[begin:end] == expected
