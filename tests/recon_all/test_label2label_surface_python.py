from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all import label2label_surface_python as surface_label


def test_forward_order_then_reverse_target_order(tmp_path: Path, monkeypatch) -> None:
    source_sphere = tmp_path / "source.sphere.reg"
    target_sphere = tmp_path / "target.sphere.reg"
    source = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], np.float32)
    target = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [20, 0, 0]], np.float32)
    monkeypatch.setattr(surface_label, "_scaled_sphere",
                        lambda path: source if Path(path) == source_sphere else target)
    white = tmp_path / "target.white"
    fsio.write_geometry(str(white), target, np.array([[0, 1, 2], [1, 2, 3]], np.int32))
    source_label = tmp_path / "source.label"
    source_label.write_text("#!ascii label\n2\n2  0  0  0 0.4\n0  0  0  0 0.7\n")

    mapper = surface_label.SurfaceLabelMapper(source_sphere, target_sphere,
                                             white, "test_subject")
    output = tmp_path / "target.label"
    ids = mapper.map_label(source_label, output)

    assert ids.tolist() == [3, 0, 1]
    assert output.read_text().splitlines() == [
        "#!ascii label  , from subject test_subject vox2ras=TkReg",
        "3",
        "3  20.000  0.000  0.000 0.4000000060",
        "0  0.000  0.000  0.000 0.6999999881",
        "1  1.000  0.000  0.000 0.6999999881",
    ]
