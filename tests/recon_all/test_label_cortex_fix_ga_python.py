from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all import label_cortex_fix_ga_python as fix_ga


def test_final_label_concatenates_ga_without_deduplicating(tmp_path: Path,
                                                          monkeypatch) -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    surface = tmp_path / "lh.white.preaparc"
    fsio.write_geometry(str(surface), vertices, np.array([[0, 1, 2]], np.int32))
    image = nib.MGHImage(np.zeros((4, 4, 4), np.int16), np.eye(4))
    aseg, entowm = tmp_path / "aseg.mgz", tmp_path / "entowm.mgz"
    nib.save(image, str(aseg))
    nib.save(image, str(entowm))
    monkeypatch.setattr(fix_ga, "cortex_label_mask",
                        lambda *args: np.array([True, False, False]))
    monkeypatch.setattr(fix_ga, "gyrus_ambiens_mask",
                        lambda *args: np.array([True, False, True]))

    output = tmp_path / "lh.cortex.label"
    base, ga = fix_ga.label_cortex_fix_ga(surface, aseg, entowm, "lh", output)

    assert base.tolist() == [0]
    assert ga.tolist() == [0, 2]
    assert output.read_text().splitlines() == [
        "#!ascii label , from subject vox2ras=TkReg",
        "3",
        "0  0.000  0.000  0.000 0.0000000000",
        "0  0.000  0.000  0.000 0.0000000000",
        "2  0.000  1.000  0.000 0.0000000000",
    ]
