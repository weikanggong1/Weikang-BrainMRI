import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.label2annot_python import write_label_annotation


def test_maxstat_tie_uses_later_label_and_unhit_uses_unknown(tmp_path):
    surface = tmp_path / "lh.orig"
    fsio.write_geometry(str(surface), np.zeros((4, 3), np.float32),
                        np.array([[0, 1, 2]], np.int32))
    table = tmp_path / "colors.txt"
    table.write_text("0 unknown 25 5 25 0\n1 A 1 2 3 0\n2 B 4 5 6 0\n")
    first, second = tmp_path / "lh.A.label", tmp_path / "lh.B.label"
    first.write_text("#!ascii label\n2\n0 0 0 0 0.5\n1 0 0 0 0.5\n")
    second.write_text("#!ascii label\n2\n1 0 0 0 0.5\n2 0 0 0 -1\n")
    output = tmp_path / "lh.test.annot"
    write_label_annotation(surface, table, [first, second], output)
    labels, _, _ = fsio.read_annot(str(output), orig_ids=True)
    assert labels.tolist() == [1 + (2 << 8) + (3 << 16),
                               4 + (5 << 8) + (6 << 16),
                               25 + (5 << 8) + (25 << 16),
                               25 + (5 << 8) + (25 << 16)]
