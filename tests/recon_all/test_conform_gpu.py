import gzip
import struct

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all.conform_gpu import (
    _sample_trilinear, _uchar_from_float, add_xform_to_header,
)


def test_trilinear_zero_pads_each_neighbor_and_rounds_half_up():
    volume = torch.zeros((2, 2, 2), dtype=torch.uint8)
    volume[0, 0, 0] = 4
    points = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0],
                           [-1.0, 0.0, 0.0]], dtype=torch.float32)
    assert _sample_trilinear(volume, points).tolist() == [4, 2, 0]


def test_histogram_scaling_clips_high_tail():
    values = np.array([0] * 8 + [10] * 1000 + [1000], dtype=np.float32)
    result = _uchar_from_float(torch.from_numpy(values)).numpy()
    assert result[0] == 0
    assert result[-1] == 255
    assert result[8] > 0


def test_add_xform_preserves_image_and_writes_freesurfer_tag(tmp_path):
    source = tmp_path / 'conform.mgz'
    target = tmp_path / 'orig.mgz'
    nib.save(nib.MGHImage(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(source))
    before = gzip.decompress(source.read_bytes())
    name = '/subject/mri/transforms/talairach.xfm'
    add_xform_to_header(source, target, name)
    after = gzip.decompress(target.read_bytes())
    assert after[:len(before)] == before
    assert after[len(before):] == struct.pack('>iq', 31, len(name) + 1) + name.encode() + b'\0'
