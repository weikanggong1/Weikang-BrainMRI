"""The native uchar conversion changes exactly three frozen T1 voxels."""

import numpy as np

from fnit.recon_all.ants_denoise_python import _to_uchar


def test_freesurfer_nint_half_up_and_clip():
    values = np.array([0.49, 0.50, 1.49, 1.50, 254.50, -0.2, 300.0],
                      dtype=np.float32)
    np.testing.assert_array_equal(_to_uchar(values), [0, 1, 1, 2, 255, 0, 255])
