"""FreeSurfer 8.2 estimated intracranial volume from a Talairach XFM."""

from pathlib import Path

import numpy as np


def estimate_tiv(xfm: str | Path, scale_factor: float = 1948.106) -> float:
    """Replay ``MRIestimateTIV`` including its float32 matrix determinant."""
    lines = Path(xfm).read_text().splitlines()
    if not lines or not lines[0].startswith("MNI Transform File"):
        raise ValueError("Expected an MNI transform file")
    index = next((i for i, line in enumerate(lines)
                  if line.strip() == "Linear_Transform ="), None)
    if index is None:
        raise ValueError("Missing Linear_Transform matrix")
    try:
        matrix = np.array([[float(value) for value in line.rstrip(" ;").split()]
                           for line in lines[index + 1:index + 4]], dtype=np.float32)
    except ValueError as exc:
        raise ValueError("Invalid Linear_Transform matrix") from exc
    if matrix.shape != (3, 4):
        raise ValueError("Expected three rows of four transform values")
    (a, b, c), (d, e, f), (g, h, i) = matrix[:, :3]
    # VNL's fixed-size determinant retains the fused product differences in
    # float32. Rounding each individual product changes the last displayed
    # six decimals of eTIV for the reference transform.
    minor1 = np.float32(float(e) * float(i) - float(f) * float(h))
    minor2 = np.float32(float(d) * float(i) - float(f) * float(g))
    minor3 = np.float32(float(d) * float(h) - float(e) * float(g))
    determinant = np.float32(np.float32(np.float32(a * minor1) -
                                       np.float32(b * minor2)) +
                             np.float32(c * minor3))
    return float(scale_factor * 1000 / float(determinant))
