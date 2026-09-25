"""Python replay of the N4 recon-all wrapper after isolated bias correction.

The FreeSurfer 8.2 wrapper computes a five-decimal global mean ratio, applies
it to ``nu0.mgz``, and maps a Talairach-centered intensity histogram to uchar.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import gzip
from pathlib import Path

import nibabel as nib
import numpy as np


_UNKNOWN_TAG_WITH_NUL = b"\0\0\0)\0\0\0\0\0\0\0\x08UNKNOWN\0"
_UNKNOWN_TAG_NATIVE = b"\0\0\0)\0\0\0\0\0\0\0\x07UNKNOWN"


def normalize_n4_footer(path: str | Path) -> bool:
    """Match the FreeSurfer N4 writer's observed MGH XFORM string length.

    Returns whether the file changed. Other footer layouts are left intact.
    """
    path = Path(path)
    image = nib.load(str(path))
    if not isinstance(image, nib.MGHImage) or image.get_data_dtype() != np.dtype("uint8"):
        raise ValueError("expected an MGH/MGZ uchar volume")
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".mgz" else path.read_bytes()
    end = int(image.header.get_data_offset()) + int(np.prod(image.shape))
    footer = raw[end:]
    if footer.count(_UNKNOWN_TAG_WITH_NUL) != 1:
        return False
    raw = raw[:end] + footer.replace(_UNKNOWN_TAG_WITH_NUL, _UNKNOWN_TAG_NATIVE, 1)
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".mgz" else raw)
    return True


def _copy_original_footer(original_file: str | Path, output_file: str | Path,
                          shape: tuple[int, ...]) -> None:
    """Replay recon-all's final ``mri_add_xform_to_header`` metadata state."""
    source, output = Path(original_file), Path(output_file)
    original_raw = gzip.decompress(source.read_bytes()) if source.suffix == ".mgz" else source.read_bytes()
    output_raw = gzip.decompress(output.read_bytes()) if output.suffix == ".mgz" else output.read_bytes()
    end = 284 + int(np.prod(shape))
    raw = output_raw[:end] + original_raw[end:]
    output.write_bytes(gzip.compress(raw, mtime=0) if output.suffix == ".mgz" else raw)


def global_mean_scale(original: np.ndarray, corrected: np.ndarray) -> float:
    """Reproduce ``mri_segstats --avgwf`` (five decimals) and ``bc -l`` ratio."""
    if original.shape != corrected.shape or original.ndim != 3:
        raise ValueError("original and corrected volumes must have the same 3D shape")
    in_mean = Decimal(f"{np.mean(original, dtype=np.float64):.5f}")
    out_mean = Decimal(f"{np.mean(corrected, dtype=np.float64):.5f}")
    if not out_mean:
        raise ValueError("corrected volume has zero mean")
    return float(in_mean / out_mean)


def read_mni_xfm(path: str | Path) -> np.ndarray:
    """Read the linear 3x4 matrix used by ``mri_make_uchar``."""
    text = Path(path).read_text()
    body = text.partition("Linear_Transform =")[2].partition(";")[0]
    values = np.fromstring(body, sep=" ")
    if values.size != 12:
        raise ValueError("expected a linear 3x4 MNI transform")
    affine = np.eye(4)
    affine[:3] = values.reshape(3, 4)
    return affine


def make_uchar(values: np.ndarray, vox2ras: np.ndarray, tal_xfm: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Replay the 50 mm Talairach-ball histogram and 1%/90% uchar mapping."""
    if values.ndim != 3:
        raise ValueError("expected a 3D volume")
    tal_from_vox = tal_xfm @ vox2ras
    counts = np.zeros(256, dtype=np.int64)
    yy, zz = np.mgrid[:values.shape[1], :values.shape[2]]
    for xx in range(values.shape[0]):
        radius2 = sum(np.square(tal_from_vox[i, 0] * xx + tal_from_vox[i, 1] * yy
                                + tal_from_vox[i, 2] * zz + tal_from_vox[i, 3])
                      for i in range(3))
        selected = np.floor(np.clip(values[xx][radius2 < 50**2], 0, 255) + .5).astype(np.uint8)
        counts += np.bincount(selected, minlength=256)
    if not counts.any():
        raise ValueError("Talairach ball contains no voxels")

    maximum = int(np.flatnonzero(counts)[-1])
    bin_size = np.float32(maximum / 99)
    bins = np.arange(100, dtype=np.float32) * bin_size
    indices = np.floor(np.arange(256, dtype=np.float32) / bin_size + .5).astype(int)
    hist = np.bincount(np.clip(indices, 0, 99), weights=counts, minlength=100).astype(np.float32)
    hist[0] = 0
    total = np.sum(hist, dtype=np.float32)
    hist /= total
    hist[hist == 0] = np.float32(1 / (10 * total))
    cdf = np.cumsum(hist, dtype=np.float64)
    cdf /= cdf[-1]
    first, white = (int(np.argmin(np.abs(cdf - q))) for q in (.01, .90))
    low, high = float(bins[first]), float(bins[white])
    if low == high:
        raise ValueError("Talairach-ball histogram has no usable range")
    slope = (110 - .01 * 255) / (high - low)
    offset = 110 - slope * high
    mapped = np.floor(np.clip(values.astype(np.float64) * slope + offset, 0, 255) + .5)
    return mapped.astype(np.uint8), (first, white)


def make_nu(original_file: str | Path, n4_file: str | Path,
            tal_xfm_file: str | Path, output_file: str | Path) -> tuple[float, tuple[int, int]]:
    """Build ``nu.mgz`` from an already corrected ``nu0.mgz``."""
    original, corrected = nib.load(str(original_file)), nib.load(str(n4_file))
    if not isinstance(original, nib.MGHImage) or not isinstance(corrected, nib.MGHImage):
        raise ValueError("expected MGH/MGZ inputs")
    original_values = np.asarray(original.dataobj)
    corrected_values = np.asarray(corrected.dataobj)
    scale = global_mean_scale(original_values, corrected_values)
    scaled = corrected_values.astype(np.float32) * np.float32(scale)
    result, bins = make_uchar(scaled, original.affine, read_mni_xfm(tal_xfm_file))
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.MGHImage(result, original.affine, header), str(output_file))
    _copy_original_footer(original_file, output_file, original.shape)
    return scale, bins


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", required=True)
    parser.add_argument("--nu0", required=True)
    parser.add_argument("--tal", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    scale, bins = make_nu(args.orig, args.nu0, args.tal, args.out)
    print(f"scale={scale:.20f} histogram_bins={bins}")


if __name__ == "__main__":
    main()
