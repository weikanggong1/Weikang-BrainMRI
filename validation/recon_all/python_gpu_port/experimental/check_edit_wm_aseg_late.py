"""Check fixed fs_sub01 WM late-edit stages and the remaining core boundary."""

import argparse
import gzip
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.edit_wm_aseg_late_python import fix_subcortical_mass_ha


def _read(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def _compare(name: str, left: np.ndarray, right: np.ndarray) -> int:
    mismatch = int(np.count_nonzero(left != right))
    print(name, mismatch, "/", left.size)
    return mismatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_mri", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    s, r = args.subject_mri, args.reference
    core = _read(r / "core.mgz")
    aseg = _read(s / "aseg.presurf.mgz")
    scm = fix_subcortical_mass_ha(core, aseg)
    differences = [
        _compare("python SCM vs native SCM", scm, _read(r / "core_scm.mgz")),
        _compare("python late without fill vs native", _read(r / "core_opts_python.mgz"),
                 _read(r / "core_opts.mgz")),
        _compare("python late with fill vs native", _read(r / "full_late_python.mgz"),
                 _read(r / "full.mgz")),
        _compare("fresh native vs original", _read(r / "full.mgz"),
                 _read(s / "wm.asegedit.mgz")),
    ]
    print("core input vs original", int(np.count_nonzero(core != _read(s / "wm.seg.mgz"))))
    print("fill and interactions vs no-fill",
          int(np.count_nonzero(_read(r / "core_opts.mgz") != _read(r / "full.mgz"))))
    full = _read(r / "full_late_python.mgz")
    print("python full voxel sha256", hashlib.sha256(full.tobytes(order="F")).hexdigest())
    native_raw = gzip.decompress((r / "full.mgz").read_bytes())
    python_raw = gzip.decompress((r / "full_late_python.mgz").read_bytes())
    end = 284 + full.size
    print("header equal", python_raw[:284] == native_raw[:284],
          "voxel bytes equal", python_raw[284:end] == native_raw[284:end],
          "scan parameters equal", python_raw[end:end + 20] == native_raw[end:end + 20])
    if any(differences):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
