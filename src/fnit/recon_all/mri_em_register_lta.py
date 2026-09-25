"""Write the voxel-to-voxel LTA produced by the fixed T1 GCA alignment."""

from pathlib import Path

import nibabel as nib
import numpy as np

from .mri_em_register import GCA


def _volume_info(filename: str, shape: tuple[int, int, int],
                 spacing: np.ndarray, directions: np.ndarray,
                 center: np.ndarray) -> list[str]:
    lines = [
        "valid = 1  # volume info valid",
        f"filename = {filename}",
        "volume = " + " ".join(str(int(value)) for value in shape),
        "voxelsize = " + " ".join(f"{float(value):.15e}" for value in spacing),
    ]
    for name, row in zip(("xras", "yras", "zras"), directions):
        lines.append(name + "   = " + " ".join(f"{float(value):.15e}" for value in row))
    lines.append("cras   = " + " ".join(f"{float(value):.15e}" for value in center))
    return lines


def write_voxel_lta(output_path: str | Path, matrix: np.ndarray,
                    nu_path: str | Path, atlas_path: str | Path,
                    atlas: GCA, masked_input: np.ndarray) -> None:
    """Serialize the transform and source/atlas volume geometry without FreeSurfer."""

    output = Path(output_path)
    source_path = Path(nu_path)
    source = nib.load(str(source_path))
    matrix = np.asarray(matrix, np.float32)
    if matrix.shape != (4, 4) or masked_input.shape != source.shape:
        raise ValueError("LTA transform and input volume shape disagree")
    foreground = np.where(masked_input > 70)
    mean = [int(axis.min()) + (int(axis.max()) - int(axis.min()) + 1) // 2
            for axis in foreground]
    lines = [
        f"# transform file {output.name}",
        "",
        "type      = 0 # LINEAR_VOX_TO_VOX",
        "nxforms   = 1",
        "mean      = " + " ".join(f"{value:.4f}" for value in mean),
        "sigma     = 10000.0000",
        "1 4 4",
    ]
    lines.extend(" ".join(f"{float(value):.15e}" for value in row) for row in matrix)
    lines.append("src volume info")
    lines.extend(_volume_info(source_path.name, source.shape,
                              np.asarray(source.header["delta"]),
                              np.asarray(source.header["Mdc"]),
                              np.asarray(source.header["Pxyz_c"])))
    lines.append("dst volume info")
    lines.extend(_volume_info(str(Path(atlas_path).resolve()), atlas.volume_shape,
                              np.asarray(atlas.voxel_sizes),
                              np.asarray(atlas.direction_cosines).reshape(3, 3),
                              np.asarray(atlas.center_ras)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
