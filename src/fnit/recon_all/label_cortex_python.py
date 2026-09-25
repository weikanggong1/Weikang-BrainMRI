"""Python translation of the fixed ``mri_label2label --label-cortex`` call.

Implements the FreeSurfer 8.2 MRIScortexLabelDECC path used with
``KeepHipAmyg01=0`` or ``1``. The default no gyrus-ambiens-fix label is
produced inside ``label-cortex --fix-ga``; that later script phase is separate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

from .inflate_python import vertex_normals


def _nearest(volume: np.ndarray, voxel: np.ndarray) -> np.ndarray:
    index = np.floor(voxel + 0.5).astype(np.int32)
    inside = np.all((index >= 0) & (index < np.array(volume.shape)), axis=1)
    index = np.clip(index, 0, np.array(volume.shape) - 1)
    values = volume[index[:, 0], index[:, 1], index[:, 2]].copy()
    values[~inside] = 0
    return values


def _voxel(points: np.ndarray, tk_to_vox: np.ndarray) -> np.ndarray:
    return points.astype(np.float64) @ tk_to_vox[:3, :3].T + tk_to_vox[:3, 3]


def _counts(volume: np.ndarray, label: int, radius: int) -> np.ndarray:
    counts = (volume == label).astype(np.int16)
    kernel = np.ones(2 * radius + 1, np.int16)
    for axis in range(3):
        counts = ndimage.convolve1d(counts, kernel, axis=axis, mode="nearest")
    return counts


def _sample_at_integer(volume: np.ndarray, voxel: np.ndarray, nearest: bool) -> np.ndarray:
    index = np.floor(voxel + 0.5 if nearest else voxel).astype(np.int32)
    index = np.clip(index, 0, np.array(volume.shape) - 1)
    return volume[index[:, 0], index[:, 1], index[:, 2]]


def _is_white_matter(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, (2, 41, 187, 186, 28, 60, 7, 46)) | ((labels >= 251) & (labels <= 255))


def _adjacency(faces: np.ndarray, nvertices: int) -> sparse.csr_matrix:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    source = np.concatenate((edges[:, 0], edges[:, 1]))
    target = np.concatenate((edges[:, 1], edges[:, 0]))
    graph = sparse.csr_matrix((np.ones(len(source), np.int8), (source, target)),
                              shape=(nvertices, nvertices))
    graph.data[:] = 1
    return graph


def _largest_component(marked: np.ndarray, graph: sparse.csr_matrix) -> np.ndarray:
    vertices = np.flatnonzero(marked)
    ncomponents, component = csgraph.connected_components(graph[vertices][:, vertices],
                                                            directed=False)
    if ncomponents <= 1:
        return marked
    counts = np.bincount(component, minlength=ncomponents)
    result = np.zeros_like(marked)
    result[vertices[component == int(np.argmax(counts))]] = True
    return result


def _dilate(marked: np.ndarray, graph: sparse.csr_matrix, times: int) -> np.ndarray:
    for _ in range(times):
        marked = marked | ((graph @ marked.astype(np.int16)) > 0)
    return marked


def _erode(marked: np.ndarray, graph: sparse.csr_matrix, times: int) -> np.ndarray:
    degree = np.diff(graph.indptr)
    for _ in range(times):
        marked = marked & ((graph @ marked.astype(np.int16)) == degree)
    return marked


def cortex_label_mask(vertices: np.ndarray, faces: np.ndarray,
                      segmentation: np.ndarray, tk_to_vox: np.ndarray,
                      keep_hip_amyg: bool = False) -> np.ndarray:
    """Return the ordered cortical vertex mask for the pinned call."""
    xyz = np.asarray(vertices, np.float32)
    aseg = np.asarray(segmentation, np.int16).copy()
    for code in (4, 43):
        locations = np.where(aseg == code)
        if locations[0].size:
            last_z = int(locations[2].max())
            posterior = aseg[:, :, last_z - 3:last_z + 1]
            posterior[posterior == code] = 0

    normals = vertex_normals(xyz, faces)
    marks = np.ones(len(xyz), bool)
    lesion = np.zeros(len(xyz), bool)
    sample0 = _nearest(aseg, _voxel(xyz, tk_to_vox))
    for step in range(5):
        depth = 0.5 * step
        sampled = (sample0 if step == 0 else
                   _nearest(aseg, _voxel(xyz.astype(np.float64)
                                          + depth * normals.astype(np.float64), tk_to_vox)))
        bad = np.isin(sampled, (4, 43, 5, 44, 14, 11, 50, 13, 52, 10, 49,
                                16, 28, 60))
        if not keep_hip_amyg:
            bad |= np.isin(sampled, (17, 18, 53, 54))
        bad |= (sampled >= 251) & (sampled <= 255)
        bad |= _is_white_matter(sampled) & _is_white_matter(sample0) & (sampled != sample0)
        marks &= ~bad
        lesion |= np.isin(sampled, (25, 57))
    marks &= ~lesion

    voxel = _voxel(xyz, tk_to_vox)
    left_thalamus = _sample_at_integer(_counts(aseg, 10, 2), voxel, nearest=True)
    right_thalamus = _sample_at_integer(_counts(aseg, 49, 2), voxel, nearest=True)
    left_ventricle = _sample_at_integer(_counts(aseg, 4, 2), voxel, nearest=True)
    right_ventricle = _sample_at_integer(_counts(aseg, 43, 2), voxel, nearest=True)
    marks &= ~(((left_thalamus > 0) & (left_thalamus >= right_thalamus)
                & (left_ventricle > 0))
               | ((right_thalamus > left_thalamus) & (right_ventricle > 0)))

    putamen_count = np.zeros(len(xyz), np.int16)
    putamen_adjacent = np.zeros(len(xyz), bool)
    for step in range(21):
        shifted = xyz.astype(np.float64).copy()
        shifted[:, 2] += 0.5 * step
        sampled = _nearest(aseg, _voxel(shifted, tk_to_vox))
        putamen = (sampled == 12) | (sampled == 51)
        putamen_count += putamen
        if step < 3:
            putamen_adjacent |= putamen
    marks &= ~(putamen_adjacent & (putamen_count > 10))

    gray = (aseg == 3) | (aseg == 42)
    nearby_gray = ndimage.maximum_filter(gray, size=11, mode="nearest")
    marks &= _sample_at_integer(nearby_gray, voxel, nearest=False)

    graph = _adjacency(np.asarray(faces, np.int32), len(xyz))
    noncortex = ~marks
    noncortex = _erode(_dilate(noncortex, graph, 1), graph, 1)
    noncortex = _largest_component(noncortex, graph) | (noncortex & lesion)
    cortex = ~noncortex
    cortex = _dilate(_erode(cortex, graph, 4), graph, 4)
    return _largest_component(cortex, graph)


def label_cortex(surface_path: str | Path, aseg_path: str | Path,
                 output_path: str | Path, keep_hip_amyg: bool = False) -> np.ndarray:
    """Create the native-format label and return its ordered vertex IDs."""
    vertices, faces = fsio.read_geometry(str(surface_path))
    image = nib.load(str(aseg_path))
    aseg = np.asanyarray(image.dataobj)
    tk_to_vox = np.linalg.inv(image.header.get_vox2ras_tkr())
    mask = cortex_label_mask(vertices, faces, aseg, tk_to_vox, keep_hip_amyg)
    selected = np.flatnonzero(mask)
    with Path(output_path).open("w") as stream:
        stream.write("#!ascii label  , from subject  vox2ras=TkReg\n")
        stream.write(f"{len(selected)}\n")
        for vertex in selected:
            x, y, z = vertices[vertex]
            stream.write(f"{vertex}  {x:.3f}  {y:.3f}  {z:.3f} 0.0000000000\n")
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path)
    parser.add_argument("aseg", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--keep-hip-amyg", action="store_true")
    args = parser.parse_args(argv)
    label_cortex(args.surface, args.aseg, args.output,
                 keep_hip_amyg=args.keep_hip_amyg)


if __name__ == "__main__":
    main()
