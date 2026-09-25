"""First FreeSurfer GCSA relabel pass using the frozen aseg volume."""

from __future__ import annotations

import numpy as np

from .gcsa_initial import InitialAtlas, initial_label


def _annotation(rgb: tuple[str, int, int, int, int]) -> int:
    return rgb[1] | (rgb[2] << 8) | (rgb[3] << 16)


def relabel_with_aseg(labels: np.ndarray, atlas: InitialAtlas,
                      classifier_indices: np.ndarray, prior_indices: np.ndarray,
                      feature: np.ndarray, smoothwm_vertices: np.ndarray,
                      aseg: np.ndarray, tk_to_vox: np.ndarray) -> np.ndarray:
    """Apply ``GCSArelabelWithAseg`` to a vertex annotation array."""
    out = np.asarray(labels, dtype=np.int32).copy()
    voxel = (np.asarray(smoothwm_vertices, dtype=np.float64)
             @ tk_to_vox[:3, :3].T + tk_to_vox[:3, 3])
    ijk = np.floor(voxel + 0.5).astype(np.int32)
    valid = np.all((ijk >= 0) & (ijk < np.array(aseg.shape)), axis=1)
    ijk = np.clip(ijk, 0, np.array(aseg.shape) - 1)
    sampled = aseg[ijk[:, 0], ijk[:, 1], ijk[:, 2]].copy()
    sampled[~valid] = 0

    by_name = {entry[0].lower(): _annotation(entry)
               for entry in atlas.color_table.values()}
    cc = by_name.get("corpuscallosum")
    cc_target = cc if cc is not None else by_name.get("medial_wall")
    medial = by_name.get("medial_wall", by_name.get("unknown"))
    cc_voxel = (sampled >= 251) & (sampled <= 255)
    medial_voxel = np.isin(sampled, (4, 5, 10, 11, 43, 44, 49, 50))
    if cc_target is not None:
        out[cc_voxel] = cc_target
    if medial is not None:
        out[~cc_voxel & medial_voxel] = medial

    if cc is not None:
        released = np.flatnonzero(~cc_voxel & ~medial_voxel & (labels == cc))
        for vertex in released:
            classifier = atlas.classifier_nodes[int(classifier_indices[vertex])]
            prior = atlas.prior_nodes[int(prior_indices[vertex])]
            candidates = [entry for entry in initial_label(classifier, prior,
                                                           float(feature[vertex]))[1]
                          if entry[0] != cc]
            out[vertex] = max(candidates, key=lambda entry: entry[2])[0] if candidates else -1
    return out
