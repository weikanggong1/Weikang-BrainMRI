"""Localize the first CA-label disagreement using fresh native snapshots."""

import json
import sys
import time
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.gcsa_aseg import relabel_with_aseg
from fnit.recon_all.gcsa_feature import mean_curvature_five, principal_directions
from fnit.recon_all.gcsa_finalize import apply_cortex_label, mode_filter_annotations, ordered_neighbors
from fnit.recon_all.gcsa_gibbs import GibbsModel
from fnit.recon_all.gcsa_initial import (
    initial_label, map_initial_nodes, read_ico_vertices, read_initial_atlas,
)
from fnit.recon_all.gcsa_islands import relabel_islands, vertex_areas
from fnit.recon_all.gcsa_reclassify import reclassify_gibbs


def main() -> None:
    bundle, subject, native_dir, hemi, atlas_code, output_name = (
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
        sys.argv[4], sys.argv[5], sys.argv[6])
    native_prefix = sys.argv[7]
    began = time.perf_counter()
    atlas = read_initial_atlas(bundle / "average" / f"{hemi}.{atlas_code}.atlas.acfb40.noaparc.i12.2016-08-02.gcs",
                               include_gibbs=True)
    smooth, faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.smoothwm"))
    sphere, _ = fsio.read_geometry(str(subject / "surf" / f"{hemi}.sphere.reg"))
    classifier, prior = map_initial_nodes(
        sphere, read_ico_vertices(bundle / "lib/bem/ic4.tri"),
        read_ico_vertices(bundle / "lib/bem/ic7.tri"))
    feature = mean_curvature_five(smooth, faces)
    principal = principal_directions(smooth, faces)
    neighbors = ordered_neighbors(faces, len(smooth))
    image = nib.load(str(subject / "mri" / "aseg.presurf.mgz"))
    aseg = np.asanyarray(image.dataobj)
    tk_to_vox = np.linalg.inv(image.header.get_vox2ras_tkr())
    stages = []

    def compare(name: str, labels: np.ndarray, filename: Path) -> None:
        expected, _, _ = fsio.read_annot(str(filename), orig_ids=True)
        bad = np.flatnonzero(labels != expected)
        stages.append({"stage": name, "exact": len(labels) - len(bad),
                       "vertices": len(labels), "first_bad": bad[:10].tolist(),
                       "pairs": [[int(expected[i]), int(labels[i])] for i in bad[:10]]})

    labels = np.asarray([
        initial_label(atlas.classifier_nodes[int(classifier[v])],
                      atlas.prior_nodes[int(prior[v])], float(feature[v]))[0]
        for v in range(len(smooth))], dtype=np.int32)
    labels = relabel_with_aseg(labels, atlas, classifier, prior, feature,
                               smooth, aseg, tk_to_vox)
    compare("initial_and_aseg1", labels, native_dir / f"{native_prefix}000.annot")
    model = GibbsModel(atlas, classifier, prior, feature, smooth, neighbors, principal, labels)

    def snapshot(iteration: int, current: np.ndarray) -> None:
        compare(f"gibbs_{iteration:03d}", current,
                native_dir / f"{native_prefix}{iteration:03d}.annot")

    gibbs_history = reclassify_gibbs(model, atlas, snapshot=snapshot)
    labels = relabel_with_aseg(labels, atlas, classifier, prior, feature,
                               smooth, aseg, tk_to_vox)
    model.labels = labels
    relabel_islands(model, vertex_areas(smooth, faces))
    compare("after_islands", labels, native_dir / f"{native_prefix}_post.annot")
    labels = mode_filter_annotations(labels, faces, atlas.color_table)
    labels = apply_cortex_label(labels,
                                fsio.read_label(str(subject / "label" / f"{hemi}.cortex.label")),
                                atlas, classifier, prior, feature, faces)
    compare("final", labels, subject / "label" / output_name)
    print(json.dumps({"hemi": hemi, "atlas": atlas_code,
                      "stages": stages, "gibbs_history": gibbs_history,
                      "total_seconds": time.perf_counter() - began}, indent=2))


if __name__ == "__main__":
    main()
