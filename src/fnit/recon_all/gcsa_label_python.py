"""Native-free FreeSurfer 8.2 one-feature GCSA cortical annotation."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from .gcsa_aseg import relabel_with_aseg
from .gcsa_feature import mean_curvature_five, principal_directions
from .gcsa_finalize import apply_cortex_label, mode_filter_annotations, ordered_neighbors
from .gcsa_gibbs import GibbsModel
from .gcsa_initial import (
    InitialAtlas, initial_label, map_initial_nodes, read_ico_vertices, read_initial_atlas,
)
from .gcsa_islands import relabel_islands, vertex_areas
from .gcsa_reclassify import reclassify_gibbs


def write_annotation(path: str | Path, labels: np.ndarray, atlas: InitialAtlas) -> None:
    """Write the v2 color-table annotation bytes used by ``MRISwriteAnnotation``."""
    nvertices = len(labels)
    vertex_and_label = np.empty((nvertices, 2), dtype=">i4")
    vertex_and_label[:, 0] = np.arange(nvertices)
    vertex_and_label[:, 1] = labels
    source_name = atlas.source_name.encode() + b"\0"
    with Path(path).open("wb") as stream:
        stream.write(struct.pack(">i", nvertices))
        stream.write(vertex_and_label.tobytes())
        stream.write(struct.pack(">iiii", 1, -2, max(atlas.color_table) + 1,
                                 len(source_name)))
        stream.write(source_name)
        stream.write(struct.pack(">i", len(atlas.color_table)))
        for index, (name, red, green, blue, transparency) in sorted(atlas.color_table.items()):
            encoded = name.encode() + b"\0"
            stream.write(struct.pack(">ii", index, len(encoded)))
            stream.write(encoded)
            stream.write(struct.pack(">iiii", red, green, blue, transparency))


def label_surface(subject: str | Path, hemi: str, atlas_file: str | Path,
                  ico4_file: str | Path, ico7_file: str | Path,
                  output_file: str | Path, *, device: str = "cpu") -> dict:
    """Run the pinned ``mris_ca_label`` sequence from fixed input files."""
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be lh or rh")
    subject = Path(subject)
    surf = subject / "surf"
    label = subject / "label"
    started = time.perf_counter()
    atlas = read_initial_atlas(atlas_file, include_gibbs=True)
    smooth, faces = fsio.read_geometry(str(surf / f"{hemi}.smoothwm"))
    sphere, sphere_faces = fsio.read_geometry(str(surf / f"{hemi}.sphere.reg"))
    if not np.array_equal(faces, sphere_faces):
        raise ValueError("smoothwm and sphere.reg topology differs")
    classifier, prior = map_initial_nodes(
        sphere, read_ico_vertices(ico4_file), read_ico_vertices(ico7_file))
    image = nib.load(str(subject / "mri" / "aseg.presurf.mgz"))
    aseg = np.asanyarray(image.dataobj)
    tk_to_vox = np.linalg.inv(image.header.get_vox2ras_tkr())
    feature = mean_curvature_five(smooth, faces, device=device)
    principal = principal_directions(smooth, faces, device=device)
    neighbors = ordered_neighbors(faces, len(smooth))
    loaded = time.perf_counter()

    annotation = np.empty(len(smooth), dtype=np.int32)
    for vertex in range(len(smooth)):
        annotation[vertex] = initial_label(
            atlas.classifier_nodes[int(classifier[vertex])],
            atlas.prior_nodes[int(prior[vertex])], float(feature[vertex]))[0]
    initially_labeled = time.perf_counter()
    annotation = relabel_with_aseg(annotation, atlas, classifier, prior, feature,
                                   smooth, aseg, tk_to_vox)
    first_aseg = time.perf_counter()

    model = GibbsModel(atlas, classifier, prior, feature, smooth, neighbors,
                       principal, annotation)
    gibbs_history = reclassify_gibbs(model, atlas)
    gibbs = time.perf_counter()
    annotation = relabel_with_aseg(annotation, atlas, classifier, prior, feature,
                                   smooth, aseg, tk_to_vox)
    model.labels = annotation
    second_aseg = time.perf_counter()
    islands_history = relabel_islands(model, vertex_areas(smooth, faces))
    islands = time.perf_counter()
    annotation = mode_filter_annotations(annotation, faces, atlas.color_table)
    filtered = time.perf_counter()
    cortex_vertices = fsio.read_label(str(label / f"{hemi}.cortex.label"))
    annotation = apply_cortex_label(annotation, cortex_vertices, atlas,
                                    classifier, prior, feature, faces)
    corrected = time.perf_counter()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_annotation(output_file, annotation, atlas)
    finished = time.perf_counter()
    return {"vertices": len(annotation), "device": device,
            "gibbs_history": gibbs_history, "islands_history": islands_history,
            "seconds": {"prepare": loaded - started,
                        "initial_classifier": initially_labeled - loaded,
                        "aseg_first": first_aseg - initially_labeled,
                        "gibbs": gibbs - first_aseg,
                        "aseg_second": second_aseg - gibbs,
                        "islands": islands - second_aseg,
                        "mode_filter": filtered - islands,
                        "cortex_label": corrected - filtered,
                        "write": finished - corrected,
                        "total": finished - started}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--ico4", type=Path, required=True)
    parser.add_argument("--ico7", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = label_surface(args.subject, args.hemi, args.atlas, args.ico4,
                           args.ico7, args.output, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
