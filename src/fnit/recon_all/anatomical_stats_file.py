"""Write cortical ROI statistics from Python reconstruction outputs."""

from pathlib import Path

from .anatomical_stats_global import anatomical_stats_global_lines
from .anatomical_stats_rows import anatomical_stats_rows


def write_anatomical_stats(subject: str | Path, hemi: str, atlas: str,
                           surface: str, brainvol_stats: dict[str, float],
                           output: str | Path, *, device: str = "cuda:0") -> Path:
    """Write a stats table with native-matched numerical rows and headers."""
    if hemi not in {"lh", "rh"} or surface not in {"white", "pial"}:
        raise ValueError("Expected lh/rh hemisphere and white/pial surface")
    subject = Path(subject)
    prefix = subject / "surf" / hemi
    label = subject / "label"
    annotation = label / f"{hemi}.{atlas}.annot"
    cortex = None if atlas.startswith("BA_exvivo") else label / f"{hemi}.cortex.label"
    area = Path(str(prefix) + (".area.pial" if surface == "pial" else ".area"))
    thickness = Path(str(prefix) + ".thickness")
    global_lines = anatomical_stats_global_lines(
        area, thickness, annotation, cortex, brainvol_stats,
        subject / "mri" / "transforms" / "talairach.xfm", surface=surface)
    rows = anatomical_stats_rows(
        Path(str(prefix) + ".white"), Path(str(prefix) + ".pial"),
        Path(str(prefix) + "." + surface), area, thickness, annotation,
        cortex, device=device)
    lines = ["# Table of FreeSurfer cortical parcellation anatomical statistics",
             "# generating_program fnit",
             f"# subjectname {subject.name}", f"# hemi {hemi}",
             f"# AnnotationFile {annotation}", *global_lines,
             "# NTableCols 10",
             "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd",
             *rows]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output
