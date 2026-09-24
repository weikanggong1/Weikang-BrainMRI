"""Build a privacy-safe public report from aggregate UKB VBM results.

The input format is documented by ``public_report_input.schema.json``. This
script deliberately rejects case-level records, image paths, manifests and
absolute filesystem paths. It writes ``report.public.json`` and ``README.md``.

Example:
    python build_public_report.py --input aggregate_input.json --out-dir public_report
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse


ANALYSIS_KEYS = ("all10", "holdout9")
FORBIDDEN_KEYS = {
    "case_id",
    "case_ids",
    "case_records",
    "cases",
    "input_file",
    "input_files",
    "manifest",
    "output_file",
    "output_files",
    "path",
    "paths",
    "subject_id",
    "subject_ids",
    "subjects",
    "subjects_root",
}
ALLOWED_CASE_LABEL_LOCATIONS = {
    ("study", "tuning_case"),
    ("methods", "geometry_fallback", "case_label"),
}
SHA256 = re.compile(r"[0-9a-f]{64}")
CASE_LABEL = re.compile(r"case\d+")
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
NIFTI_PATH = re.compile(r"\.nii(?:\.gz)?(?:$|[\s,;:)])", re.IGNORECASE)
PRIVATE_PATH_FRAGMENT = re.compile(
    r"(?:/cwStorage/|/public/home/|\\cwStorage\\|\\public\\home\\)", re.IGNORECASE
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_keys(item: dict, required: set[str], location: str,
                 optional: set[str] | None = None) -> None:
    optional = optional or set()
    missing = required - set(item)
    require(not missing, f"{location}: missing {', '.join(sorted(missing))}")
    extra = set(item) - required - optional
    require(not extra, f"{location}: unexpected {', '.join(sorted(extra))}")


def finite_number(value, location: str, *, minimum=None, maximum=None) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{location}: expected a number")
    require(math.isfinite(value), f"{location}: expected a finite number")
    if minimum is not None:
        require(value >= minimum, f"{location}: must be at least {minimum}")
    if maximum is not None:
        require(value <= maximum, f"{location}: must be at most {maximum}")
    return float(value)


def positive_integer(value, location: str, *, maximum=None) -> int:
    finite_number(value, location, minimum=1, maximum=maximum)
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{location}: expected an integer")
    return value


def is_web_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def check_public_strings(value, location=()) -> None:
    """Reject data structures that can disclose case-level or local data."""
    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), f"{'.'.join(location)}: keys must be strings")
            require(key.lower() not in FORBIDDEN_KEYS,
                    f"{'.'.join(location + (key,))}: case-level or path key is forbidden")
            check_public_strings(child, location + (key,))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            check_public_strings(child, location + (str(index),))
        return
    if not isinstance(value, str):
        return

    require("\x00" not in value, f"{'.'.join(location)}: NUL byte is forbidden")
    require(not PRIVATE_PATH_FRAGMENT.search(value),
            f"{'.'.join(location)}: private server path is forbidden")
    require("manifest" not in value.lower(),
            f"{'.'.join(location)}: manifest references are forbidden")
    require(not NIFTI_PATH.search(value),
            f"{'.'.join(location)}: NIfTI file references are forbidden")
    if is_web_url(value):
        return
    require(not value.startswith(("/", "~/", "~\\")) and not WINDOWS_ABSOLUTE.match(value),
            f"{'.'.join(location)}: absolute filesystem paths are forbidden")
    if CASE_LABEL.fullmatch(value):
        require(location in ALLOWED_CASE_LABEL_LOCATIONS,
                f"{'.'.join(location)}: case labels are allowed only for tuning and fallback notes")


def validate_figure(figure: dict, index: int) -> None:
    location = f"figures[{index}]"
    require(isinstance(figure, dict), f"{location}: expected an object")
    require_keys(figure, {"file", "caption", "n"}, location)
    file = figure["file"]
    require(isinstance(file, str), f"{location}.file: expected a string")
    path = PurePosixPath(file)
    require(not path.is_absolute() and ".." not in path.parts,
            f"{location}.file: expected a safe relative path")
    require(path.suffix.lower() == ".png", f"{location}.file: only PNG is allowed")
    require(isinstance(figure["caption"], str) and figure["caption"].strip(),
            f"{location}.caption: expected non-empty text")
    require(positive_integer(figure["n"], f"{location}.n") == 10,
            f"{location}.n: public figures must aggregate all 10 cases")


def validate_metric(metric: dict, location: str, analysis_n: int) -> None:
    require(isinstance(metric, dict), f"{location}: expected an object")
    require_keys(metric, {
        "comparison", "metric", "statistic", "n", "left_label", "left_value",
        "right_label", "right_value", "difference_label", "difference_value",
    }, location, {"exact_permutation_p", "direction_counts", "note"})
    for field in ("comparison", "metric", "statistic", "left_label", "right_label",
                  "difference_label"):
        require(isinstance(metric[field], str) and metric[field].strip(),
                f"{location}.{field}: expected non-empty text")
    n = positive_integer(metric["n"], f"{location}.n", maximum=analysis_n)
    for field in ("left_value", "right_value", "difference_value"):
        finite_number(metric[field], f"{location}.{field}")
    if "exact_permutation_p" in metric:
        finite_number(metric["exact_permutation_p"],
                      f"{location}.exact_permutation_p", minimum=0, maximum=1)
    if "direction_counts" in metric:
        counts = metric["direction_counts"]
        require(isinstance(counts, dict), f"{location}.direction_counts: expected an object")
        require_keys(counts, {"left_better", "right_better", "ties"},
                     f"{location}.direction_counts")
        parsed = []
        for key in ("left_better", "right_better", "ties"):
            value = counts[key]
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    f"{location}.direction_counts.{key}: expected a non-negative integer")
            parsed.append(value)
        require(sum(parsed) == n,
                f"{location}.direction_counts: counts must sum to metric N={n}")
    if "note" in metric:
        require(isinstance(metric["note"], str) and metric["note"].strip(),
                f"{location}.note: expected non-empty text")


def validate_timing(timing: dict, location: str, analysis_n: int) -> None:
    require(isinstance(timing, dict), f"{location}: expected an object")
    require_keys(timing, {"method", "scope", "device", "statistic", "seconds", "n"},
                 location)
    for field in ("method", "scope", "device", "statistic"):
        require(isinstance(timing[field], str) and timing[field].strip(),
                f"{location}.{field}: expected non-empty text")
    finite_number(timing["seconds"], f"{location}.seconds", minimum=0)
    positive_integer(timing["n"], f"{location}.n", maximum=analysis_n)


def validate(data: dict) -> None:
    require(isinstance(data, dict), "input: expected a JSON object")
    require_keys(data, {
        "schema_version", "report", "study", "methods", "templates", "environment",
        "timing_definitions", "analyses", "figures", "limitations",
    }, "input")
    require(data["schema_version"] == "1.0", "schema_version: expected '1.0'")
    check_public_strings(data)

    report = data["report"]
    require(isinstance(report, dict), "report: expected an object")
    require_keys(report, {"title", "run_date", "cohort_description"}, "report")
    for field in ("title", "run_date", "cohort_description"):
        require(isinstance(report[field], str) and report[field].strip(),
                f"report.{field}: expected non-empty text")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", report["run_date"]) is not None,
            "report.run_date: expected YYYY-MM-DD")

    study = data["study"]
    require(isinstance(study, dict), "study: expected an object")
    require_keys(study, {"all_cases_n", "holdout_n", "tuning_case"}, "study")
    require(study["all_cases_n"] == 10, "study.all_cases_n: expected 10")
    require(study["holdout_n"] == 9, "study.holdout_n: expected 9")
    require(study["tuning_case"] == "case01", "study.tuning_case: expected case01")

    methods = data["methods"]
    require(isinstance(methods, dict), "methods: expected an object")
    require_keys(methods, {"reference", "gpu_alternative", "gdc", "geometry_fallback"},
                 "methods")
    for item_name in ("reference", "gpu_alternative", "gdc", "geometry_fallback"):
        require(isinstance(methods[item_name], dict), f"methods.{item_name}: expected an object")
    fallback = methods["geometry_fallback"]
    require_keys(methods["reference"], {"label", "description", "scope"},
                 "methods.reference")
    require_keys(methods["gpu_alternative"],
                 {"label", "description", "status", "equivalence"},
                 "methods.gpu_alternative")
    require_keys(methods["gdc"], {"applied", "reason"}, "methods.gdc")
    require_keys(fallback, {"applied", "case_label", "reason", "method"},
                 "methods.geometry_fallback")
    for group, fields in {
        "reference": ("label", "description", "scope"),
        "gpu_alternative": ("label", "description", "status", "equivalence"),
        "gdc": ("reason",),
        "geometry_fallback": ("case_label", "reason", "method"),
    }.items():
        for field in fields:
            require(isinstance(methods[group][field], str) and methods[group][field].strip(),
                    f"methods.{group}.{field}: expected non-empty text")
    require(methods["gpu_alternative"]["status"] == "experimental",
            "methods.gpu_alternative.status: expected experimental")
    require(methods["gpu_alternative"]["equivalence"] == "not_fnirt_equivalent",
            "methods.gpu_alternative.equivalence: expected not_fnirt_equivalent")
    require(methods["gdc"]["applied"] is False,
            "methods.gdc.applied: expected false for this experiment")
    require(fallback["applied"] is True and fallback["case_label"] == "case02",
            "methods.geometry_fallback: expected the case02 fallback record")

    templates = data["templates"]
    require(isinstance(templates, list) and len(templates) >= 2,
            "templates: expected at least two aggregate provenance records")
    seen_template_ids = set()
    for index, template in enumerate(templates):
        location = f"templates[{index}]"
        require(isinstance(template, dict), f"{location}: expected an object")
        require_keys(template, {"id", "label", "role", "url", "sha256", "distributed",
                                "source_status"}, location)
        for field in ("id", "label", "role", "sha256", "source_status"):
            require(isinstance(template[field], str) and template[field].strip(),
                    f"{location}.{field}: expected non-empty text")
        require(template["id"] not in seen_template_ids, f"{location}.id: duplicate value")
        seen_template_ids.add(template["id"])
        require(SHA256.fullmatch(template["sha256"]) is not None,
                f"{location}.sha256: expected lowercase SHA-256")
        require(template["url"] is None or is_web_url(template["url"]),
                f"{location}.url: expected an HTTP(S) URL or null")
        require(template["distributed"] is False,
                f"{location}.distributed: templates must not be included in the release")
    require("ukb_v1_5" in seen_template_ids,
            "templates: missing the ukb_v1_5 provenance record")

    environment = data["environment"]
    require(isinstance(environment, dict) and environment,
            "environment: expected non-empty aggregate environment fields")
    for group_name, fields in environment.items():
        require(isinstance(fields, dict) and fields,
                f"environment.{group_name}: expected a non-empty object")
        for key, value in fields.items():
            require(isinstance(value, str) and value.strip(),
                    f"environment.{group_name}.{key}: expected non-empty text")

    definitions = data["timing_definitions"]
    require(isinstance(definitions, list) and definitions,
            "timing_definitions: expected at least one record")
    for index, definition in enumerate(definitions):
        location = f"timing_definitions[{index}]"
        require(isinstance(definition, dict), f"{location}: expected an object")
        require_keys(definition, {"label", "includes", "excludes"}, location)
        for field in ("label", "includes", "excludes"):
            require(isinstance(definition[field], str) and definition[field].strip(),
                    f"{location}.{field}: expected non-empty text")

    analyses = data["analyses"]
    require(isinstance(analyses, dict) and set(analyses) == set(ANALYSIS_KEYS),
            "analyses: expected all10 and holdout9")
    expected_n = {"all10": 10, "holdout9": 9}
    for name in ANALYSIS_KEYS:
        analysis = analyses[name]
        location = f"analyses.{name}"
        require(isinstance(analysis, dict), f"{location}: expected an object")
        require_keys(analysis, {"label", "n", "finding", "metrics", "timings"}, location)
        require(analysis["n"] == expected_n[name],
                f"{location}.n: expected {expected_n[name]}")
        require(isinstance(analysis["finding"], str) and analysis["finding"].strip(),
                f"{location}.finding: expected non-empty text")
        require(isinstance(analysis["metrics"], list) and analysis["metrics"],
                f"{location}.metrics: expected at least one aggregate metric")
        require(isinstance(analysis["timings"], list),
                f"{location}.timings: expected a list")
        for index, metric in enumerate(analysis["metrics"]):
            validate_metric(metric, f"{location}.metrics[{index}]", analysis["n"])
        for index, timing in enumerate(analysis["timings"]):
            validate_timing(timing, f"{location}.timings[{index}]", analysis["n"])

    require(isinstance(data["figures"], list) and data["figures"],
            "figures: expected at least one PNG reference")
    for index, figure in enumerate(data["figures"]):
        validate_figure(figure, index)
    require(isinstance(data["limitations"], list) and data["limitations"],
            "limitations: expected at least one item")
    for index, item in enumerate(data["limitations"]):
        require(isinstance(item, str) and item.strip(),
                f"limitations[{index}]: expected non-empty text")


def format_number(value) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, (int, float)):
        return str(value)
    absolute = abs(value)
    if absolute != 0 and (absolute < 1e-4 or absolute >= 1e5):
        return f"{value:.4e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_p(value) -> str:
    return f"{value:.6g}"


def escape_table(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def environment_rows(environment: dict) -> list[str]:
    rows = []
    for group, fields in environment.items():
        for key, value in fields.items():
            rows.append(f"| {escape_table(group)} | {escape_table(key)} | {escape_table(value)} |")
    return rows


def render_metric_table(metrics: list[dict]) -> list[str]:
    rows = [
        "| Comparison | Metric | Statistic | N | Left | Right | Difference | Exact paired-label p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        left = f"{metric['left_label']}: {format_number(metric['left_value'])}"
        right = f"{metric['right_label']}: {format_number(metric['right_value'])}"
        difference = f"{metric['difference_label']}: {format_number(metric['difference_value'])}"
        p_value = metric.get("exact_permutation_p")
        rows.append("| " + " | ".join(escape_table(value) for value in (
            metric["comparison"], metric["metric"], metric["statistic"], metric["n"],
            left, right, difference, "NA" if p_value is None else format_p(p_value),
        )) + " |")
        counts = metric.get("direction_counts")
        if counts:
            rows.append(
                f"|  | Direction count |  | {metric['n']} | "
                f"{metric['left_label']}: {counts['left_better']} | "
                f"{metric['right_label']}: {counts['right_better']} | "
                f"ties: {counts['ties']} |  |"
            )
        if metric.get("note"):
            rows.append(f"|  | Note |  |  | {escape_table(metric['note'])} |  |  |  |")
    return rows


def render_timing_table(timings: list[dict]) -> list[str]:
    if not timings:
        return ["No separate timing summary was defined for this analysis."]
    rows = [
        "| Method | Scope | Device | Statistic | N | Seconds |",
        "|---|---|---|---:|---:|---:|",
    ]
    for timing in timings:
        rows.append("| " + " | ".join(escape_table(value) for value in (
            timing["method"], timing["scope"], timing["device"], timing["statistic"],
            timing["n"], format_number(timing["seconds"]),
        )) + " |")
    return rows


def render_markdown(data: dict) -> str:
    report = data["report"]
    study = data["study"]
    methods = data["methods"]
    main_finding = ". ".join(
        data["analyses"]["all10"]["finding"].split(". ")[:2]
    ).rstrip(".") + "."
    lines = [
        f"# {report['title']}",
        "",
        f"Run date: {report['run_date']}",
        "",
        report["cohort_description"],
        "",
        "This release reports an experimental PyTorch GPU alternative to the UK Biobank "
        "v1.5 FSL VBM workflow. The GPU method does not reproduce FNIRT and must not be "
        "treated as an FNIRT-equivalent implementation.",
        "",
        f"Main result: {main_finding}",
        "",
        "## Evaluation design",
        "",
        f"The full analysis contains {study['all_cases_n']} T1-weighted scans. "
        f"{study['tuning_case']} was used to select the GPU smoothness setting. The "
        f"holdout analysis contains the remaining {study['holdout_n']} scans.",
        "",
        "Template comparisons use the fixed union of template values above 0.01 "
        "(214,263 voxels at 2 mm); subject outputs do not define the mask. Each score compares one scan with the "
        "mean of the other scans in the same arm. The exact test enumerates every "
        "within-scan template-label assignment and rebuilds both leave-one-out references.",
        "",
        f"The FSL arm is {methods['reference']['label']}. "
        f"{methods['reference']['description']} {methods['reference']['scope']}",
        "",
        f"The GPU arm is {methods['gpu_alternative']['label']}. "
        f"{methods['gpu_alternative']['description']}",
        "",
        f"Gradient distortion correction was omitted because {methods['gdc']['reason']}",
        "",
        f"The standard transform failed for {methods['geometry_fallback']['case_label']}. "
        f"For that scan, {methods['geometry_fallback']['method']} was used because "
        f"{methods['geometry_fallback']['reason']}",
        "",
        "## Template provenance",
        "",
        "No template is included in this release.",
        "",
        "| Template | Role | Source | SHA-256 | Source status |",
        "|---|---|---|---|---|",
    ]
    for template in data["templates"]:
        source = template["url"] or "Not public"
        if template["url"]:
            source = f"[source]({template['url']})"
        lines.append(
            f"| {escape_table(template['label'])} | {escape_table(template['role'])} | "
            f"{source} | `{template['sha256']}` | "
            f"{escape_table(template['source_status'])} |"
        )

    lines.extend([
        "",
        "## Runtime environment",
        "",
        "| Group | Field | Value |",
        "|---|---|---|",
        *environment_rows(data["environment"]),
        "",
        "## Timing definitions",
        "",
    ])
    for definition in data["timing_definitions"]:
        lines.extend([
            f"### {definition['label']}",
            "",
            f"Includes: {definition['includes']}",
            "",
            f"Excludes: {definition['excludes']}",
            "",
        ])

    lines.extend([
        "## Aggregate results",
        "",
        "Leave-one-subject-out similarity measures cohort consistency. They do not measure "
        "anatomical accuracy. Timing values are interpretable only under the definitions and "
        "runtime environment recorded above.",
        "",
    ])
    for name in ANALYSIS_KEYS:
        analysis = data["analyses"][name]
        lines.extend([
            f"### {analysis['label']} (N={analysis['n']})",
            "",
            analysis["finding"],
            "",
            *render_metric_table(analysis["metrics"]),
            "",
            *render_timing_table(analysis["timings"]),
            "",
        ])

    lines.extend(["## Figures", ""])
    for figure in data["figures"]:
        lines.extend([
            f"![{figure['caption']}]({figure['file']})",
            "",
            f"{figure['caption'].rstrip('.')}. Aggregated over N={figure['n']}; "
            "no individual scan is shown.",
            "",
        ])

    lines.extend([
        "## Rebuild this report",
        "",
        "The committed aggregate input contains no case-level records or image paths. "
        "From the repository root, rebuild both text artifacts with:",
        "",
        "```bash",
        "python tools/experimental/ukb_vbm/build_public_report.py \\",
        "  --input validation/ukb_vbm/report.input.json \\",
        "  --out-dir validation/ukb_vbm",
        "```",
        "",
        "Verify the committed report and figures with:",
        "",
        "```bash",
        "(cd validation/ukb_vbm && sha256sum -c SHA256SUMS)",
        "```",
        "",
        "## Limits",
        "",
    ])
    for limitation in data["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def public_json(data: dict) -> dict:
    return {
        "schema_version": data["schema_version"],
        "release_status": "experimental",
        "fnirt_equivalent": False,
        "contains_case_level_records": False,
        "contains_subject_level_nifti": False,
        "contains_templates": False,
        "contains_aggregate_figures": True,
        "report": data["report"],
        "study": data["study"],
        "methods": data["methods"],
        "templates": data["templates"],
        "environment": data["environment"],
        "timing_definitions": data["timing_definitions"],
        "analyses": data["analyses"],
        "figures": data["figures"],
        "limitations": data["limitations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="aggregate JSON following public_report_input.schema.json")
    parser.add_argument("--out-dir", type=Path,
                        help="writes README.md and report.public.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    require(args.validate_only or args.out_dir is not None,
            "--out-dir is required unless --validate-only is used")

    data = json.loads(args.input.read_text())
    validate(data)
    if args.validate_only:
        print("aggregate input is valid for public release")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.public.json").write_text(
        json.dumps(public_json(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (args.out_dir / "README.md").write_text(render_markdown(data))
    print(json.dumps({
        "markdown": str(args.out_dir / "README.md"),
        "json": str(args.out_dir / "report.public.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
