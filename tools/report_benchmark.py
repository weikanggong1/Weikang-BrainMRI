"""Render public benchmark summaries without opening clinical images or manifests.

python tools/report_benchmark.py --summary summary.public.json --out-dir report

Writes timing.csv, accuracy.csv, benchmark.png, and summary.md. The figure
requires matplotlib; aggregation uses only the Python standard library. Every
case, function, arm, and predefined comparison is retained, including pending
and failed results. No original subject paths are read or copied.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import median


ARMS = ("reference_cpu", "reference_gpu", "candidate_cpu", "candidate_gpu")
PAIRS = (("reference_cpu", "candidate_cpu"), ("reference_gpu", "candidate_gpu"),
         ("reference_cpu", "reference_gpu"), ("candidate_cpu", "candidate_gpu"))
FUNCTIONS = {"strip": "SynthStrip", "morph": "SynthMorph"}
PENDING = {None, "not_run", "running"}
METRICS = {
    "strip": (
        ("mask_dice", ("mask", "dice"), "fraction"),
        ("mask_disagreeing_voxels", ("mask", "disagreeing_voxels"), "voxels"),
        ("mask_geometry_error", ("mask", "geometry", "vox2world_max_abs_error"), "affine coefficient"),
        ("sdt_mae_mm", ("distance", "mean_abs_error"), "mm"),
        ("sdt_rmse_mm", ("distance", "rmse"), "mm"),
        ("sdt_max_error_mm", ("distance", "max_abs_error"), "mm"),
        ("stripped_image_nrmse", ("image", "nrmse_reference_rms"), "relative to reference RMS"),
        ("stripped_image_max_error", ("image", "max_abs_error"), "input intensity"),
    ),
    "morph": (
        ("warp_vector_mean_error_mm", ("transform", "vector_error_mean_mm"), "mm"),
        ("warp_vector_p95_error_mm", ("transform", "vector_error_p95_mm"), "mm"),
        ("warp_vector_max_error_mm", ("transform", "vector_error_max_mm"), "mm"),
        ("warp_component_rmse_mm", ("transform", "rmse"), "mm"),
        ("warp_component_max_error_mm", ("transform", "max_abs_error"), "mm"),
        ("warp_source_geometry_error", ("transform", "source_geometry", "vox2world_max_abs_error"), "affine coefficient"),
        ("warp_target_geometry_error", ("transform", "target_geometry", "vox2world_max_abs_error"), "affine coefficient"),
        ("moved_image_nrmse", ("moved", "nrmse_reference_rms"), "relative to reference RMS"),
        ("moved_image_max_error", ("moved", "max_abs_error"), "input intensity"),
    ),
}
JACOBIAN_METRICS = (
    ("jacobian_fraction_nonpositive", "fraction_nonpositive", "fraction of target-grid voxels"),
    ("jacobian_minimum", "minimum", "dimensionless"),
    ("jacobian_median", "median", "dimensionless"),
    ("jacobian_finite_fraction", "finite_fraction", "fraction of target-grid voxels"),
)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def lookup(data, path):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def label(function, arm):
    if arm == "reference_gpu" and function == "strip":
        return "Official source CUDA"
    return {"reference_cpu": "FreeSurfer CPU", "reference_gpu": "FreeSurfer GPU",
            "candidate_cpu": "PyTorch CPU", "candidate_gpu": "PyTorch GPU"}[arm]


def execution_status(record):
    raw = record.get("status")
    if raw in PENDING:
        return "pending"
    if raw != "success" or not finite_number(record.get("wall_seconds")) or record["wall_seconds"] <= 0:
        return "failure"
    return "success"


def comparison_status(entry, first, second, comparison):
    if comparison.get("available"):
        return "success", ""
    runs = entry.get("runs", {})
    statuses = [execution_status(runs.get(arm, {})) for arm in (first, second)]
    if "failure" in statuses or any(runs.get(arm, {}).get("quality_error") for arm in (first, second)):
        return "failure", "run_or_quality_failure"
    if "pending" in statuses:
        return "pending", "awaiting_runs"
    return "failure", "comparison_unavailable"


def build_rows(summary):
    cases = summary.get("case_ids", [])
    if (not cases or len(cases) != summary.get("case_count") or len(cases) != len(set(cases))
            or any(not isinstance(case, str) or not re.fullmatch(r"case[0-9]+", case) for case in cases)):
        raise ValueError("summary must contain unique public case IDs matching case_count")
    timing, accuracy = [], []
    for function in FUNCTIONS:
        entries = {entry["case_id"]: entry for entry in summary.get("functions", {}).get(function, {}).get("cases", [])}
        for case in cases:
            entry = entries.get(case, {})
            runs = entry.get("runs", {})
            for arm in ARMS:
                record = runs.get(arm, {})
                timing.append({
                    "row_type": "run", "function": function, "case_id": case, "arm": arm,
                    "label": label(function, arm), "comparison": "", "first_arm": "", "second_arm": "",
                    "status": execution_status(record), "wall_seconds": record.get("wall_seconds"),
                    "paired_elapsed_ratio_first_over_second": None,
                    "max_rss_kib": record.get("max_rss_kib"), "threads": record.get("threads"),
                    "cpu_affinity": record.get("cpu_affinity"), "cuda_visible_devices": record.get("cuda_visible_devices"),
                    "warmup_runs_requested": record.get("warmup_runs_requested"),
                    "quality_status": "failure" if record.get("quality_error") else "",
                })
            for first, second in PAIRS:
                key = f"{first}__{second}"
                a, b = runs.get(first, {}), runs.get(second, {})
                states = [execution_status(a), execution_status(b)]
                ratio_status = "failure" if "failure" in states else ("pending" if "pending" in states else "success")
                timing.append({
                    "row_type": "paired_ratio", "function": function, "case_id": case, "arm": "",
                    "label": f"{label(function, first)} / {label(function, second)}", "comparison": key,
                    "first_arm": first, "second_arm": second, "status": ratio_status,
                    "wall_seconds": None,
                    "paired_elapsed_ratio_first_over_second": a["wall_seconds"] / b["wall_seconds"] if ratio_status == "success" else None,
                    "max_rss_kib": None, "threads": None, "cpu_affinity": None, "cuda_visible_devices": None,
                    "warmup_runs_requested": None, "quality_status": "",
                })
                comparison = entry.get("comparisons", {}).get(key, {})
                status, detail = comparison_status(entry, first, second, comparison)
                for metric, path, unit in METRICS[function]:
                    value = lookup(comparison.get("outputs", {}), path) if status == "success" else None
                    metric_status = status if status != "success" or finite_number(value) else "failure"
                    accuracy.append({"function": function, "case_id": case, "kind": "comparison", "arm": "",
                                     "comparison": key, "metric": metric, "value": value if finite_number(value) else None,
                                     "units": unit, "status": metric_status,
                                     "detail": detail if status != "success" else ("" if finite_number(value) else "missing_or_nonfinite_metric")})
            if function == "morph":
                for arm in ARMS:
                    record = runs.get(arm, {})
                    status = execution_status(record)
                    if record.get("quality_error"):
                        status = "failure"
                    for metric, key, unit in JACOBIAN_METRICS:
                        value = lookup(record, ("forward_warp_jacobian", key)) if status == "success" else None
                        metric_status = status if status != "success" or finite_number(value) else "failure"
                        accuracy.append({"function": function, "case_id": case, "kind": "single_arm", "arm": arm,
                                         "comparison": "", "metric": metric, "value": value if finite_number(value) else None,
                                         "units": unit, "status": metric_status,
                                         "detail": "" if metric_status == "success" else ("awaiting_run" if metric_status == "pending" else "run_or_jacobian_failure")})
    return cases, timing, accuracy


def quantile(values, fraction):
    values = sorted(values)
    point = (len(values) - 1) * fraction
    left = int(point)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (point - left)


def distribution(values, *, precision=4):
    if not values:
        return "pending / unavailable"
    return f"{median(values):.{precision}g} [{quantile(values, .25):.{precision}g}, {quantile(values, .75):.{precision}g}]"


def status_counts(rows):
    return {state: sum(row["status"] == state for row in rows) for state in ("success", "pending", "failure")}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(cases, timing, accuracy):
    runs = [row for row in timing if row["row_type"] == "run"]
    counts = status_counts(runs)
    quality_counts = status_counts(accuracy)
    lines = [
        f"The benchmark includes {len(cases)} T1w cases and {len(runs)} requested fresh-process runs: "
        f"{counts['success']} completed, {counts['pending']} pending, and {counts['failure']} failed.",
        "", "Wall-clock time includes process/framework startup, model and input loading, inference, and output writing. "
        "Filesystem caches were not reset and other queues may share host resources. "
        "Paired elapsed-time ratios are descriptive; they do not establish performance with exclusive hardware access.",
        "", "SynthStrip's **Official source CUDA** arm runs the unchanged FreeSurfer script in the project's CUDA-enabled "
        "Python environment. It is distinct from the bundled FreeSurfer command runtime.",
        "", "All brackets below contain the 25th and 75th percentiles. Missing and failed observations remain listed in the CSV files.",
        "", "| Function | Arm | Completed / requested | Pending | Failed | Seconds, median [IQR] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for function in FUNCTIONS:
        for arm in ARMS:
            rows = [row for row in runs if row["function"] == function and row["arm"] == arm]
            count = status_counts(rows)
            values = [row["wall_seconds"] for row in rows if row["status"] == "success"]
            lines.append(f"| {FUNCTIONS[function]} | {label(function, arm)} | {count['success']} / {len(cases)} | "
                         f"{count['pending']} | {count['failure']} | {distribution(values)} |")
    lines += ["", "| Function | Paired elapsed-time ratio (first / second) | Available / requested | Pending | Failed | Median [IQR] |",
              "|---|---|---:|---:|---:|---:|"]
    for function in FUNCTIONS:
        for first, second in PAIRS:
            rows = [row for row in timing if row["function"] == function and row["row_type"] == "paired_ratio"
                    and row["comparison"] == f"{first}__{second}"]
            count = status_counts(rows)
            values = [row["paired_elapsed_ratio_first_over_second"] for row in rows if row["status"] == "success"]
            lines.append(f"| {FUNCTIONS[function]} | {label(function, first)} / {label(function, second)} | "
                         f"{count['success']} / {len(cases)} | {count['pending']} | {count['failure']} | {distribution(values)} |")
    lines += ["", "NRMSE is the image RMSE divided by the reference image RMS. Warp errors use physical RAS displacement "
              "in millimetres. Jacobian fractions cover the entire forward-warp target grid.",
              "", f"The detailed accuracy table contains {quality_counts['success']} available, {quality_counts['pending']} pending, "
              f"and {quality_counts['failure']} failed metric observations.",
              "", "| Function | Comparison | Metric | Available / requested | Pending | Failed | Median [IQR] | Min–max |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    selected = {"strip": ("mask_dice", "mask_disagreeing_voxels", "sdt_max_error_mm", "sdt_mae_mm", "stripped_image_nrmse"),
                "morph": ("warp_vector_mean_error_mm", "warp_vector_max_error_mm", "moved_image_nrmse")}
    for function in FUNCTIONS:
        for first, second in PAIRS:
            for metric in selected[function]:
                rows = [row for row in accuracy if row["function"] == function and row["comparison"] == f"{first}__{second}"
                        and row["metric"] == metric]
                count = status_counts(rows)
                values = [row["value"] for row in rows if row["status"] == "success"]
                precision = 10 if metric == "mask_dice" else 4
                bounds = f"{min(values):.{precision}g}–{max(values):.{precision}g}" if values else "unavailable"
                lines.append(f"| {FUNCTIONS[function]} | {label(function, first)} vs {label(function, second)} | {metric} | "
                             f"{count['success']} / {len(cases)} | {count['pending']} | {count['failure']} | {distribution(values, precision=precision)} | {bounds} |")
    for arm in ARMS:
        rows = [row for row in accuracy if row["arm"] == arm and row["metric"] == "jacobian_fraction_nonpositive"]
        count = status_counts(rows)
        values = [row["value"] for row in rows if row["status"] == "success"]
        bounds = f"{min(values):.4g}–{max(values):.4g}" if values else "unavailable"
        lines.append(f"| SynthMorph | {label('morph', arm)} | jacobian_fraction_nonpositive | "
                     f"{count['success']} / {len(cases)} | {count['pending']} | {count['failure']} | {distribution(values)} | {bounds} |")
    return "\n".join(lines) + "\n"


def plot(timing, cases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ("reference_cpu", "candidate_cpu", "reference_gpu", "candidate_gpu")
    colors = ("#527CA5", "#DE9867", "#245785", "#BD5C27")
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    runs = [row for row in timing if row["row_type"] == "run"]
    for axis, function in zip(axes, FUNCTIONS):
        all_values, ticks = [], []
        for position, (arm, color) in enumerate(zip(order, colors), 1):
            rows = [row for row in runs if row["function"] == function and row["arm"] == arm]
            successful = [row for row in rows if row["status"] == "success"]
            values = [row["wall_seconds"] for row in successful]
            count = status_counts(rows)
            arm_label = label(function, arm)
            if arm_label == "Official source CUDA":
                arm_label = "Official source\nCUDA*"
            else:
                arm_label = arm_label.replace(" ", "\n", 1)
            ticks.append(f"{arm_label}\nn={count['success']}/{len(cases)}; P={count['pending']} F={count['failure']}")
            if values:
                boxes = axis.boxplot([values], positions=[position], widths=.52, patch_artist=True, showfliers=False,
                                     medianprops={"color": "#202020", "linewidth": 1.6})
                boxes["boxes"][0].set(facecolor=color, alpha=.35, edgecolor=color)
                offsets = [(cases.index(row["case_id"]) / max(len(cases) - 1, 1) - .5) * .28 for row in successful]
                axis.scatter([position + offset for offset in offsets], values, s=23, c=color, alpha=.9, linewidths=0, zorder=3)
                all_values.extend(values)
            else:
                axis.text(position, .04, "pending" if count["pending"] else "failed", ha="center",
                          transform=axis.get_xaxis_transform(), color="#666666", fontsize=9)
        axis.set_title(FUNCTIONS[function], fontsize=13, fontweight="bold")
        axis.set_xticks(range(1, 5), ticks, fontsize=9)
        axis.set_xlim(.45, 4.55)
        if all_values and max(all_values) / min(all_values) > 8:
            from matplotlib.ticker import LogLocator, ScalarFormatter
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.set_ylabel("Wall-clock seconds (log scale)")
        else:
            axis.set_ylim(bottom=0)
            axis.set_ylabel("Wall-clock seconds")
        axis.grid(axis="y", alpha=.22)
        axis.spines[["top", "right"]].set_visible(False)
    counts = status_counts(runs)
    figure.suptitle(f"Fresh-process elapsed time · {counts['success']}/{len(runs)} complete, "
                    f"{counts['pending']} pending, {counts['failure']} failed", fontsize=14)
    figure.text(.5, .05, "* SynthStrip: unchanged official script in the project CUDA-enabled Python.  P = pending; F = failed.",
                ha="center", fontsize=9)
    figure.text(.5, .018, "Includes startup, model/input loading, inference and output writing. Filesystem caches and shared-resource effects are not controlled.",
                ha="center", fontsize=8.5, color="#555555")
    figure.subplots_adjust(left=.075, right=.985, top=.86, bottom=.25, wspace=.26)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    cases, timing, accuracy = build_rows(summary)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "timing.csv", timing)
    write_csv(root / "accuracy.csv", accuracy)
    (root / "summary.md").write_text(markdown(cases, timing, accuracy))
    plot(timing, cases, root / "benchmark.png")
    print(json.dumps({"cases": len(cases), "timing_rows": len(timing), "accuracy_rows": len(accuracy),
                      "run_status": status_counts([row for row in timing if row["row_type"] == "run"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
