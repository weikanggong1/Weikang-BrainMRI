"""Compare first pial outer-pass native optimizer trajectories by binary class."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LINE = re.compile(r"(?m)^(\d{3}): dt: ([0-9.]+), sse=([0-9.]+), rms=([0-9.]+)")


def first_pass(path: Path) -> tuple[dict[int, dict], str]:
    log = path.read_text()
    steps: dict[int, dict] = {}
    for number, dt, sse, rms in LINE.findall(log):
        index = int(number)
        if index in steps or index > 26:
            break
        steps[index] = {"dt": float(dt), "sse": float(sse), "rms": float(rms)}
    if set(steps) != set(range(27)):
        raise ValueError(f"expected first 26 accepted steps in {path}; got {sorted(steps)}")
    return steps, log


def first_crossing(steps: list[dict], key: str, threshold: float) -> int | None:
    return next((item["accepted_step"] for item in steps if abs(item[key]) > threshold), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-log", type=Path, required=True)
    parser.add_argument("--pristine-source-log", type=Path, required=True)
    parser.add_argument("--instrumented-source-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    traces = {}
    logs = {}
    for name, path in (
        ("installed_normal_cli", args.installed_log),
        ("pristine_source_normal_cli", args.pristine_source_log),
        ("instrumented_source_diagnostic", args.instrumented_source_log),
    ):
        traces[name], logs[name] = first_pass(path)
    rows = []
    for index in range(27):
        installed = traces["installed_normal_cli"][index]
        pristine = traces["pristine_source_normal_cli"][index]
        instrumented = traces["instrumented_source_diagnostic"][index]
        rows.append({
            "accepted_step": index,
            "installed": installed,
            "pristine_source": pristine,
            "instrumented_source": instrumented,
            "pristine_minus_installed_sse": pristine["sse"] - installed["sse"],
            "instrumented_minus_installed_sse": instrumented["sse"] - installed["sse"],
            "instrumented_minus_pristine_sse": instrumented["sse"] - pristine["sse"],
            "all_dt_equal": installed["dt"] == pristine["dt"] == instrumented["dt"],
        })
    report = {
        "reference_class": "normal installed and pristine-source CLI logs, plus separately instrumented source diagnostic; frozen input provenance in companion report",
        "first_outer_n_averages": 16,
        "first_outer_sigma": 2,
        "accepted_step_count": 26,
        "rows": rows,
        "all_dt_decisions_equal": all(item["all_dt_equal"] for item in rows),
        "first_sse_absolute_difference_over": {
            name: {f"{threshold:g}": first_crossing(rows, name, threshold)
                   for threshold in (0.1, 1, 10, 100, 400)}
            for name in ("pristine_minus_installed_sse", "instrumented_minus_installed_sse", "instrumented_minus_pristine_sse")
        },
        "rejected_trial_lines": {
            name: re.findall(r"(?m)^rms = .*time step reduction.*$\n\s*RMS increased, rejecting step", log)[:1]
            for name, log in logs.items()
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("all_dt_decisions_equal", "first_sse_absolute_difference_over", "rejected_trial_lines")}, indent=2))


if __name__ == "__main__":
    main()
