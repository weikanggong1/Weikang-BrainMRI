#!/usr/bin/env python3
"""Check the documented provisional aggregate gates in a comparator JSON report."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def _number(value, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError("Expected a finite nonnegative number within the metric's range")
    return value


def check_report(report):
    """Fail on a failed comparator, missing metrics, or any aggregate violation."""
    checks = report.get("checks", {})
    results = {"comparator": {"status": "passed" if report.get("passed") is True
                              and report.get("failed_checks") == [] else "failed"}}

    def check(name, operation, limit, *, lower=False, maximum=None):
        result = {"limit": limit, "operator": ">=" if lower else "<="}
        try:
            value = _number(operation(), maximum=maximum)
            passed = value >= limit if lower else value <= limit
            result.update(value=value, status="passed" if passed else "failed")
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            result.update(status="failed", reason=f"Missing or invalid metric: {error}")
        results[name] = result

    for hemi in ("lh", "rh"):
        for surface in ("white", "pial"):
            key = f"surf/{hemi}.{surface}"
            for metric, limit in (("mae", 0.01), ("p99_abs_error", 0.05)):
                check(f"{key}.{metric}", lambda: checks[key]["displacement_mm"][metric], limit)
        key = f"surf/{hemi}.thickness"
        check(f"{key}.mae", lambda: checks[key]["mae"], 0.005)
        for atlas in ("aparc", "aparc.DKTatlas", "aparc.a2009s"):
            key = f"label/{hemi}.{atlas}.annot"
            check(f"{key}.agreement", lambda: checks[key]["agreement"], 0.999,
                  lower=True, maximum=1)

    for volume in ("aseg.mgz", "aparc+aseg.mgz"):
        key = f"mri/{volume}"

        def foreground_macro_dice():
            rows = checks[key]["labels"]
            values = [_number(row["dice"], maximum=1) for label, row in rows.items() if label != "0"]
            if not values:
                raise ValueError("No foreground labels")
            return mean(values)

        check(f"{key}.foreground_macro_dice", foreground_macro_dice, 0.999,
              lower=True, maximum=1)

    failed = [name for name, row in results.items() if row["status"] != "passed"]
    return {"schema_version": 1, "profile": "fs820-provisional-aggregate-v1",
            "provisional": True, "passed": not failed, "checks": results,
            "failed_checks": failed,
            "not_assessed": ["independent generation", "runtime dependency closure",
                             "surface self-intersections", "runtime/speedup"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.comparison.read_bytes()
    report = json.loads(source)
    if not isinstance(report, dict):
        parser.error("comparison must be a JSON object")
    result = check_report(report)
    result.update(comparison=str(args.comparison), comparison_sha256=hashlib.sha256(source).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {len(result['failed_checks'])} failed gates; {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
