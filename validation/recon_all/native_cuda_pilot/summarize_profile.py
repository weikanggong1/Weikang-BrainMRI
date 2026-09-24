#!/usr/bin/env python3
"""Aggregate the two scoped PROFILE record types from one replay log."""

import json
import pathlib
import statistics
import sys


log = pathlib.Path(sys.argv[1])
records = {}
for line in log.read_text(errors="replace").splitlines():
    if not line.startswith("#@# PROFILE "):
        continue
    parts = line.split()
    kind = parts[2]
    values = {key: float(value) for token in parts[3:] if "=" in token for key, value in [token.split("=", 1)]}
    records.setdefault(kind, []).append(values)

result = {}
for kind, rows in records.items():
    keys = sorted(rows[0])
    summary = {key: round(sum(row[key] for row in rows), 3) for key in keys}
    total = summary["total_ms"]
    summary["percent_total"] = {
        key: round(value / total * 100, 2)
        for key, value in summary.items()
        if key.endswith("_ms") and key != "total_ms"
    }
    summary["median_call_ms"] = round(statistics.median(row["total_ms"] for row in rows), 3)
    result[kind] = {"calls": len(rows), "sum": summary}
print(json.dumps(result, indent=2))
