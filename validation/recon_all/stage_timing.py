"""Summarize recon-all stage timing from its recorded command script.

Stage times include command startup and are useful for prioritizing experiments.
Pair savings are ideal upper bounds; they are not measured speedups.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re


_STAGE = re.compile(
    r"^#@# (?P<name>.+?) "
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
    r"(?P<date>[A-Z][a-z]{2} +\d+ \d\d:\d\d:\d\d) "
    r"\S+ (?P<year>\d{4})$"
)
_FSTIME = re.compile(r"^@#@FSTIME +\S+ (?P<tool>\S+) N \d+ e (?P<seconds>\d+(?:\.\d+)?)\b")
_SURFACE_PAIRS = {
    "QSphere": "mris_sphere",
    "Fix Topology": "mris_fix_topology",
    "Curv .H and .K": "mris_curvature",
    "Sphere": "mris_sphere",
    "Surf Reg": "mris_register",
    "T1PialSurf": "mris_place_surface",
}


def parse_stages(command_text: str) -> list[dict]:
    """Return timed stages, omitting the last stage without an end marker."""
    markers = []
    for line in command_text.splitlines():
        match = _STAGE.match(line)
        if match:
            stamp = datetime.strptime(
                f"{match['date']} {match['year']}", "%b %d %H:%M:%S %Y"
            )
            markers.append((match["name"].strip(), stamp))
    result = []
    for (name, start), (_, end) in zip(markers, markers[1:]):
        seconds = (end - start).total_seconds()
        if seconds < 0:
            raise ValueError(f"stage timestamps out of order: {name}")
        result.append({"stage": name, "seconds": seconds})
    return result


def hemisphere_pairs(stages: list[dict]) -> list[dict]:
    """Calculate ideal left/right bounds, without asserting safe concurrency."""
    groups: dict[str, dict[str, float]] = {}
    for stage in stages:
        match = re.fullmatch(r"(.+) (lh|rh)", stage["stage"])
        if match:
            groups.setdefault(match[1], {})[match[2]] = stage["seconds"]
    pairs = []
    for name, values in groups.items():
        if set(values) != {"lh", "rh"}:
            continue
        left, right = values["lh"], values["rh"]
        pairs.append({
            "stage": name,
            "lh_seconds": left,
            "rh_seconds": right,
            "sequential_seconds": left + right,
            "ideal_parallel_seconds": max(left, right),
            "maximum_possible_saving_seconds": min(left, right),
        })
    return sorted(pairs, key=lambda item: item["maximum_possible_saving_seconds"], reverse=True)


def surface_process_pairs(log_text: str) -> list[dict]:
    """Profile surface commands with separate hemisphere outputs from a log.

    A pair's saving is a sequential-time bound, not a prediction under CPU
    contention. WhitePreAparc and WhiteSurfs are intentionally excluded because
    their reference commands write the same --outvol for both hemispheres.
    """
    elapsed: dict[str, float] = {}
    stage = ""
    for line in log_text.splitlines():
        marker = _STAGE.match(line)
        if marker:
            stage = marker["name"].strip()
            continue
        record = _FSTIME.match(line)
        if not record:
            continue
        hemisphere = re.fullmatch(r"(.+) (lh|rh)", stage)
        if hemisphere and _SURFACE_PAIRS.get(hemisphere[1]) == record["tool"]:
            elapsed[stage] = elapsed.get(stage, 0) + float(record["seconds"])
    return hemisphere_pairs([
        {"stage": stage, "seconds": seconds}
        for stage, seconds in elapsed.items()
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands", type=Path, help="recon-all.cmd from a completed run")
    parser.add_argument("--log", type=Path, help="recon-all.log from the same run")
    args = parser.parse_args()
    stages = parse_stages(args.commands.read_text())
    report = {
        "stages": sorted(stages, key=lambda item: item["seconds"], reverse=True),
        "stage_pairs_ideal_bounds_not_concurrency_approval": hemisphere_pairs(stages),
    }
    if args.log:
        report["surface_process_pairs_ideal_bounds"] = surface_process_pairs(args.log.read_text())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
