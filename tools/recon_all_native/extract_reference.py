#!/usr/bin/env python3
"""Extract candidate commands and resources without access to the FS installation.

FSTIME rows distinguish timed execution from command-like log mentions. Files
under a subject-directory fsaverage alias are listed as default-template
candidates; the on-server audit must verify the actual source path and hash.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--fs-home", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    commands = {}
    resources = {}
    fsaverage = {}
    home = args.fs_home.rstrip("/")
    path_pattern = re.compile(re.escape(home) + r"/[^\s\"'<>;,()\[\]{}]+")
    fsaverage_pattern = re.compile(r"/[^\s\"'<>;,()\[\]{}]*/fsaverage/((?:label|surf|mri)/[^\s\"'<>;,()\[\]{}]+)")
    command_pattern = re.compile(r"^(?:mri_|mris_|mrisp_|mris2|Ants|rca-|fs-|seg2cc$|lta_convert$)")
    data = args.log.read_text(errors="replace")
    for number, line in enumerate(data.splitlines(), 1):
        timing = re.match(r"@#@FSTIME\s+\S+\s+(\S+)\s+N\s+\d+\s+e\s+(\S+)", line)
        if timing:
            name = Path(timing[1]).name
            row = commands.setdefault(name, {"log_lines": [], "timings_seconds": []})
            row["log_lines"].append(number)
            row["timings_seconds"].append(float(timing[2]))
        try:
            words = shlex.split(line)
        except ValueError:
            words = []
        if words:
            if words[0] in {"cmdline", "fs_time", "time"}:
                words = words[1:]
            if words:
                name = Path(words[0]).name
                if command_pattern.match(name) and ":" not in name and "-Run-Time-" not in name:
                    commands.setdefault(name, {"log_lines": [], "timings_seconds": []})["log_lines"].append(number)
        for match in path_pattern.finditer(line):
            path = match.group().rstrip(":").removesuffix("...").split("#")[0]
            relative = str(Path(path).relative_to(home))
            if relative not in {".license", "license.txt"}:
                resources.setdefault(relative, []).append(number)
        for match in fsaverage_pattern.finditer(line):
            relative = "subjects/fsaverage/" + match[1].removesuffix("...")
            fsaverage.setdefault(relative, []).append(number)
    aliases = {}
    for name in list(commands):
        longer = [other for other in commands if other != name and other.startswith(name)]
        if len(longer) == 1 and commands[name]["timings_seconds"]:
            target = longer[0]
            commands[target]["timings_seconds"].extend(commands[name]["timings_seconds"])
            commands[target]["log_lines"].extend(commands[name]["log_lines"])
            aliases[name] = target
            del commands[name]
    for row in commands.values():
        row["log_lines"] = sorted(set(row["log_lines"]))
    report = {
        "reference_log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
        "fs_home": home, "reference_reported_success": "finished without error" in data,
        "commands": dict(sorted(commands.items())),
        "unique_prefix_timing_label_aliases": aliases,
        "fs_home_path_mentions": dict(sorted(resources.items())),
        "fsaverage_candidates_requiring_alias_verification": dict(sorted(fsaverage.items())),
        "closure_complete": False,
        "notes": [
            "A command-like log mention can be informational; timed rows prove a timed invocation.",
            "Nested calls may not have timing rows. Presence in bin/ and ELF status require server audit.",
            "Default neural weights, internal file reads and Python imports can be absent from the log.",
            "Do not add nested timing rows to wrapper timings when computing total wall time.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_requirements.json").write_text(json.dumps(report, indent=2) + "\n")
    paths = {"bin/" + name for name in commands} | set(resources) | set(fsaverage)
    (args.output_dir / "reference_files.txt").write_text("\n".join(sorted(paths)) + "\n")
    print(json.dumps({"command_candidates": len(commands),
                      "timed_command_names": sum(bool(row["timings_seconds"]) for row in commands.values()),
                      "fs_home_path_mentions": len(resources),
                      "fsaverage_candidates": len(fsaverage),
                      "closure_complete": False}))


if __name__ == "__main__":
    main()
