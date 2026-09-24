"""Checks for timing calculations, not wall-time performance claims."""

from validation.recon_all.stage_timing import (
    hemisphere_pairs, parse_stages, surface_process_pairs,
)


def test_stage_intervals_and_hemisphere_ideal_bound():
    commands = "\n".join([
        "#@# Sphere lh Wed Sep 23 21:14:28 CST 2026",
        "#@# Sphere rh Wed Sep 23 21:18:37 CST 2026",
        "#@# Surf Reg Wed Sep 23 21:20:51 CST 2026",
    ])
    stages = parse_stages(commands)
    assert stages == [
        {"stage": "Sphere lh", "seconds": 249.0},
        {"stage": "Sphere rh", "seconds": 134.0},
    ]
    assert hemisphere_pairs(stages) == [{
        "stage": "Sphere", "lh_seconds": 249.0, "rh_seconds": 134.0,
        "sequential_seconds": 383.0, "ideal_parallel_seconds": 249.0,
        "maximum_possible_saving_seconds": 134.0,
    }]


def test_unpaired_stage_not_reported_as_parallel():
    assert hemisphere_pairs([{"stage": "Sphere lh", "seconds": 10.0}]) == []


def test_process_pairs_exclude_shared_outvol_stages():
    def marker(name):
        return f"#@# {name} Wed Sep 23 21:14:28 CST 2026"

    def timing(tool, seconds):
        return f"@#@FSTIME  2026:09:23:21:14:28 {tool} N 6 e {seconds:.2f} S 0 U 0"

    log = "\n".join([
        marker("WhitePreAparc lh"), timing("mris_place_surface", 140),
        marker("WhitePreAparc rh"), timing("mris_place_surface", 125),
        marker("Surf Reg lh"), timing("mris_register", 190),
        marker("Surf Reg rh"), timing("mris_register", 164),
    ])
    assert surface_process_pairs(log) == [{
        "stage": "Surf Reg", "lh_seconds": 190.0, "rh_seconds": 164.0,
        "sequential_seconds": 354.0, "ideal_parallel_seconds": 190.0,
        "maximum_possible_saving_seconds": 164.0,
    }]
