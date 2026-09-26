"""FreeSurfer 8.2 conventional-sphere integration scale decisions."""

from __future__ import annotations

import math


def next_standard_sphere_scale(stage: str, weight: float, averages: int,
                               steps_at_scale: int, starting_sse: float,
                               ending_sse: float, selected_dt: float
                               ) -> tuple[str, float, int, int] | None:
    """Advance the first-pass ``MRISintegrate`` schedule after one update.

    The source uses ``100 * (old_sse - sse) / sse < tol`` and scales tol by
    ``sqrt((navgs + 1) / 1024)``. A zero time step or 25 updates also ends a
    scale. ``None`` marks the end of the nonlinear fold-repair sweep.
    """
    base_tol = 0.01 if stage == "nonlinear_repair" else 0.5
    tol = base_tol * math.sqrt((averages + 1) / 1024)
    complete = (ending_sse == 0 or selected_dt == 0 or steps_at_scale + 1 >= 25
                or 100 * (starting_sse - ending_sse) / ending_sse < tol)
    if not complete:
        return stage, weight, averages, steps_at_scale + 1
    if averages:
        return stage, weight, averages // 4, 0
    if stage in ("initial_repair", "nonlinear_repair"):
        weights = (1e-6, 1e-5, 1e-3, 1e-2, 0.1)
        next_weight = weights.index(weight) + 1
        if next_weight < len(weights):
            return stage, weights[next_weight], (32 if stage == "nonlinear_repair" else 1024), 0
        if stage == "initial_repair":
            return "unfold_epoch_1", 0.1, 1024, 0
        return None
    if stage == "unfold_epoch_1":
        return "unfold_epoch_2", 1.0, 1024, 0
    if stage == "unfold_epoch_2":
        return "nonlinear_repair", 1e-6, 32, 0
    raise ValueError(f"unsupported first-pass stage: {stage}")
