"""Source-derived average and sigma transitions for FreeSurfer registration."""

from __future__ import annotations

import math


_SMOOTHWM_SIGMAS = (4.0, 2.0, 1.0, 0.5)


def next_smoothwm_scale(stage: str, sigma_index: int, averages: int,
                        steps_at_scale: int, starting_sse: float,
                        ending_sse: float, selected_dt: float,
                        negative_faces: int = 0,
                        ) -> tuple[str, int, int, int] | None:
    """Advance the default smoothwm registration after one saved update.

    `None` means registration integration is complete. `negative_faces` is
    used only after the final smoothwm sigma to choose fold cleanup.
    """
    if stage not in ("smoothwm", "fold_cleanup") or not 0 <= sigma_index < len(_SMOOTHWM_SIGMAS):
        raise ValueError("invalid registration stage or sigma")
    tolerance = (0.1 if stage == "fold_cleanup" else 1.0) * math.sqrt(
        (averages + 1) / 1024)
    complete = (ending_sse == 0 or selected_dt == 0 or steps_at_scale + 1 >= 25
                or 100 * (starting_sse - ending_sse) / ending_sse < tolerance)
    if not complete:
        return stage, sigma_index, averages, steps_at_scale + 1
    if averages:
        return stage, sigma_index, averages // 4, 0
    if stage == "fold_cleanup":
        return None
    if sigma_index + 1 < len(_SMOOTHWM_SIGMAS):
        return stage, sigma_index + 1, 1024, 0
    return ("fold_cleanup", sigma_index, 64, 0) if negative_faces else None


def next_sulc_scale(phase: str, sigma_index: int, averages: int,
                    steps_at_scale: int, starting_sse: float, ending_sse: float,
                    selected_dt: float) -> tuple[str, int, int, int] | None:
    """Advance FreeSurfer's default sulc pass, including its one big-average epoch."""
    if phase not in ("big", "normal") or not 0 <= sigma_index < len(_SMOOTHWM_SIGMAS):
        raise ValueError("invalid sulc registration phase or sigma")
    if phase == "big" and sigma_index != 0:
        raise ValueError("big-average sulc pass runs only at sigma 4")
    tolerance = 0.5 * math.sqrt((averages + 1) / 1024)
    complete = (ending_sse == 0 or selected_dt == 0 or steps_at_scale + 1 >= 25
                or 100 * (starting_sse - ending_sse) / ending_sse < tolerance)
    if not complete:
        return phase, sigma_index, averages, steps_at_scale + 1
    if averages:
        return phase, sigma_index, averages // 4, 0
    if phase == "big":
        return "normal", 0, 1024, 0
    return ("normal", sigma_index + 1, 1024, 0) if sigma_index + 1 < len(_SMOOTHWM_SIGMAS) else None
