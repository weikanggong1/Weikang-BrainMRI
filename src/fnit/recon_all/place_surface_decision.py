"""FreeSurfer 8.2 pial placement time-step decision for the fixed T1 call."""

from __future__ import annotations


def pial_step_decision(
    last_sse: float, last_rms: float, sse: float, rms: float,
    dt: float, reductions: int,
) -> tuple[float, int, bool, bool, bool]:
    """Return next dt, reduction count, reduced, rejected, stop.

    This is the check_tol=0, l_location=0 branch of MRISpositionSurface.
    Its fixed parameters are tol=1e-4, REDUCTION_PCT=0.5, and
    MAX_REDUCTIONS=2.
    """
    reduce = 100.0 * (last_sse - sse) / last_sse < 1e-4 or rms > last_rms - 0.05
    reductions += int(reduce)
    if reduce:
        dt *= 0.5
    reject = reduce and rms > last_rms
    return dt, reductions, reduce, reject, reductions > 2
