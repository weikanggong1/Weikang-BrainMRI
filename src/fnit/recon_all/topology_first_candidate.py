"""Source-order nonintersecting edge selection for the first GA patch candidate.

This primitive accepts an already MRI-scored candidate table in qsort order.
It does not implement patch fitness, genetic search, or all-defect repair.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .topology_preflight_python import _edge_intersects


@njit
def accept_first_candidate_edges(canonical: np.ndarray, candidates: np.ndarray,
                    used: np.ndarray, local_edges: np.ndarray) -> np.ndarray:
    active = np.empty((len(local_edges) + len(candidates), 2), np.int32)
    active[:len(local_edges)] = local_edges
    nactive = len(local_edges)
    accepted = np.zeros(len(candidates), np.bool_)
    for index in range(len(candidates)):
        if used[index] == 1:  # USED_IN_TESSELLATION
            continue
        a, b = candidates[index]
        intersects = False
        for prior in range(nactive):
            if _edge_intersects(canonical, a, b,
                                active[prior, 0], active[prior, 1]):
                intersects = True
                break
        if not intersects:
            accepted[index] = True
            active[nactive, 0] = a
            active[nactive, 1] = b
            nactive += 1
    return accepted
