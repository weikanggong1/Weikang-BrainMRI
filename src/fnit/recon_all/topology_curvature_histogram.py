"""First-topology-defect principal-curvature PDFs from vertex k1/k2 values.

The two one-dimensional histograms and the joint 100-by-100 PDF follow the
pinned FreeSurfer 8.2 ``mrisComputePrincipalCurvatureDistributions`` path.
Principal-curvature fitting and topology search are separate operations.
"""

from __future__ import annotations

import numpy as np


def _source_mean(bins: np.ndarray, counts: np.ndarray,
                 bin_size: np.float32) -> np.float32:
    maximum = np.max(counts)
    weighted = total = np.float32(0)
    for bin_value, count in zip(bins, counts):
        if count < float(maximum) / 100.0:
            continue
        value = np.float32(float(bin_value) - float(bin_size) / 2.0)
        weighted = np.float32(weighted + np.float32(value * count))
        total = np.float32(total + count)
    return np.float32(weighted / total)


def curvature_histograms(k1: np.ndarray, k2: np.ndarray) -> dict[str, object]:
    """Build 100-bin k1/k2 PDFs and source-order normalized joint PDF."""
    first = np.asarray(k1, np.float32)
    second = np.asarray(k2, np.float32)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("k1 and k2 must be equally sized vertex arrays")
    starts = [np.maximum(np.float32(-3), np.min(values)) for values in (first, second)]
    ends = [np.minimum(np.float32(3), np.max(values)) for values in (first, second)]
    widths = [np.float32(np.float32(end - start) / np.float32(100))
              for start, end in zip(starts, ends)]
    bins = []
    counts = []
    indices = []
    for values, start, width in zip((first, second), starts, widths):
        centers = np.empty(100, np.float32)
        point = np.float32(start)
        for index in range(100):
            centers[index] = point
            point = np.float32(point + width)
        index = np.clip(np.trunc(np.float32(np.float32(values - start) / width)).astype(np.int32), 0, 99)
        count = np.bincount(index, minlength=100).astype(np.float32)
        count[count == 0] = np.float32(0.01)
        count = np.float32(count / np.float32(len(values)))
        bins.append(centers)
        counts.append(count)
        indices.append(index)
    joint = np.bincount(indices[0] * 100 + indices[1], minlength=10_000).reshape(100, 100).astype(np.float32)
    joint[joint == 0] = np.float32(0.1)
    norm = np.float32(0)
    for row in joint:
        for value in row:
            norm = np.float32(norm + value)
    joint = np.float32(joint / norm)
    return {"k1_bins": bins[0], "k2_bins": bins[1],
            "k1_counts": counts[0], "k2_counts": counts[1],
            "k1_mean": _source_mean(bins[0], counts[0], widths[0]),
            "k2_mean": _source_mean(bins[1], counts[1], widths[1]),
            "k1_start": starts[0], "k2_start": starts[1],
            "k1_width": widths[0], "k2_width": widths[1],
            "joint": joint, "joint_norm": norm}


def joint_curvature_log_likelihood(k1: np.ndarray, k2: np.ndarray,
                                   histogram: dict[str, object]) -> float:
    """Mean log joint-PDF value used by first-candidate ``qcurv_ll``."""
    starts = np.float32(histogram["k1_start"]), np.float32(histogram["k2_start"])
    widths = np.float32(histogram["k1_width"]), np.float32(histogram["k2_width"])
    indices = [np.clip(np.trunc(np.float32(np.float32(values) - start) / width).astype(np.int32), 0, 99)
               for values, start, width in zip((k1, k2), starts, widths)]
    probabilities = histogram["joint"][indices[0], indices[1]]
    return float(sum(np.log(float(value)) for value in probabilities) / len(probabilities))
