"""Exact source-order CPU averaging for spherical registration gradients."""

import numpy as np
import torch
from numba import njit


@njit
def _average_numpy(gradient, neighbors, degrees, reciprocals, iterations):
    current = gradient.copy()
    following = np.empty_like(current)
    for _ in range(iterations):
        for vertex in range(len(current)):
            degree = degrees[vertex]
            for axis in range(3):
                value = current[vertex, axis]
                for index in range(degree):
                    value = np.float32(value + current[neighbors[vertex, index], axis])
                following[vertex, axis] = np.float32(value * reciprocals[vertex])
        current, following = following, current
    return current


@torch.no_grad()
def average_gradients_exact_cpu(gradient: torch.Tensor,
                                neighbors: torch.Tensor, degrees: torch.Tensor,
                                iterations: int) -> torch.Tensor:
    """Match MRISaverageGradients float32 arithmetic with an ordered CPU loop."""
    if gradient.device.type != "cpu":
        raise ValueError("source-order averaging requires CPU tensors")
    reciprocals = (1.0 / (degrees + 1).float()).numpy()
    result = _average_numpy(gradient.numpy(), neighbors.numpy(), degrees.numpy(),
                            reciprocals, iterations)
    return torch.from_numpy(result)
