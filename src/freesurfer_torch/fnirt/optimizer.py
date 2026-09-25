"""Matrix-free linear algebra for FNIRT's Gauss-Newton/LM updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PCGReport:
    iterations: int
    converged: bool
    relative_residual: float


def preconditioned_conjugate_gradient(
    matvec,
    rhs: torch.Tensor,
    *,
    diagonal: torch.Tensor | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 500,
):
    """Solve a symmetric positive-definite system using matrix-free PCG.

    FSL configures the FNIRT equation solver with a relative tolerance of
    ``1e-3`` and at most 500 iterations.  The stopping rule here uses the
    unpreconditioned residual norm relative to the right-hand-side norm.  This
    is the stopping rule in the IML++ ``CG`` routine used by FSL 6.0.7.4.
    """
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid PCG stopping parameters")
    if rhs.ndim != 1:
        raise ValueError("rhs must be a vector")
    if diagonal is None:
        inverse_diagonal = torch.ones_like(rhs)
    else:
        if diagonal.shape != rhs.shape:
            raise ValueError("diagonal and rhs must have the same shape")
        floor = torch.finfo(rhs.dtype).eps * diagonal.abs().mean().clamp_min(1)
        inverse_diagonal = diagonal.clamp_min(floor).reciprocal()

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = torch.linalg.vector_norm(rhs)
    if float(rhs_norm) == 0:
        return solution, PCGReport(0, True, 0.0)
    preconditioned = inverse_diagonal * residual
    direction = preconditioned.clone()
    rz = torch.dot(residual, preconditioned)

    relative = 1.0
    for iteration in range(1, max_iterations + 1):
        product = matvec(direction)
        denominator = torch.dot(direction, product)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            return solution, PCGReport(iteration - 1, False, relative)
        alpha = rz / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * product
        relative = float(torch.linalg.vector_norm(residual) / rhs_norm)
        if relative <= tolerance:
            return solution, PCGReport(iteration, True, relative)
        preconditioned = inverse_diagonal * residual
        new_rz = torch.dot(residual, preconditioned)
        beta = new_rz / rz.clamp_min(torch.finfo(rhs.dtype).tiny)
        direction = preconditioned + beta * direction
        rz = new_rz
    return solution, PCGReport(max_iterations, False, relative)


__all__ = ["PCGReport", "preconditioned_conjugate_gradient"]
