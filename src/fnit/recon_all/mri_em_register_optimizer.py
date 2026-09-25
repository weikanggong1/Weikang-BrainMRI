"""Isolated FreeSurfer 8.2 first-pass EM line-search candidate.

The complete mri_em_register stage and its LTA output are not validated yet.
"""

from dataclasses import dataclass

import numpy as np

from .mri_em_register_em_candidate import EMObjective


@dataclass
class _Step:
    x: float = 0.
    fx: float = 0.
    dx: float = 0.
    y: float = 0.
    fy: float = 0.
    dy: float = 0.
    p: float = 0.
    bracketed: bool = False


def _mcstep(state: _Step, value: float, derivative: float,
            lower: float, upper: float) -> int:
    """More-Thuente's safeguarded trial step for the VXL L-BFGS search."""

    x, fx, dx = state.x, state.fx, state.dx
    y, fy, dy, p = state.y, state.fy, state.dy, state.p
    if ((state.bracketed and (p <= min(x, y) or p >= max(x, y))) or
            dx * (p - x) >= 0 or upper < lower):
        return 0
    sign = derivative * (dx / abs(dx))
    if value > fx:
        case, bound = 1, True
        theta = 3 * (fx - value) / (p - x) + dx + derivative
        scale = max(abs(theta), abs(dx), abs(derivative))
        gamma = scale * np.sqrt((theta / scale) ** 2 -
                                dx / scale * (derivative / scale))
        if p < x:
            gamma = -gamma
        q = gamma - dx + gamma + derivative
        cubic = x + (gamma - dx + theta) / q * (p - x)
        quadratic = x + dx / ((fx - value) / (p - x) + dx) / 2 * (p - x)
        next_step = (cubic if abs(cubic - x) < abs(quadratic - x)
                     else cubic + (quadratic - cubic) / 2)
        state.bracketed = True
    elif sign < 0:
        case, bound = 2, False
        theta = 3 * (fx - value) / (p - x) + dx + derivative
        scale = max(abs(theta), abs(dx), abs(derivative))
        gamma = scale * np.sqrt((theta / scale) ** 2 -
                                dx / scale * (derivative / scale))
        if p > x:
            gamma = -gamma
        cubic = p + (gamma - derivative + theta) / (
            gamma - derivative + gamma + dx) * (x - p)
        quadratic = p + derivative / (derivative - dx) * (x - p)
        next_step = cubic if abs(cubic - p) > abs(quadratic - p) else quadratic
        state.bracketed = True
    elif abs(derivative) < abs(dx):
        case, bound = 3, True
        theta = 3 * (fx - value) / (p - x) + dx + derivative
        scale = max(abs(theta), abs(dx), abs(derivative))
        gamma = scale * np.sqrt(max(0., (theta / scale) ** 2 -
                                    dx / scale * (derivative / scale)))
        if p > x:
            gamma = -gamma
        ratio = (gamma - derivative + theta) / (gamma + dx - derivative + gamma)
        cubic = p + ratio * (x - p) if ratio < 0 and gamma != 0 else (
            upper if p > x else lower)
        quadratic = p + derivative / (derivative - dx) * (x - p)
        if state.bracketed:
            next_step = cubic if abs(p - cubic) < abs(p - quadratic) else quadratic
        else:
            next_step = cubic if abs(p - cubic) > abs(p - quadratic) else quadratic
    else:
        case, bound = 4, False
        if state.bracketed:
            theta = 3 * (value - fy) / (y - p) + dy + derivative
            scale = max(abs(theta), abs(dy), abs(derivative))
            gamma = scale * np.sqrt((theta / scale) ** 2 -
                                    dy / scale * (derivative / scale))
            if p > y:
                gamma = -gamma
            next_step = p + (gamma - derivative + theta) / (
                gamma - derivative + gamma + dy) * (y - p)
        else:
            next_step = upper if p > x else lower

    if value > fx:
        state.y, state.fy, state.dy = p, value, derivative
    else:
        if sign < 0:
            state.y, state.fy, state.dy = x, fx, dx
        state.x, state.fx, state.dx = p, value, derivative
    next_step = min(upper, max(lower, next_step))
    if state.bracketed and bound:
        if state.y > state.x:
            next_step = min(state.x + .66 * (state.y - state.x), next_step)
        else:
            next_step = max(state.x + .66 * (state.y - state.x), next_step)
    state.p = next_step
    return case


def first_em_line_search(objective: EMObjective, initial: np.ndarray,
                         ) -> tuple[np.ndarray, float, list[tuple[float, float]]]:
    """One VXL L-BFGS first-direction search, retaining its best trial."""

    origin = np.asarray(initial[:3], np.float32).ravel().astype(np.float64)

    def matrix_at(parameters: np.ndarray) -> np.ndarray:
        return np.vstack((parameters.astype(np.float32).reshape(3, 4),
                          [0, 0, 0, 1])).astype(np.float32)

    def evaluate(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        matrix = matrix_at(parameters)
        cost = float(np.float32(objective.cost(matrix)))
        gradient = objective.gradient(matrix).astype(np.float32).ravel().astype(np.float64)
        return cost, gradient

    first_cost, first_gradient = evaluate(origin)
    direction = -first_gradient
    initial_derivative = float(np.dot(first_gradient, direction))
    step = _Step(fx=first_cost, dx=initial_derivative,
                 fy=first_cost, dy=initial_derivative,
                 p=1 / np.linalg.norm(first_gradient))
    width = 1e20 - 1e-20
    previous_width = width / .5
    stage_one = True
    best = (first_cost, origin.copy())
    history = []
    for iteration in range(20):
        if step.bracketed:
            lower, upper = min(step.x, step.y), max(step.x, step.y)
        else:
            lower, upper = step.x, step.p + 4 * (step.p - step.x)
        step.p = min(1e20, max(1e-20, step.p))
        if ((step.bracketed and (step.p <= lower or step.p >= upper)) or
                iteration >= 19 or
                (step.bracketed and upper - lower <= 1e-16 * upper)):
            step.p = step.x
        parameters = origin + step.p * direction
        value, gradient = evaluate(parameters)
        derivative = float(np.dot(gradient, direction))
        history.append((step.p, value))
        if value < best[0]:
            best = (value, parameters.copy())
        test_value = first_cost + step.p * 1e-4 * initial_derivative
        if value <= test_value and abs(derivative) <= .1 * (-initial_derivative):
            break
        if iteration == 19:
            break
        if stage_one and value <= test_value and derivative >= min(1e-4, .1) * initial_derivative:
            stage_one = False
        if stage_one and value <= step.fx and value > test_value:
            derivative_test = 1e-4 * initial_derivative
            modified = value - step.p * derivative_test
            step.fx -= step.x * derivative_test
            step.fy -= step.y * derivative_test
            step.dx -= derivative_test
            step.dy -= derivative_test
            _mcstep(step, modified, derivative - derivative_test, lower, upper)
            step.fx += step.x * derivative_test
            step.fy += step.y * derivative_test
            step.dx += derivative_test
            step.dy += derivative_test
        else:
            _mcstep(step, value, derivative, lower, upper)
        if step.bracketed:
            if abs(step.y - step.x) >= .66 * previous_width:
                step.p = step.x + .5 * (step.y - step.x)
            previous_width, width = width, abs(step.y - step.x)
    return matrix_at(best[1]), best[0], history
