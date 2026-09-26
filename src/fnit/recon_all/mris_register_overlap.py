"""FreeSurfer sphere-registration overlap smoothing in PyTorch."""

from __future__ import annotations

import torch

from .mris_register_kernels import project_sphere
from .mris_register_nonlinear import face_area_normals, ordered_neighbors_from_faces


@torch.no_grad()
def remove_overlap_sphere(positions: torch.Tensor, faces: torch.Tensor,
                          *, start_iteration: int = 0,
                          max_iterations: int = 1000) -> tuple[torch.Tensor, list[int]]:
    """Smooth vertices around negative sphere faces and return pre-step counts.

    The default registration uses a 0.99 step, expanding the marked region
    from negative-face corners to their one-ring when progress stalls.
    """
    xyz = positions.float().clone()

    def negative_faces() -> torch.Tensor:
        area, _ = face_area_normals(xyz, faces, signed_sphere=True)
        return area < 0

    negative = negative_faces()
    count = int(negative.sum())
    history: list[int] = []
    if count == 0:
        return xyz, history
    neighbors, degrees = ordered_neighbors_from_faces(faces, len(xyz))
    active = torch.arange(neighbors.shape[1], device=xyz.device)[None, :] < degrees[:, None]
    min_negative, min_iteration = count, 0
    iteration, last_expand, same, max_neighbors = start_iteration, 0, 0, 0
    dt = 0.99
    while count > 0:
        history.append(count)
        marked = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
        marked[faces[negative].reshape(-1)] = True
        if max_neighbors:
            marked |= (marked[neighbors] & active).any(dim=1)

        xyz_double = xyz.double()
        displacement = torch.zeros_like(xyz_double)
        for index in range(neighbors.shape[1]):
            difference = xyz_double[neighbors[:, index]] - xyz_double
            displacement = torch.where(active[:, index, None],
                                       displacement + difference, displacement)
        displacement = (displacement / degrees[:, None].double()).float()
        moved = (xyz_double + dt * displacement.double()).float()
        xyz = project_sphere(torch.where(marked[:, None], moved, xyz))

        old_count = count
        iteration += 1
        negative = negative_faces()
        count = int(negative.sum())
        if count < min_negative:
            min_negative, min_iteration = count, iteration
        elif (iteration - min_iteration) % 10 == 0 and iteration > min_iteration:
            if dt > 0.01:
                dt *= 0.95
        elif iteration > min_iteration + 50 and iteration > last_expand + 25:
            if max_neighbors < 1:
                dt, max_neighbors = 0.99, 1
            last_expand, same = iteration, 0
        if iteration > min_iteration + 1000:
            break
        if old_count == count:
            if same > 25 and max_neighbors < 1:
                max_neighbors, dt, last_expand, same = 1, 0.99, iteration, 0
            else:
                same += 1
        else:
            same = 0
        if iteration - start_iteration > max_iterations:
            break
    return xyz, history
