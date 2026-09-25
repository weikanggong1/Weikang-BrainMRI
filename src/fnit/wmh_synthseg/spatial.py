"""WMH-SynthSeg orientation and interpolation helpers.

Adapted from FreeSurfer 8.2.0-1 ``WMHSynthSeg/utils.py`` under the FreeSurfer
Software License. See ``THIRD_PARTY_NOTICES.md`` and ``licenses/FreeSurfer.txt``.
"""

import numpy as np
import torch


def get_ras_axes(aff, n_dims=3):
    """Return image axes corresponding to the reference RAS dimensions."""
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[:n_dims, :n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i
    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=3):
    """Reorder and flip a tensor to the orientation specified by ``aff_ref``."""
    aff_flo = aff.copy()
    if aff_ref is None:
        aff_ref = np.eye(4)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            volume = torch.swapaxes(volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])[0][0]
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            volume = torch.flip(volume, [i])
            aff_flo[:, i] = -aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (volume.shape[i] - 1)
    return (volume, aff_flo) if return_aff else volume


def myzoom_torch(X, factor, device, aff=None):
    """Trilinear 3D/4D zoom with the reference half-voxel grid and clamped edges."""
    if len(X.shape) == 3:
        X = X[..., None]
    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0],
                      dtype=torch.float, device=device)[:newsize[0]]
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1],
                      dtype=torch.float, device=device)[:newsize[1]]
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2],
                      dtype=torch.float, device=device)[:newsize[2]]
    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > X.shape[0] - 1] = X.shape[0] - 1
    vy[vy > X.shape[1] - 1] = X.shape[1] - 1
    vz[vz > X.shape[2] - 1] = X.shape[2] - 1

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > X.shape[0] - 1] = X.shape[0] - 1
    wcx = vx - fx
    wfx = 1 - wcx
    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > X.shape[1] - 1] = X.shape[1] - 1
    wcy = vy - fy
    wfy = 1 - wcy
    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > X.shape[2] - 1] = X.shape[2] - 1
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]],
                    dtype=torch.float, device=device)
    for channel in range(X.shape[3]):
        Xc = X[:, :, :, channel]
        tmp1 = torch.zeros([newsize[0], Xc.shape[1], Xc.shape[2]],
                           dtype=torch.float, device=device)
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] + wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros([newsize[0], newsize[1], Xc.shape[2]],
                           dtype=torch.float, device=device)
        for j in range(newsize[1]):
            tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] + wcy[j] * tmp1[:, cy[j], :]
        for k in range(newsize[2]):
            Y[:, :, k, channel] = wfz[k] * tmp2[:, :, fz[k]] + wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:, :, :, 0]
    if aff is None:
        return Y
    aff_new = aff.copy()
    for c in range(3):
        aff_new[:-1, c] = aff_new[:-1, c] / factor
    aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (0.5 - 0.5 / (factor * np.ones(3)))
    return Y, aff_new
