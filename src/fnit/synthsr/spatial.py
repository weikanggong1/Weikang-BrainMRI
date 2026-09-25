"""Spatial operations used by the FreeSurfer SynthSR inference pipeline.

The sampling grid and affine updates follow FreeSurfer's ``mri_synthsr``
script. See ``licenses/FreeSurfer.txt`` for its license.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def resample_volume(volume, aff, new_vox_size=(1.0, 1.0, 1.0), interpolation="linear", blur=True):
    """Resample a 3D volume while retaining its field of view in RAS space."""
    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    factor = pixdim / np.asarray(new_vox_size)
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0
    volume_filt = gaussian_filter(volume, sigmas) if blur else volume

    axes = [np.arange(size) for size in volume_filt.shape]
    sampler = RegularGridInterpolator(axes, volume_filt, method=interpolation)
    start = -(factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(np.asarray(volume_filt.shape) * factor)
    coords = [np.arange(start=s, stop=e, step=d) for s, e, d in zip(start, stop, step)]
    for axis, coord in enumerate(coords):
        np.clip(coord, 0, volume_filt.shape[axis] - 1, out=coord)
    grid = np.meshgrid(*coords, indexing="ij", sparse=True)
    result = sampler(tuple(grid))

    out_aff = aff.copy()
    for axis in range(3):
        out_aff[:-1, axis] /= factor[axis]
    out_aff[:-1, -1] -= out_aff[:-1, :-1] @ (0.5 * (factor - 1))
    return result, out_aff


def get_ras_axes(aff, n_dims=3):
    """Map RAS axes to voxel axes using the reference affine convention."""
    aff_inverted = np.linalg.inv(aff)
    axes = np.argmax(np.abs(aff_inverted[:n_dims, :n_dims]), axis=0)
    for axis in range(n_dims):
        if axis not in axes:
            unique, counts = np.unique(axes, return_counts=True)
            duplicate = unique[np.argmax(counts)]
            axes[np.where(axes == duplicate)[0][-1]] = axis
    return axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=3):
    """Permute and flip voxel axes to match a reference orientation."""
    result = volume.copy()
    out_aff = aff.copy()
    if aff_ref is None:
        aff_ref = np.eye(4)
    axes_ref = get_ras_axes(aff_ref, n_dims)
    axes_flo = get_ras_axes(out_aff, n_dims)

    out_aff[:, axes_ref] = out_aff[:, axes_flo]
    for axis in range(n_dims):
        if axes_flo[axis] != axes_ref[axis]:
            result = np.swapaxes(result, axes_flo[axis], axes_ref[axis])
            other = np.where(axes_flo == axes_ref[axis])[0][0]
            axes_flo[other], axes_flo[axis] = axes_flo[axis], axes_flo[other]

    dots = np.sum(out_aff[:3, :3] * aff_ref[:3, :3], axis=0)
    for axis in range(n_dims):
        if dots[axis] < 0:
            result = np.flip(result, axis=axis)
            out_aff[:, axis] = -out_aff[:, axis]
            out_aff[:3, 3] -= out_aff[:3, axis] * (result.shape[axis] - 1)
    return (result, out_aff) if return_aff else result


def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    """Center a 3D volume in a larger array and record the inverse crop."""
    shape = np.asarray(volume.shape[:3])
    target = np.asarray(padding_shape, dtype=int)
    if target.size == 1:
        target = np.repeat(target, 3)
    before = np.maximum(np.floor((target - shape) / 2).astype(int), 0)
    after = np.maximum(np.ceil((target - shape) / 2).astype(int), 0)
    pad_idx = np.concatenate((before, before + shape))
    margins = tuple((int(lo), int(hi)) for lo, hi in zip(before, after))
    margins += ((0, 0),) * (volume.ndim - 3)
    result = np.pad(volume, margins, mode="constant", constant_values=padding_value)

    outputs = [result]
    if aff is not None:
        out_aff = aff.copy()
        out_aff[:-1, -1] -= out_aff[:-1, :-1] @ before
        outputs.append(out_aff)
    if return_pad_idx:
        outputs.append(pad_idx)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def crop_volume_with_idx(volume, crop_idx, aff=None, n_dims=3):
    """Undo padding or crop a 3D volume using lower and upper bounds."""
    lower = np.asarray(crop_idx[:n_dims], dtype=int)
    upper = np.asarray(crop_idx[n_dims : 2 * n_dims], dtype=int)
    result = volume[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))].copy()
    if aff is None:
        return result
    out_aff = aff.copy()
    out_aff[:3, 3] += out_aff[:3, :3] @ lower
    return result, out_aff
