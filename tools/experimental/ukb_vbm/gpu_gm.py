#!/usr/bin/env python3
"""Make a gray-matter probability map from T1 with PyTorch WMH-SynthSeg.

This substitutes SynthSeg tissue probabilities for FAST's gray-matter PVE.
It is an experimental input to the UKB VBM registration step, not a strict
reproduction of UK Biobank's FAST-based pipeline.
"""

import argparse
from pathlib import Path
import tempfile

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch

from freesurfer_torch.wmh_synthseg import LABEL_IDS, WMHSynthSeg
from freesurfer_torch.wmh_synthseg.spatial import align_volume_to_ref, myzoom_torch


# Cerebral/cerebellar cortex and deep gray matter on both sides. Brainstem is
# excluded because its SynthSeg class mixes gray and white matter.
GM_LABELS = (3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52,
             17, 53, 18, 54, 26, 58, 28, 60)
GM_CHANNELS = tuple(LABEL_IDS.index(label) for label in GM_LABELS)
FLIPPED_CHANNELS = tuple(range(7)) + tuple(range(20, 33)) + tuple(range(7, 20))


def _validate_t1(data):
    if data.ndim != 3:
        raise ValueError("Input must be a single 3D T1 image")
    if not np.isfinite(data).all():
        raise ValueError("T1 image contains NaN or infinity")


class SynthSegGM:
    """Keep one checkpoint on the selected GPU across multiple T1 volumes."""

    def __init__(self, weights=None, device="cuda:0", threads=None, crop=True):
        self.segmenter = WMHSynthSeg(weights=weights, device=device, threads=threads)
        self.crop = crop

    @torch.inference_mode()
    def predict(self, image):
        """Return GM probability, brain mask, and the native 1 mm affine."""
        data = image.get_fdata()
        _validate_t1(data)
        device = self.segmenter.device
        volume = torch.tensor(data.astype(float), device=device)
        volume, affine = align_volume_to_ref(
            volume, image.affine, aff_ref=np.eye(4), return_aff=True, n_dims=3)
        maximum = torch.max(volume)
        if maximum <= 0:
            raise ValueError("T1 image must contain positive intensities")
        volume = volume / maximum
        voxel_size = np.linalg.norm(affine[:3, :3], axis=0)
        volume = myzoom_torch(volume, voxel_size, device=device)
        affine_1mm = affine.copy()
        affine_1mm[:3, :3] /= voxel_size[np.newaxis, :]
        affine_1mm[:3, 3] -= affine_1mm[:3, :3] @ (0.5 * (voxel_size - 1))
        if self.crop:
            volume, affine_1mm = self.segmenter._crop(volume, affine_1mm)

        shape = tuple(volume.shape)
        padded_shape = tuple((int(s) + 31) // 32 * 32 for s in shape)
        padded = torch.zeros(padded_shape, dtype=volume.dtype, device=device)
        spatial_slices = tuple(slice(0, s) for s in shape)
        padded[spatial_slices] = volume
        model = self.segmenter.model
        first = model(padded[None, None])[0, :33]
        first = first[(slice(None),) + spatial_slices]
        probabilities = 0.5 * torch.softmax(first, dim=0)
        del first
        second = model(torch.flip(padded, [0])[None, None])[0, :33]
        second = torch.flip(second, [1])[(slice(None),) + spatial_slices]
        probabilities += 0.5 * torch.softmax(second[list(FLIPPED_CHANNELS)], dim=0)

        gm = probabilities[list(GM_CHANNELS)].sum(dim=0).float().cpu().numpy()
        hard_labels = self.segmenter.labels[probabilities.argmax(dim=0)]
        brain = ((hard_labels != 0) & (hard_labels != 24)).to(torch.uint8).cpu().numpy()
        return gm, brain, affine_1mm

    def run(self, source, gm_output, *, brain_output=None, grid="input"):
        """Read one T1 and save GM probability on its input or native 1 mm grid."""
        image = nib.load(str(source))
        gm, brain, affine_1mm = self.predict(image)
        if grid == "input":
            target = (image.shape, image.affine)
            gm = np.asarray(resample_from_to(
                nib.Nifti1Image(gm, affine_1mm), target, order=1).dataobj,
                dtype=np.float32)
            brain = np.asarray(resample_from_to(
                nib.Nifti1Image(brain, affine_1mm), target, order=0).dataobj,
                dtype=np.uint8)
            output_affine = image.affine
        elif grid == "native-1mm":
            output_affine = affine_1mm
        else:
            raise ValueError("grid must be 'input' or 'native-1mm'")
        _save(gm_output, gm, output_affine, image if grid == "input" else None)
        if brain_output is not None:
            _save(brain_output, brain, output_affine, image if grid == "input" else None)


def _save(path, data, affine, source=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = nib.Nifti1Image(data, affine)
    if source is not None:
        sform_code = int(source.header["sform_code"]) or 1
        result.set_sform(affine, code=sform_code)
        qform, qform_code = source.get_qform(coded=True)
        if qform_code:
            result.set_qform(qform, code=qform_code)
    nib.save(result, str(path))


def _self_test():
    """Check NIfTI round-trip and 1 mm to anisotropic-grid resampling."""
    _validate_t1(np.ones((2, 2, 2), dtype=np.float32))
    for invalid in (np.full((2, 2, 2), np.nan), np.full((2, 2, 2), np.inf)):
        try:
            _validate_t1(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite T1 data must be rejected")
    affine = np.array([[0, -2, 0, 20], [1.5, 0, 0, -10],
                       [0, 0, 2, 5], [0, 0, 0, 1]], dtype=float)
    shape = (8, 9, 10)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gm.nii.gz"
        data = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
        _save(path, data, affine)
        loaded = nib.load(str(path))
        assert loaded.shape == shape
        np.testing.assert_allclose(loaded.affine, affine)
        np.testing.assert_allclose(loaded.get_fdata(), data, atol=1e-6)
        voxel_size = np.linalg.norm(affine[:3, :3], axis=0)
        native_affine = affine.copy()
        native_affine[:3, :3] /= voxel_size[np.newaxis, :]
        native_affine[:3, 3] -= native_affine[:3, :3] @ (0.5 * (voxel_size - 1))
        native_shape = tuple(np.round(np.array(shape) * voxel_size).astype(int))
        native_coords = np.indices(native_shape).reshape(3, -1)
        world = native_affine[:3, :3] @ native_coords + native_affine[:3, 3:4]
        native_values = (world[0] / 100 + world[1] / 200 + world[2] / 300)
        native = nib.Nifti1Image(native_values.reshape(native_shape), native_affine)
        restored = resample_from_to(native, (shape, affine), order=1)
        input_coords = np.indices(shape).reshape(3, -1)
        expected_world = affine[:3, :3] @ input_coords + affine[:3, 3:4]
        expected = (expected_world[0] / 100 + expected_world[1] / 200
                    + expected_world[2] / 300).reshape(shape)
        np.testing.assert_allclose(restored.get_fdata(), expected, atol=1e-6)
        starts = np.array((2, 3, 4))
        crop_shape = np.array((8, 10, 12))
        slices = tuple(slice(int(a), int(a + n)) for a, n in zip(starts, crop_shape))
        cropped_affine = native_affine.copy()
        cropped_affine[:3, 3] += native_affine[:3, :3] @ starts
        cropped = nib.Nifti1Image(native_values.reshape(native_shape)[slices],
                                  cropped_affine)
        restored_crop = resample_from_to(cropped, (shape, affine), order=1)
        crop_voxels = np.linalg.inv(cropped_affine)[:3, :3] @ expected_world
        crop_voxels += np.linalg.inv(cropped_affine)[:3, 3:4]
        inside = np.all((crop_voxels >= -1e-6)
                        & (crop_voxels <= crop_shape[:, None] - 1 + 1e-6), axis=0)
        expected_crop = np.where(inside.reshape(shape), expected, 0)
        np.testing.assert_allclose(restored_crop.get_fdata(), expected_crop, atol=1e-6)
    print("NIfTI geometry and read/write check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="3D T1 NIfTI files")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--weights", type=Path, help="WMH-SynthSeg checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--grid", choices=("input", "native-1mm"), default="input",
                        help="input: exact T1 grid; native-1mm: model's 1 mm grid")
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True,
                        help="locate and process a 192x224x192 brain region (default: on)")
    parser.add_argument("--brain-mask", action="store_true",
                        help="also save a hard intracranial mask")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.inputs or args.out_dir is None:
        parser.error("provide T1 inputs and --out-dir")
    worker = SynthSegGM(weights=args.weights, device=args.device,
                        threads=args.threads, crop=args.crop)
    for index, source in enumerate(args.inputs):
        case_id = f"case_{index:04d}"
        subject_dir = args.out_dir / case_id
        gm_path = subject_dir / "gm_prob.nii.gz"
        brain_path = subject_dir / "brain_mask.nii.gz" if args.brain_mask else None
        worker.run(source, gm_path, brain_output=brain_path, grid=args.grid)
        print(f"{case_id} complete")


if __name__ == "__main__":
    main()
