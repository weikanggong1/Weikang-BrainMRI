"""End-to-end GPU FAST VBM pipeline."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import surfa as sf
import torch

from ..fast import FASTResult, TorchFAST
from .registration import VBMRegistrationResult, register_gm
from .synthmorph_backend import SynthMorphDeformRegistration


OUTPUT_FILENAMES = {
    "brain": "T1_brain.nii.gz",
    "brain_mask": "brain_mask.nii.gz",
    "pve_csf": "T1_brain_pve_0.nii.gz",
    "pve_gm": "T1_brain_pve_1.nii.gz",
    "pve_wm": "T1_brain_pve_2.nii.gz",
    "hard_segmentation": "T1_brain_seg.nii.gz",
    "pve_segmentation": "T1_brain_pveseg.nii.gz",
    "mixel_type": "T1_brain_mixeltype.nii.gz",
    "bias_field": "T1_brain_bias.nii.gz",
    "restored": "T1_brain_restore.nii.gz",
    "warped_gm": "T1_GM_to_template_GM.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
    "modulated_gm": "T1_GM_to_template_GM_mod.nii.gz",
}


def _load_volume(value, name):
    if isinstance(value, (str, os.PathLike)):
        value = sf.load_volume(str(value))
    if not isinstance(value, sf.Volume):
        raise TypeError(f"{name} must be a path or surfa.Volume")
    data = np.asarray(value.data)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
        value = value.new(data)
    if data.ndim != 3:
        raise ValueError(f"{name} must contain one 3D frame")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value, data


def _same_grid(image, mask):
    return image.shape[:3] == mask.shape[:3] and np.allclose(
        image.geom.vox2world.matrix,
        mask.geom.vox2world.matrix,
        atol=1e-5,
        rtol=0,
    )


def _validate_geometry(volume, name):
    affine = np.asarray(volume.geom.vox2world.matrix, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{name} affine must be a finite 4x4 matrix")
    linear = affine[:3, :3]
    voxel_size = np.linalg.norm(linear, axis=0)
    if np.any(voxel_size <= 0) or abs(np.linalg.det(linear)) < 1e-8:
        raise ValueError(f"{name} affine must be invertible with positive voxel sizes")


@dataclass
class FastVBMResult:
    """Brain extraction, FAST tissue maps, and template-space VBM outputs."""

    brain: sf.Volume
    brain_mask: sf.Volume
    fast: FASTResult
    registration: VBMRegistrationResult
    settings: dict
    timing_sec: dict

    @property
    def pve_gm(self):
        return self.fast.pve_gm

    @property
    def warped_gm(self):
        return self.registration.warped_gm

    @property
    def jacobian(self):
        return self.registration.jacobian

    @property
    def modulated_gm(self):
        return self.registration.modulated_gm

    def volumes(self):
        """Return every image output keyed by its stable API name."""
        return {
            "brain": self.brain,
            "brain_mask": self.brain_mask,
            "pve_csf": self.fast.pve_csf,
            "pve_gm": self.fast.pve_gm,
            "pve_wm": self.fast.pve_wm,
            "hard_segmentation": self.fast.hard_segmentation,
            "pve_segmentation": self.fast.pve_segmentation,
            "mixel_type": self.fast.mixel_type,
            "bias_field": self.fast.bias_field,
            "restored": self.fast.restored,
            "warped_gm": self.registration.warped_gm,
            "jacobian": self.registration.jacobian,
            "modulated_gm": self.registration.modulated_gm,
        }

    def report(self):
        """Return JSON-serializable settings, timing, and registration QC."""
        mask = np.asarray(self.brain_mask.data) > 0
        bias = np.asarray(self.fast.bias_field.data)
        return {
            "status": "experimental",
            "method": (
                "TorchFAST GM; independent PyTorch FLIRT-compatible 12-DOF "
                "affine; PyTorch SynthMorph deform"
            ),
            "fnirt_equivalent": False,
            "fsl_flirt_equivalent": False,
            "settings": self.settings,
            "timing_sec": self.timing_sec,
            "timing_definition": {
                "total": (
                    "API call entry through CPU output materialization; includes "
                    "input reads and the first call's lazy SynthStrip load; excludes "
                    "FASTVBMResult.save and output NIfTI writes"
                ),
                "brain_extraction": (
                    "brain extraction only; the first call includes lazy SynthStrip "
                    "weight loading, later calls reuse the loaded model"
                ),
                "fast": "TorchFAST inference and GPU-to-CPU output materialization",
                "registration_jacobian_modulation": (
                    "linear registration, SynthMorph deform inference, "
                    "Jacobian, modulation, and CPU output materialization; "
                    "the first call includes lazy SynthMorph weight loading"
                ),
            },
            "fast": {
                "tissue_means": list(self.fast.tissue_means),
                "tissue_variances": list(self.fast.tissue_variances),
                "bias_range_inside_mask": [
                    float(bias[mask].min()),
                    float(bias[mask].max()),
                ],
            },
            "registration": {
                "fit_score_normalized_correlation": self.registration.fit_score,
                "pull_world_affine": self.registration.pull_world_affine.tolist(),
                "maximum_displacement_mm": (
                    self.registration.maximum_displacement_mm
                ),
                "jacobian_convention": (
                    "nonlinear-only: full pull determinant divided by affine "
                    "determinant"
                ),
                **self.registration.qc,
            },
            "outputs": OUTPUT_FILENAMES,
        }

    def save(self, output_dir, *, overwrite=False, report=True):
        """Save all NIfTI outputs atomically per file and return their paths."""
        output_dir = Path(output_dir)
        paths = {
            name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()
        }
        report_path = output_dir / "fast_vbm_report.json"
        candidates = [*paths.values(), report_path]
        existing = [path for path in candidates if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"output exists: {existing[0]}; pass overwrite=True to replace it"
            )
        if overwrite and report_path.exists():
            report_path.unlink()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name or 'fast-vbm'}.tmp-", dir=output_dir.parent
        ) as directory:
            stage = Path(directory)
            for name, volume in self.volumes().items():
                volume.save(stage / OUTPUT_FILENAMES[name])
            if report:
                (stage / report_path.name).write_text(
                    json.dumps(self.report(), indent=2, allow_nan=False) + "\n"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, path in paths.items():
                os.replace(stage / OUTPUT_FILENAMES[name], path)
            if report:
                os.replace(stage / report_path.name, report_path)
        return {name: str(path) for name, path in paths.items()}


class FastVBM:
    """Reusable raw-T1 to VBM pipeline using SynthStrip and TorchFAST.

    SynthStrip is loaded lazily and is not required when ``brain_mask`` is
    supplied. Linear registration is a PyTorch FLIRT-compatible 12-DOF fit.
    Nonlinear registration uses this package's PyTorch SynthMorph implementation
    with the official deform checkpoint.
    """

    def __init__(
        self,
        *,
        device="cpu",
        threads=None,
        synthstrip_weights=None,
        synthmorph_weights=None,
        bias_correction=True,
        linear_strides=(4, 2, 1),
        linear_steps=(80, 60, 50),
        linear_learning_rates=(0.05, 0.025, 0.0125),
        synthmorph_extent=256,
        synthmorph_hyper=0.5,
        synthmorph_steps=7,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if threads is not None:
            if not isinstance(threads, int) or threads < 1:
                raise ValueError("threads must be a positive integer")
            torch.set_num_threads(threads)
        self.synthstrip_weights = synthstrip_weights
        self.synthmorph_weights = synthmorph_weights
        self.threads = threads
        self.extractor = None
        self.deform_model = None
        self.fast = TorchFAST(
            device=self.device,
            threads=threads,
            bias_fwhm_mm=20.0 if bias_correction else 0.0,
        )
        self.bias_correction = bias_correction
        self.linear_strides = tuple(linear_strides)
        self.linear_steps = tuple(linear_steps)
        self.linear_learning_rates = tuple(linear_learning_rates)
        self.synthmorph_extent = synthmorph_extent
        self.synthmorph_hyper = synthmorph_hyper
        self.synthmorph_steps = synthmorph_steps

    def _extract(self, image):
        if self.extractor is None:
            from ..synthstrip import SynthStrip

            self.extractor = SynthStrip(
                weights=self.synthstrip_weights,
                device=self.device,
                threads=self.threads,
            )
        result = self.extractor(image)
        return result.image, result.mask

    def _deform_model(self):
        if self.deform_model is None:
            self.deform_model = SynthMorphDeformRegistration(
                weights=self.synthmorph_weights,
                device=self.device,
                extent=self.synthmorph_extent,
                hyper=self.synthmorph_hyper,
                steps=self.synthmorph_steps,
            )
        return self.deform_model

    def __call__(
        self,
        image,
        template,
        *,
        brain_mask=None,
        initial_pull=None,
        initial_pull_convention=None,
    ):
        """Run one T1 image; all returned template maps use ``template``'s grid."""
        total_started = time.perf_counter()
        image, image_data = _load_volume(image, "image")
        template, template_data = _load_volume(template, "template")
        _validate_geometry(image, "image")
        _validate_geometry(template, "template")
        if not np.any(template_data > 0):
            raise ValueError("template must contain positive GM values")

        extraction_started = time.perf_counter()
        if brain_mask is None:
            brain, mask = self._extract(image)
            mask_source = "synthstrip"
        else:
            mask, mask_data = _load_volume(brain_mask, "brain_mask")
            if not _same_grid(image, mask):
                raise ValueError("brain_mask must use the same shape and geometry as image")
            binary = np.asarray(mask_data > 0, dtype=np.uint8)
            if not np.any(binary):
                raise ValueError("brain_mask is empty")
            mask = image.new(binary)
            brain = image.copy()
            brain[binary == 0] = min(float(image_data.min()), 0.0)
            mask_source = "explicit"
        mask_data = np.asarray(mask.data)
        if not np.isfinite(mask_data).all() or not np.any(mask_data > 0):
            raise ValueError("brain mask must be finite and nonempty")
        if not _same_grid(image, mask):
            raise ValueError("brain mask must use the same shape and geometry as image")
        extraction_sec = time.perf_counter() - extraction_started

        fast_started = time.perf_counter()
        fast_result = self.fast(brain, mask=mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        fast_sec = time.perf_counter() - fast_started

        registration_started = time.perf_counter()
        registration = register_gm(
            fast_result.pve_gm,
            template,
            device=self.device,
            initial_pull=initial_pull,
            initial_pull_convention=initial_pull_convention,
            linear_strides=self.linear_strides,
            linear_steps=self.linear_steps,
            linear_learning_rates=self.linear_learning_rates,
            synthmorph_weights=self.synthmorph_weights,
            synthmorph_extent=self.synthmorph_extent,
            synthmorph_hyper=self.synthmorph_hyper,
            synthmorph_steps=self.synthmorph_steps,
            deform_model=self._deform_model(),
        )
        registration_sec = time.perf_counter() - registration_started
        timing = {
            "brain_extraction": extraction_sec,
            "fast": fast_sec,
            "registration_jacobian_modulation": registration_sec,
            "total": time.perf_counter() - total_started,
        }
        settings = {
            "device": str(self.device),
            "mask_source": mask_source,
            "bias_correction": self.bias_correction,
            "linear_backend": "independent-pytorch-flirt-compatible",
            "linear_cost": "normalized correlation",
            "linear_optimizer": "Adam",
            "linear_forward_convention": "moving-to-fixed-world-ras",
            "linear_pull_convention": "fixed-to-moving-world-ras",
            "linear_strides": list(self.linear_strides),
            "linear_steps": list(self.linear_steps),
            "linear_learning_rates": list(self.linear_learning_rates),
            "nonlinear_backend": "pytorch-synthmorph-deform",
            "synthmorph_implementation": (
                "freesurfer_torch.synthmorph.SynthMorph"
            ),
            "synthmorph_extent": self.synthmorph_extent,
            "synthmorph_hyper": self.synthmorph_hyper,
            "synthmorph_integration_steps": self.synthmorph_steps,
            "synthmorph_mid_space": False,
            "synthmorph_warp_convention": "fixed-grid-target-to-source-disp-ras",
            "torch_threads": torch.get_num_threads(),
            "fast": asdict(self.fast.config),
        }
        return FastVBMResult(
            brain=brain,
            brain_mask=mask,
            fast=fast_result,
            registration=registration,
            settings=settings,
            timing_sec=timing,
        )

    def run(
        self,
        image,
        template,
        output_dir,
        *,
        brain_mask=None,
        initial_pull=None,
        initial_pull_convention=None,
        overwrite=False,
    ):
        """Run one T1 and save all outputs under ``output_dir``."""
        result = self(
            image,
            template,
            brain_mask=brain_mask,
            initial_pull=initial_pull,
            initial_pull_convention=initial_pull_convention,
        )
        result.save(output_dir, overwrite=overwrite)
        return result


FASTVBMResult = FastVBMResult


__all__ = ["FastVBMResult", "FASTVBMResult", "FastVBM", "OUTPUT_FILENAMES"]
