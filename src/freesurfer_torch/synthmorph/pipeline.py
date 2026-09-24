"""SynthMorph registration with PyTorch inference and native image geometry."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.linalg
import surfa as sf
import torch
from .spatial import compose, transform
from ..weights import resolve_weights
from .._batch_table import cases_from_table
from .._parallel_table import run_parallel


@dataclass
class RegistrationResult:
    moved: sf.Volume
    fixed_moved: sf.Volume
    transform: object
    inverse: object


def network_space(image, shape, center=None):
    new = sf.ImageGeometry(shape=shape, voxsize=1, rotation='LIA',
                           center=image.geom.center if center is None else center.geom.center,
                           shear=None)
    return ((image.geom.world2vox @ new.vox2world).matrix,
            (new.world2vox @ image.geom.vox2world).matrix)


def _load(image, single_frame=True):
    out = sf.load_volume(str(image)) if isinstance(image, (str, Path)) else image
    if not isinstance(out, sf.Volume) or len(out.shape) not in (3, 4):
        raise ValueError('input must be a 3D volume with optional frames')
    if single_frame and len(out.shape) != 3:
        raise ValueError('registration inputs must be single-frame 3D volumes')
    if not np.isfinite(out.data).all():
        raise ValueError('input contains NaN or infinity')
    return out


def _tensor(image, device):
    return torch.as_tensor(np.array(image.data, dtype=np.float32, copy=True), device=device)[None, None]


def _numpy(tensor):
    if tensor.ndim == 2:
        return tensor.detach().cpu().numpy()
    return tensor[0].permute(1, 2, 3, 0).detach().cpu().numpy()


class SynthMorph:
    """Reusable registration model; weights is a directory or named path mapping.

    model: joint (affine + deformable), deform, affine, or rigid.
    hyper: warp regularity in (0,1), fixed for this model instance.
    extent: 192 or 256 isotropic 1-mm network voxels per axis.
    Networks and coordinate composition run on device. Final image resampling
    uses Surfa on CPU to preserve the original command's boundary conventions.
    """
    def __init__(self, weights=None, device='cpu', model='joint', extent=256,
                 hyper=0.5, steps=7):
        from .models import SynthMorphNetwork
        if model not in ('joint', 'deform', 'affine', 'rigid'):
            raise ValueError('unknown registration model')
        if extent not in (192, 256):
            raise ValueError('extent must be 192 or 256')
        if not 0 < hyper < 1 or steps < 5:
            raise ValueError('hyper must be in (0,1) and steps must be >=5')
        names = {'affine': 'synthmorph.affine.2.h5', 'deform': 'synthmorph.deform.3.h5',
                 'rigid': 'synthmorph.rigid.1.h5'}
        needed = ('affine', 'deform') if model == 'joint' else (model,)
        paths = {key: resolve_weights(names[key], weights.get(key) if isinstance(weights, dict) else weights)
                 for key in needed}
        self.device, self.model, self.extent = torch.device(device), model, extent
        self._batch_weights = {key: str(path.resolve()) for key, path in paths.items()}
        self._batch_hyper, self._batch_steps = hyper, steps
        # Preserve FP32 accuracy for the TF checkpoint conversion on Ampere/Hopper.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.network = SynthMorphNetwork(weights=paths, model=model, hyper=hyper,
                                        int_steps=steps, device=device)

    @torch.inference_mode()
    def __call__(self, moving, fixed, init=None, mid_space=False, header_only=False,
                 output_dir=None):
        mov, fix = _load(moving), _load(fixed)
        is_matrix = self.model in ('affine', 'rigid')
        if header_only and not is_matrix:
            raise ValueError('header_only requires affine or rigid model')
        if mid_space and init is None:
            raise ValueError('mid_space initialization requires init')
        shape = (self.extent,) * 3
        net_to_mov, mov_to_net = network_space(mov, shape, fix if self.model == 'deform' else None)
        net_to_fix, fix_to_net = network_space(fix, shape)
        if init is not None:
            initial = sf.load_affine(str(init)) if isinstance(init, (str, Path)) else init
            initial = initial.convert(space='voxel')
            if not sf.transform.image_geometry_equal(mov.geom, initial.source, tol=1e-3) or not sf.transform.image_geometry_equal(fix.geom, initial.target, tol=1e-3):
                raise ValueError('initial transform geometry does not match input images')
            initial = fix_to_net @ initial.matrix @ net_to_mov
            if mid_space:
                initial = scipy.linalg.sqrtm(initial)
                if np.iscomplexobj(initial) and np.max(np.abs(initial.imag)) > 1e-5:
                    raise ValueError('initial affine has no usable real square root')
                initial = np.real(initial)
                net_to_fix = net_to_fix @ initial
                fix_to_net = np.linalg.inv(net_to_fix)
            net_to_mov = net_to_mov @ np.linalg.inv(initial)
            mov_to_net = np.linalg.inv(net_to_mov)
        native = [_tensor(im, self.device) for im in (mov, fix)]
        inputs = []
        for image, matrix in zip(native, (net_to_mov, net_to_fix)):
            out = transform(image, matrix, shape=shape)
            out -= out.min()
            maximum = out.max()
            if maximum <= 0:
                raise ValueError('input has no intensity variation in network space')
            inputs.append(out / maximum)
        fw_net, bw_net = self.network(*inputs)
        fw = compose((net_to_mov, fw_net, fix_to_net), shape=fix.shape)
        bw = compose((net_to_fix, bw_net, mov_to_net), shape=mov.shape)
        if is_matrix:
            forward = sf.Affine(_numpy(bw), source=mov, target=fix, space='voxel')
            inverse = sf.Affine(_numpy(fw), source=fix, target=mov, space='voxel')
            output_format = dict(space='world')
        else:
            forward = sf.Warp(_numpy(fw), source=mov, target=fix, format=sf.Warp.Format.disp_crs)
            inverse = sf.Warp(_numpy(bw), source=fix, target=mov, format=sf.Warp.Format.disp_crs)
            output_format = dict(format=sf.Warp.Format.disp_ras)
        # Surfa has a different valid interpolation domain from the network's
        # TensorFlow sampler. Preserve native CRS transforms until after resampling.
        moved = mov.transform(forward, resample=not header_only)
        fixed_moved = fix.transform(inverse, resample=not header_only)
        forward = forward.convert(**output_format)
        inverse = inverse.convert(**output_format)
        if output_dir:
            root = Path(output_dir)
            root.mkdir(parents=True, exist_ok=True)
            net_mov = sf.Volume(_numpy(inputs[0])[..., 0], geometry=sf.ImageGeometry(shape, vox2world=mov.geom.vox2world.matrix @ net_to_mov))
            net_fix = sf.Volume(_numpy(inputs[1])[..., 0], geometry=sf.ImageGeometry(shape, vox2world=fix.geom.vox2world.matrix @ net_to_fix))
            net_mov.save(root / 'inp_1.nii.gz')
            net_fix.save(root / 'inp_2.nii.gz')
            np.savez_compressed(root / 'network_transforms.npz', forward=_numpy(fw_net), inverse=_numpy(bw_net))
        return RegistrationResult(moved, fixed_moved, forward, inverse)

    def predict_batch(self, table, fixed, workers=1, threads_per_worker=1):
        """Register each input to one shared or row-aligned list of fixed images."""
        cases = cases_from_table(table)
        if isinstance(fixed, list):
            if len(fixed) != len(cases):
                raise ValueError('fixed list length must equal the number of input rows')
            fixed_images = fixed
        else:
            fixed_images = ([_load(fixed)] if workers == 1 else [fixed]) * len(cases) if cases else []
        if not cases:
            return []
        if workers != 1 and len(cases) > 1 and any(
                not isinstance(target, (str, Path)) for target in fixed_images):
            raise TypeError('parallel fixed images must be file paths')
        transform_suffix = '.lta' if self.model in ('affine', 'rigid') else '.mgz'
        def paths_for(prefix):
            return {"moved": Path(f"{prefix}_moved.nii.gz"),
                    "fixed_moved": Path(f"{prefix}_fixed_moved.nii.gz"),
                    "transform": Path(f"{prefix}_transform{transform_suffix}"),
                    "inverse": Path(f"{prefix}_inverse{transform_suffix}")}

        def run_local(case):
            source, prefix, target = case
            result = self(source, target)
            paths = paths_for(prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            for name, path in paths.items():
                getattr(result, name).save(path)
            return paths

        def make_job(case):
            source, prefix, target = case
            hyper = (self.network.deform.hyper if self.model in ('joint', 'deform')
                     else self._batch_hyper)
            return {"task": "synthmorph",
                    "model": {"weights": self._batch_weights, "model": self.model,
                              "extent": self.extent, "hyper": hyper,
                              "steps": self._batch_steps},
                    "kwargs": {"moving": source, "fixed": target},
                    "outputs": paths_for(prefix)}

        rows = [(source, prefix, target) for (source, prefix), target in zip(cases, fixed_images)]
        return run_parallel(rows, self, 'synthmorph', workers, threads_per_worker,
                            make_job, run_local)


@torch.inference_mode()
def apply_transform(image, transformation, device='cpu', method='linear', fill=0,
                    dtype='float32', header_only=False):
    """Apply an LTA or RAS warp with the original Surfa CPU resampling rules.

    ``device`` is retained for API compatibility; applying a saved transform
    is CPU postprocessing and does not invoke a neural network.
    """
    image = _load(image, single_frame=False)
    if isinstance(transformation, (str, Path)):
        path = str(transformation)
        transformation = sf.load_affine(path) if path.endswith('.lta') else sf.load_warp(path)
    if header_only and not isinstance(transformation, sf.Affine):
        raise ValueError('header_only requires an affine')
    if isinstance(transformation, sf.Warp):
        # Warp source geometry determines CRS coordinates; disallow silent mismatch.
        if not sf.transform.image_geometry_equal(image.geom, transformation.source, tol=1e-3):
            raise ValueError('warp source geometry does not match image')
    return image.transform(transformation, method=method, fill=fill,
                           resample=not header_only).astype(dtype)
