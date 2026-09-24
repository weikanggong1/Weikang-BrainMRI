"""Command-line entry points for single-image inference."""
import argparse
import csv
import os
from pathlib import Path
import uuid


def _atomic_save(volume, path):
    path = Path(path)
    suffix = next((value for value in ('.nii.gz', '.nii', '.mgz', '.npz')
                   if path.name.endswith(value)), path.suffix)
    temporary = path.with_name(
        f'.{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}{suffix}')
    try:
        volume.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wmh_suffix(path):
    name = Path(path).name
    for suffix in ('.nii.gz', '.nii', '.mgz'):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    raise ValueError('WMH-SynthSeg supports .nii, .nii.gz and .mgz images')


def _run_wmh(args):
    from .wmh_synthseg import LABEL_IDS, LABEL_NAMES, WMHSynthSeg

    source, target = Path(args.i), Path(args.o)
    if not source.is_file():
        raise FileNotFoundError(source)
    _wmh_suffix(source)
    _wmh_suffix(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.csv_vols:
        Path(args.csv_vols).parent.mkdir(parents=True, exist_ok=True)
    model = WMHSynthSeg(weights=args.weights, device=args.device, threads=args.threads)
    rows = []
    result = model(source, crop=args.crop,
                   save_lesion_probabilities=args.save_lesion_probabilities)
    result.segmentation.save(target)
    print(target)
    if args.save_lesion_probabilities:
        stem, suffix = _wmh_suffix(target)
        probability = target.with_name(f'{stem}.lesion_probs{suffix}')
        result.lesion_probability.save(probability)
        print(probability)
    if args.csv_vols:
        import numpy as np
        volumes = result.volumes_mm3
        ordered = np.asarray([volumes[label] for label in LABEL_IDS], dtype=np.float32)
        rows.append([str(target), str(np.sum(ordered[1:])),
                     *(str(value) for value in ordered[1:])])
    if args.csv_vols:
        with Path(args.csv_vols).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Input-file', 'Intracranial-volume',
                             *(f'{name}({label})' for label, name in zip(LABEL_IDS, LABEL_NAMES)
                               if label != 0)])
            writer.writerows(rows)
        print(args.csv_vols)


def _synthsr_suffix(path):
    name = Path(path).name
    for suffix in ('.nii.gz', '.nii', '.mgz', '.npz'):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    raise ValueError('SynthSR supports .nii, .nii.gz, .mgz and .npz images')


def _run_synthsr(args):
    from .synthsr import SynthSR

    source, target = Path(args.i), Path(args.o)
    if not source.is_file():
        raise FileNotFoundError(source)
    _synthsr_suffix(source)
    if target.name.endswith(('.nii', '.nii.gz', '.mgz', '.npz')):
        output = target
    else:
        if target.suffix == '.txt':
            raise ValueError('A .txt output list is not supported by the single-image CLI')
        stem, suffix = _synthsr_suffix(source)
        output = target / f'{stem}_synthsr{suffix}'
    model = SynthSR(weights=args.weights, device='cpu' if args.cpu else args.device,
                    lowfield=args.lowfield, v1=args.v1, threads=args.threads)
    model(source, ct=args.ct, disable_flipping=args.disable_flipping,
          disable_sharpening=args.disable_sharpening).image.save(output)
    print(output)


def _run_fast(args):
    from .fast import TorchFAST

    model = TorchFAST(
        device=args.device,
        threads=args.threads,
        init_iterations=args.init_iterations,
        bias_iterations=args.bias_iterations,
        fixed_iterations=args.fixed_iterations,
        bias_fwhm_mm=0.0 if args.no_bias else args.bias_fwhm_mm,
        init_mrf=args.init_mrf,
        mrf=args.mrf,
        mixel_mrf=args.mixel_mrf,
        pve_steps=args.pve_steps,
    )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "_pve_0.nii.gz": "pve_csf",
        "_pve_1.nii.gz": "pve_gm",
        "_pve_2.nii.gz": "pve_wm",
        "_seg.nii.gz": "hard_segmentation",
        "_pveseg.nii.gz": "pve_segmentation",
        "_mixeltype.nii.gz": "mixel_type",
    }
    if args.save_bias:
        fields["_bias.nii.gz"] = "bias_field"
    if args.save_restored:
        fields["_restore.nii.gz"] = "restored"
    outputs = {suffix: Path(f"{prefix}{suffix}") for suffix in fields}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; use --overwrite")
    result = model(args.image, mask=args.mask)
    for suffix, path in outputs.items():
        _atomic_save(getattr(result, fields[suffix]), path)
        print(path)


def _run_flirt(args):
    import numpy as np
    import torch

    from .fast_vbm import TorchFLIRT

    if args.output is None and args.omat is None:
        raise ValueError("provide -out and/or -omat")
    destinations = [Path(value) for value in (args.output, args.omat) if value]
    if (
        len(destinations) == 2
        and destinations[0].resolve() == destinations[1].resolve()
    ):
        raise ValueError("-out and -omat must use different paths")
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; use --overwrite")
    torch.set_num_threads(args.threads)
    model = TorchFLIRT(
        device=args.device,
        strides=tuple(args.strides),
        steps=tuple(args.steps),
        learning_rates=tuple(args.learning_rates),
        cost=args.cost,
    )
    result = model(args.input, args.reference)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        _atomic_save(result.moved, args.output)
        print(args.output)
    if args.omat:
        matrix_path = Path(args.omat)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = matrix_path.with_name(
            f".{matrix_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            np.savetxt(temporary, result.matrix, fmt="%.12g")
            os.replace(temporary, matrix_path)
        finally:
            temporary.unlink(missing_ok=True)
        print(matrix_path)


def _run_fast_vbm(args):
    from .fast_vbm import FastVBM, OUTPUT_FILENAMES

    output_dir = Path(args.output_dir)
    report_path = output_dir / "fast_vbm_report.json"
    outputs = [output_dir / filename for filename in OUTPUT_FILENAMES.values()]
    existing = [path for path in (*outputs, report_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; use --overwrite")

    model = FastVBM(
        device=args.device,
        threads=args.threads,
        synthstrip_weights=args.synthstrip_weights,
        synthmorph_weights=args.synthmorph_weights,
        bias_correction=not args.no_bias,
        linear_strides=tuple(args.linear_strides),
        linear_steps=tuple(args.linear_steps),
        linear_learning_rates=tuple(args.linear_learning_rates),
        synthmorph_extent=args.synthmorph_extent,
        synthmorph_hyper=args.synthmorph_hyper,
        synthmorph_steps=args.synthmorph_steps,
        registration_backend=args.registration_backend,
        fnirt_strides=tuple(args.fnirt_strides),
        fnirt_steps=tuple(args.fnirt_steps),
        fnirt_learning_rates=tuple(args.fnirt_learning_rates),
        fnirt_input_fwhm_mm=tuple(args.fnirt_input_fwhm_mm),
        fnirt_reference_fwhm_mm=tuple(args.fnirt_reference_fwhm_mm),
        fnirt_warp_resolution_mm=args.fnirt_warp_resolution_mm,
        fnirt_regularization=tuple(args.fnirt_regularization),
        fnirt_jacobian_penalty=args.fnirt_jacobian_penalty,
    )
    result = model(args.image, args.template, brain_mask=args.brain_mask)
    paths = result.save(output_dir, overwrite=args.overwrite)
    for name in OUTPUT_FILENAMES:
        print(paths[name])
    print(report_path)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='fs-torch')
    parser.add_argument('--version', action='version', version='freesurfer-torch 0.8.0')
    commands = parser.add_subparsers(dest='command', required=True)
    strip = commands.add_parser('synthstrip', help='brain extraction')
    strip.add_argument('-i', '--image', required=True)
    strip.add_argument('-o', '--out')
    strip.add_argument('-m', '--mask')
    strip.add_argument('-d', '--sdt')
    strip.add_argument('--weights')
    strip.add_argument('--device', default='cpu')
    strip.add_argument('--no-csf', action='store_true')
    strip.add_argument('-b', '--border', type=float, default=1)
    strip.add_argument('-f', '--fill', type=float)
    strip.add_argument('-j', '--threads', type=int, default=4)
    reg = commands.add_parser('synthmorph', help='rigid/affine/deformable/joint registration')
    reg.add_argument('moving')
    reg.add_argument('fixed')
    reg.add_argument('-m', '--model', choices=('joint', 'deform', 'affine', 'rigid'), default='joint')
    reg.add_argument('--weights', help='directory containing official checkpoint files')
    reg.add_argument('--device', default='cpu')
    reg.add_argument('-o', '--out-moving')
    reg.add_argument('-O', '--out-fixed')
    reg.add_argument('-t', '--trans')
    reg.add_argument('-T', '--inverse')
    reg.add_argument('-i', '--init')
    reg.add_argument('-M', '--mid-space', action='store_true')
    reg.add_argument('-H', '--header-only', action='store_true')
    reg.add_argument('-e', '--extent', choices=(192, 256), type=int, default=256)
    reg.add_argument('-r', '--hyper', type=float, default=0.5)
    reg.add_argument('-n', '--steps', type=int, default=7)
    reg.add_argument('-j', '--threads', type=int, default=4)
    reg.add_argument('-d', '--output-dir')
    apply = commands.add_parser('apply', help='apply an LTA or RAS warp')
    apply.add_argument('transform')
    apply.add_argument('image')
    apply.add_argument('output')
    apply.add_argument('--device', default='cpu')
    apply.add_argument('-m', '--method', choices=('linear', 'nearest'), default='linear')
    apply.add_argument('-f', '--fill', type=float, default=0)
    apply.add_argument('-t', '--dtype', choices=('uint8', 'uint16', 'int16', 'int32', 'float32'), default='float32')
    apply.add_argument('-H', '--header-only', action='store_true')
    wmh = commands.add_parser('wmh-synthseg', help='WMH and anatomy segmentation')
    wmh.add_argument('--i', '-i', required=True, help='single 3D input image')
    wmh.add_argument('--o', '-o', required=True, help='segmentation image')
    wmh.add_argument('--csv_vols', '--csv-vols')
    wmh.add_argument('--device', default='cpu')
    wmh.add_argument('--threads', type=int, default=1)
    wmh.add_argument('--crop', action='store_true')
    wmh.add_argument('--save_lesion_probabilities', '--save-lesion-probabilities', action='store_true')
    wmh.add_argument('--weights', help='official checkpoint file or containing directory')
    sr = commands.add_parser('synthsr', help='synthesize a 1 mm T1-weighted image')
    sr.add_argument('--i', '-i', required=True, help='single input image')
    sr.add_argument('--o', '-o', required=True, help='output image or directory for this image')
    sr.add_argument('--device', default='cpu')
    sr.add_argument('--cpu', action='store_true', help='use CPU, matching the original --cpu')
    sr.add_argument('--threads', type=int, default=1)
    sr.add_argument('--ct', action='store_true')
    sr.add_argument('--lowfield', action='store_true')
    sr.add_argument('--v1', action='store_true')
    sr.add_argument('--disable_sharpening', action='store_true')
    sr.add_argument('--disable_flipping', action='store_true')
    sr.add_argument('--weights', '--model', help='official checkpoint file or containing directory')
    fast = commands.add_parser(
        'fast', help='three-tissue T1 segmentation and bias correction')
    fast.add_argument('-i', '--image', required=True,
                      help='brain-extracted, single-channel T1 image')
    fast.add_argument('-o', '--output-prefix', required=True,
                      help='output basename, matching FSL FAST -o')
    fast.add_argument('--mask', help='optional mask on the input grid')
    fast.add_argument('--device', default='cpu')
    fast.add_argument('--threads', type=int, default=1)
    fast.add_argument('-W', '--init-iterations', type=int, default=15)
    fast.add_argument('-I', '--bias-iterations', type=int, default=4)
    fast.add_argument('-O', '--fixed-iterations', type=int, default=4)
    fast.add_argument('-l', '--bias-fwhm-mm', type=float, default=20.0)
    fast.add_argument('-f', '--init-mrf', type=float, default=0.02)
    fast.add_argument('-H', '--mrf', type=float, default=0.1)
    fast.add_argument('-R', '--mixel-mrf', type=float, default=0.3)
    fast.add_argument('--pve-steps', type=int, default=100)
    fast.add_argument('-N', '--no-bias', action='store_true')
    fast.add_argument('-b', '--save-bias', action='store_true')
    fast.add_argument('-B', '--save-restored', action='store_true')
    fast.add_argument('--overwrite', action='store_true')
    flirt = commands.add_parser(
        'flirt', help='PyTorch 12-DOF affine with FSL FLIRT file contracts')
    flirt.add_argument('-in', '--in', dest='input', required=True,
                       help='moving/input image')
    flirt.add_argument('-ref', '--ref', dest='reference', required=True,
                       help='fixed/reference image defining the output grid')
    flirt.add_argument('-out', '--out', dest='output')
    flirt.add_argument('-omat', '--omat')
    flirt.add_argument('-dof', type=int, choices=(12,), default=12)
    flirt.add_argument('-cost', choices=('normcorr',), default='normcorr')
    flirt.add_argument('--device', default='cpu')
    flirt.add_argument('--threads', type=int, default=1)
    flirt.add_argument('--strides', type=int, nargs=3, default=(4, 2, 1))
    flirt.add_argument('--steps', type=int, nargs=3, default=(80, 60, 50))
    flirt.add_argument('--learning-rates', type=float, nargs=3,
                       default=(0.05, 0.025, 0.0125))
    flirt.add_argument('--overwrite', action='store_true')
    fast_vbm = commands.add_parser(
        'fast-vbm', help='raw T1 to bias-corrected FAST VBM maps')
    fast_vbm.add_argument('-i', '--image', required=True,
                          help='single-frame raw T1 image')
    fast_vbm.add_argument('--template', required=True,
                          help='GM template defining the output grid')
    fast_vbm.add_argument('-o', '--output-dir', required=True)
    fast_vbm.add_argument('--brain-mask',
                          help='optional input-grid mask; skips SynthStrip')
    fast_vbm.add_argument('--synthstrip-weights',
                          help='official SynthStrip checkpoint or containing directory')
    fast_vbm.add_argument('--synthmorph-weights',
                          help='official SynthMorph deform checkpoint; used by the synthmorph backend')
    fast_vbm.add_argument('--registration-backend', choices=('synthmorph', 'fnirt'),
                          default='synthmorph',
                          help='nonlinear registration backend')
    fast_vbm.add_argument('--device', default='cpu')
    fast_vbm.add_argument('--threads', type=int)
    fast_vbm.add_argument('--linear-strides', type=int, nargs=3,
                          default=(4, 2, 1), metavar=('COARSE', 'MIDDLE', 'FINE'))
    fast_vbm.add_argument('--linear-steps', type=int, nargs=3,
                          default=(80, 60, 50), metavar=('COARSE', 'MIDDLE', 'FINE'))
    fast_vbm.add_argument('--linear-learning-rates', type=float, nargs=3,
                          default=(0.05, 0.025, 0.0125),
                          metavar=('COARSE', 'MIDDLE', 'FINE'))
    fast_vbm.add_argument('--synthmorph-extent', type=int, choices=(192, 256),
                          default=256)
    fast_vbm.add_argument('--synthmorph-hyper', type=float, default=0.5)
    fast_vbm.add_argument('--synthmorph-steps', type=int, default=7)
    fast_vbm.add_argument('--fnirt-strides', type=int, nargs=4,
                          default=(4, 2, 1, 1),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-steps', type=int, nargs=4,
                          default=(20, 20, 30, 20),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-learning-rates', type=float, nargs=4,
                          default=(0.5, 0.25, 0.1, 0.05),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-input-fwhm-mm', type=float, nargs=4,
                          default=(6.0, 4.0, 2.0, 2.0),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-reference-fwhm-mm', type=float, nargs=4,
                          default=(4.0, 2.0, 0.0, 0.0),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-warp-resolution-mm', type=float, default=10.0)
    fast_vbm.add_argument('--fnirt-regularization', type=float, nargs=4,
                          default=(150.0, 75.0, 50.0, 30.0),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-jacobian-penalty', type=float, default=1.0)
    fast_vbm.add_argument('--no-bias', action='store_true',
                          help='disable TorchFAST bias-field correction')
    fast_vbm.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'wmh-synthseg':
        _run_wmh(args)
        return
    if args.command == 'synthsr':
        _run_synthsr(args)
        return
    if args.command == 'fast':
        _run_fast(args)
        return
    if args.command == 'flirt':
        _run_flirt(args)
        return
    if args.command == 'fast-vbm':
        _run_fast_vbm(args)
        return
    if args.command == 'synthstrip':
        from .synthstrip import SynthStrip
        if not any((args.out, args.mask, args.sdt)):
            parser.error('provide at least one -o, -m or -d output')
        result = SynthStrip(args.weights, args.device, args.no_csf, args.threads)(args.image, args.border, args.fill)
        outputs = ((result.image, args.out), (result.mask, args.mask), (result.distance, args.sdt))
    elif args.command == 'synthmorph':
        import torch
        from .synthmorph import SynthMorph
        if not any((args.out_moving, args.out_fixed, args.trans, args.inverse, args.output_dir)):
            parser.error('provide at least one registration output')
        torch.set_num_threads(args.threads)
        model = SynthMorph(args.weights, args.device, args.model, args.extent, args.hyper, args.steps)
        result = model(args.moving, args.fixed, args.init, args.mid_space, args.header_only, args.output_dir)
        outputs = ((result.moved, args.out_moving), (result.fixed_moved, args.out_fixed),
                   (result.transform, args.trans), (result.inverse, args.inverse))
    else:
        from .synthmorph import apply_transform
        result = apply_transform(args.image, args.transform, args.device, args.method,
                                  args.fill, args.dtype, args.header_only)
        outputs = ((result, args.output),)
    for volume, path in outputs:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            volume.save(path)
            print(path)


if __name__ == '__main__':
    main()
