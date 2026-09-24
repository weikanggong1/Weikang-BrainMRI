"""Command-line entry points for single images and JSON batch manifests."""
import argparse
from dataclasses import asdict
import csv
import json
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
    if source.is_file():
        _wmh_suffix(source)
        _wmh_suffix(target)
        cases = [(source, target)]
    elif source.is_dir():
        if target.suffix in ('.nii', '.gz', '.mgz'):
            raise ValueError('Directory input requires an output directory')
        files = sorted(path for path in source.iterdir() if path.is_file()
                       and path.name.endswith(('.nii', '.nii.gz', '.mgz')))
        if not files:
            raise ValueError(f'No supported MRI images in {source}')
        cases = []
        for path in files:
            stem, suffix = _wmh_suffix(path)
            cases.append((path, target / f'{stem}_seg{suffix}'))
    else:
        raise FileNotFoundError(source)

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    if args.csv_vols:
        Path(args.csv_vols).parent.mkdir(parents=True, exist_ok=True)
    model = WMHSynthSeg(weights=args.weights, device=args.device, threads=args.threads)
    rows = []
    for image, output in cases:
        result = model(image, crop=args.crop,
                       save_lesion_probabilities=args.save_lesion_probabilities)
        result.segmentation.save(output)
        print(output)
        if args.save_lesion_probabilities:
            stem, suffix = _wmh_suffix(output)
            probability = output.with_name(f'{stem}.lesion_probs{suffix}')
            result.lesion_probability.save(probability)
            print(probability)
        if args.csv_vols:
            import numpy as np
            volumes = result.volumes_mm3
            ordered = np.asarray([volumes[label] for label in LABEL_IDS], dtype=np.float32)
            rows.append([str(output), str(np.sum(ordered[1:])),
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
    if source.is_dir():
        if target.name.endswith(('.nii', '.nii.gz', '.mgz', '.npz', '.txt')):
            raise ValueError('Directory input requires an output directory')
        inputs = sorted(path for path in source.iterdir() if path.is_file()
                        and path.name.endswith(('.nii', '.nii.gz', '.mgz', '.npz')))
        cases = [(path, target / f'{_synthsr_suffix(path)[0]}_synthsr{_synthsr_suffix(path)[1]}')
                 for path in inputs]
    elif source.is_file() and source.suffix == '.txt':
        if target.suffix != '.txt':
            raise ValueError('A .txt input list requires a .txt output list')
        inputs = [Path(line.strip()) for line in source.read_text().splitlines() if line.strip()]
        outputs = [Path(line.strip()) for line in target.read_text().splitlines() if line.strip()]
        if len(inputs) != len(outputs):
            raise ValueError('Input and output lists must have equal length')
        cases = list(zip(inputs, outputs))
    elif source.is_file():
        _synthsr_suffix(source)
        if target.name.endswith(('.nii', '.nii.gz', '.mgz', '.npz')):
            cases = [(source, target)]
        else:
            stem, suffix = _synthsr_suffix(source)
            cases = [(source, target / f'{stem}_synthsr{suffix}')]
    else:
        raise FileNotFoundError(source)
    if not cases:
        raise ValueError(f'No supported MRI images in {source}')
    model = SynthSR(weights=args.weights, device='cpu' if args.cpu else args.device,
                    lowfield=args.lowfield, v1=args.v1, threads=args.threads)
    for image, output in cases:
        model(image, ct=args.ct, disable_flipping=args.disable_flipping,
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
        bias_correction=not args.no_bias,
        affine_steps=args.affine_steps,
        deform_steps=args.deform_steps,
        smoothness=args.smoothness,
    )
    result = model(args.image, args.template, brain_mask=args.brain_mask)
    paths = result.save(output_dir, overwrite=args.overwrite)
    for name in OUTPUT_FILENAMES:
        print(paths[name])
    print(report_path)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='fs-torch')
    parser.add_argument('--version', action='version', version='freesurfer-torch 0.6.0')
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
    wmh.add_argument('--i', '-i', required=True, help='3D input image or directory')
    wmh.add_argument('--o', '-o', required=True, help='segmentation image or directory')
    wmh.add_argument('--csv_vols', '--csv-vols')
    wmh.add_argument('--device', default='cpu')
    wmh.add_argument('--threads', type=int, default=1)
    wmh.add_argument('--crop', action='store_true')
    wmh.add_argument('--save_lesion_probabilities', '--save-lesion-probabilities', action='store_true')
    wmh.add_argument('--weights', help='official checkpoint file or containing directory')
    sr = commands.add_parser('synthsr', help='synthesize a 1 mm T1-weighted image')
    sr.add_argument('--i', '-i', required=True, help='input image, directory, or .txt path list')
    sr.add_argument('--o', '-o', required=True, help='output image, directory, or .txt path list')
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
    fast_vbm.add_argument('--device', default='cpu')
    fast_vbm.add_argument('--threads', type=int)
    fast_vbm.add_argument('--affine-steps', type=int, default=50)
    fast_vbm.add_argument('--deform-steps', type=int, default=40)
    fast_vbm.add_argument('--smoothness', type=float, default=10.0)
    fast_vbm.add_argument('--no-bias', action='store_true',
                          help='disable TorchFAST bias-field correction')
    fast_vbm.add_argument('--overwrite', action='store_true')
    batch = commands.add_parser('batch', help='run a JSON list of jobs with persistent GPU workers')
    batch.add_argument('manifest')
    batch.add_argument('--devices', nargs='+', default=['cuda:0'])
    batch.add_argument('--workers-per-device', type=int, default=1)
    batch.add_argument('--threads-per-worker', type=int, default=4)
    batch.add_argument('--report', required=True)
    batch.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'batch':
        from .batch import run_batch
        jobs = json.loads(Path(args.manifest).read_text())
        if any(job.get('task') == 'fast_vbm' for job in jobs):
            parser.error('fast_vbm multi-subject execution is available through the Python BatchRunner API only')
        path = Path(args.report).expanduser().resolve()
        manifest_path = Path(args.manifest).expanduser().resolve()
        if path == manifest_path:
            parser.error('--report must differ from the manifest')
        for index, job in enumerate(jobs):
            outputs = list(job.get('outputs', {}).values())
            if job.get('task') == 'synthmorph' and job.get('kwargs', {}).get('output_dir'):
                debug = Path(job['kwargs']['output_dir'])
                outputs.append(debug)
                outputs.extend(
                    debug / name for name in
                    ('inp_1.nii.gz', 'inp_2.nii.gz', 'network_transforms.npz')
                )
            for output in outputs:
                if Path(output).expanduser().resolve() == path:
                    parser.error(f'--report conflicts with job {index} output: {path}')
        if path.exists() and not args.overwrite:
            parser.error(f'report exists: {path}; use --overwrite to replace it')
        results = run_batch(jobs, devices=args.devices, workers_per_device=args.workers_per_device,
                            threads_per_worker=args.threads_per_worker, overwrite=args.overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.json')
        try:
            temporary.write_text(json.dumps([asdict(result) for result in results], indent=2))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        raise SystemExit(int(any(result.error for result in results)))
    if args.command == 'wmh-synthseg':
        _run_wmh(args)
        return
    if args.command == 'synthsr':
        _run_synthsr(args)
        return
    if args.command == 'fast':
        _run_fast(args)
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
