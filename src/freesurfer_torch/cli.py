"""Command-line entry points for single images and JSON batch manifests."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='fs-torch')
    parser.add_argument('--version', action='version', version='freesurfer-torch 0.2.0')
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
        results = run_batch(jobs, devices=args.devices, workers_per_device=args.workers_per_device,
                            threads_per_worker=args.threads_per_worker, overwrite=args.overwrite)
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(result) for result in results], indent=2))
        raise SystemExit(int(any(result.error for result in results)))
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
