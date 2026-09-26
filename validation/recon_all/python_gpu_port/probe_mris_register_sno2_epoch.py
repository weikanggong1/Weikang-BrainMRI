"""Continue an exact sulc checkpoint through the first smoothwm integration."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
import torch

from fnit.recon_all.mris_register_atlas import sample_atlas_on_canonical_sphere
from fnit.recon_all.mris_register_blur import blur_atlas_frame
from fnit.recon_all.mris_register_kernels import normalize_mean_curvature, project_sphere
from fnit.recon_all.mris_register_line_search import first_registration_line_search, first_registration_sse
from fnit.recon_all.mris_register_nonlinear import (
    apply_spherical_gradient, average_gradients, correlation_gradient_add,
    face_area_normals, first_area_gradient, first_distance_gradient,
    ordered_neighbors_from_faces, original_chord_distances, registration_orig_area,
    registration_total_area, sphere_arc_distances, sphere_vertex_normals,
    spring_gradient_add, tangent_basis,
)
from fnit.recon_all.mris_register_parameterization import parameterize_curvature


def compare(candidate, reference_file):
    actual = candidate.cpu().numpy()
    if reference_file.suffix == '.bin':
        reference = np.fromfile(reference_file, dtype='<f4').reshape(actual.shape)
    else:
        reference, faces = fsio.read_geometry(str(reference_file))
    difference = np.abs(actual - reference)
    vertex_exact = np.all(actual == reference, axis=1) if actual.ndim == 2 else actual == reference
    vertex_close = np.all(difference <= 1e-5, axis=1) if actual.ndim == 2 else difference <= 1e-5
    return {'exact_vertices': int(vertex_exact.sum()),
            'within_1e-5_vertices': int(vertex_close.sum()),
            'total_vertices': len(actual),
            'max_abs_error': float(difference.max()),
            'p95_abs_error': float(np.percentile(difference, 95)),
            'reference_sha256': hashlib.sha256(reference_file.read_bytes()).hexdigest(),
            'predicted_array_sha256': hashlib.sha256(actual.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    for name in ('sphere', 'smoothwm', 'seed', 'atlas', 'raw_curvature',
                 'native_capture', 'native_output', 'report'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--native-line-gradient', type=Path)
    parser.add_argument('--skip-native-checkpoints', action='store_true')
    parser.add_argument('--native-prefix', type=Path)
    parser.add_argument('--hemisphere', choices=('lh', 'rh'), default='lh')
    parser.add_argument('--first-epoch', type=int)
    parser.add_argument('--next-epochs', type=int, choices=range(51), default=0)
    parser.add_argument('--resume-epoch', type=int)
    parser.add_argument('--resume-report', type=Path)
    parser.add_argument('--dump-mismatch', type=Path)
    parser.add_argument('--continue-on-mismatch', action='store_true')
    args = parser.parse_args()
    if args.next_epochs and (args.native_prefix is None or args.first_epoch not in (56, 57)):
        parser.error('--next-epochs requires --native-prefix and first smoothwm epoch 56 or 57')
    if args.next_epochs and args.first_epoch != (57 if args.hemisphere == 'lh' else 56):
        parser.error('continuation must start at the hemisphere first smoothwm epoch')
    if args.hemisphere == 'rh' and args.next_epochs > 41:
        parser.error('RH fixed schedule ends at epoch 97')
    if (args.resume_epoch is None) != (args.resume_report is None):
        parser.error('--resume-epoch and --resume-report must be supplied together')
    torch.set_num_threads(4)
    start = perf_counter()
    sphere, faces = fsio.read_geometry(str(args.sphere))
    smoothwm, smoothwm_faces = fsio.read_geometry(str(args.smoothwm))
    seed, seed_faces = fsio.read_geometry(str(args.seed))
    native_output, output_faces = fsio.read_geometry(str(args.native_output))
    assert np.array_equal(faces, smoothwm_faces) and np.array_equal(faces, seed_faces)
    assert np.array_equal(faces, output_faces)
    vertices = torch.from_numpy(sphere.astype(np.float32))
    original = torch.from_numpy(smoothwm.astype(np.float32))
    current = torch.from_numpy(seed.astype(np.float32))
    triangles = torch.from_numpy(faces.astype(np.int64))
    neighbors, degrees = ordered_neighbors_from_faces(triangles, len(current))
    original_distances = original_chord_distances(original, neighbors, degrees)
    original_areas, _ = face_area_normals(original, triangles)
    original_area = registration_orig_area(vertices, triangles)
    total_area = registration_total_area()
    area_scale = float(np.float32(original_area / total_area))
    dist_scale = torch.tensor(math.sqrt(area_scale), dtype=torch.float32)
    atlas = tifffile.imread(args.atlas)
    raw = torch.from_numpy(np.fromfile(args.raw_curvature, dtype='<f4').copy())
    normalized = normalize_mean_curvature(raw)
    source_grid = parameterize_curvature(current, normalized)
    curvature = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        current, blur_atlas_frame(source_grid, 4.0)))
    raw_mean = torch.from_numpy(atlas[6].view(np.float32).copy())
    raw_variance = torch.from_numpy(atlas[7].view(np.float32).copy())
    mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        current, blur_atlas_frame(raw_mean, 4.0)))
    mean_grid = parameterize_curvature(current, mean_curve)
    variance_grid = blur_atlas_frame(raw_variance, 4.0)
    setup_seconds = perf_counter() - start

    projected = project_sphere(current)
    normals = sphere_vertex_normals(projected, triangles)
    distances = sphere_arc_distances(projected, neighbors, degrees)
    avg_vertex_dist = float(distances.double().sum() / degrees.sum())
    after_area = first_area_gradient(vertices, original, current, triangles,
                                     first_distance_gradient(vertices, original, current, triangles))
    e1, e2 = tangent_basis(normals)
    unit_force = correlation_gradient_add(after_area, projected, curvature, e1, e2,
                                          mean_grid, variance_grid, avg_vertex_dist, l_corr=0.05)
    after_correlation = unit_force
    force_seconds = perf_counter() - start - setup_seconds
    averaged = average_gradients(after_correlation, neighbors, degrees, 1024)
    average_seconds = perf_counter() - start - setup_seconds - force_seconds
    after_spring = spring_gradient_add(averaged, projected, neighbors, degrees,
                                       dist_scale, 0.5)
    spring_seconds = perf_counter() - start - setup_seconds - force_seconds - average_seconds

    active = torch.arange(neighbors.shape[1])[None, :] < degrees[:, None]

    trial_terms = []
    l_parea, l_nlarea, l_dist = 0.2, 1.0, 5.0
    l_corr, l_spring = float(np.float32(0.05)), 0.5

    def objective(trial):
        terms = first_registration_sse(trial, triangles, neighbors, degrees,
                                       original_distances, original_areas,
                                       curvature, mean_grid, variance_grid,
                                       original_area, total_area, return_terms=True)
        trial_distances = sphere_arc_distances(trial, neighbors, degrees)
        spring_sse = area_scale * float(((trial_distances.double() ** 2) * active).sum())
        weighted = {'sse_area': (l_parea / 0.2) * terms['sse_area'],
                    'sse_nl_area': l_nlarea * terms['sse_nl_area'],
                    'sse_dist': (l_dist / 5.0) * terms['sse_dist'],
                    'sse_corr': l_corr * terms['sse_corr'],
                    'sse_spring': l_spring * spring_sse}
        weighted['total'] = sum(weighted.values())
        trial_terms.append(weighted)
        return weighted['total']

    line_gradient = after_spring
    if args.native_line_gradient is not None:
        line_gradient = torch.from_numpy(np.fromfile(args.native_line_gradient, dtype='<f4').copy().reshape(-1, 3))
    dt, samples = first_registration_line_search(projected, line_gradient, objective)
    predicted = apply_spherical_gradient(projected, line_gradient, dt)
    line_seconds = perf_counter() - start - setup_seconds - force_seconds - average_seconds - spring_seconds
    cap = args.native_capture
    checks = {}
    if not args.skip_native_checkpoints:
        checks = {'positions': compare(projected, cap / 'before_distance_positions.bin'),
                  'normals': compare(normals, cap / 'before_distance_normals.bin'),
                  'curvature': compare(curvature, cap / 'before_distance_curvature.bin'),
                  'area': compare(after_area, cap / 'after_area_gradient.bin'),
                  'correlation': compare(after_correlation, cap / 'after_correlation_gradient.bin'),
                  'after_average': compare(averaged, cap / 'after_average_gradient.bin')}
        spring_reference = cap / 'after_spring_gradient.bin'
        if spring_reference.exists():
            checks['after_spring'] = compare(after_spring, spring_reference)
    checks['saved_surface'] = compare(predicted, args.native_output)
    report = {'input_sha256': {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                               for name in ('sphere', 'smoothwm', 'seed', 'atlas',
                                            'raw_curvature', 'native_output')},
              'dt': dt, 'line_samples': samples, 'line_sample_terms': trial_terms,
              'native_line_gradient_sha256': None if args.native_line_gradient is None else
                  hashlib.sha256(args.native_line_gradient.read_bytes()).hexdigest(),
              'seconds_excluding_io': {'setup': setup_seconds, 'force': force_seconds,
                                       'average': average_seconds, 'spring': spring_seconds,
                                       'line': line_seconds},
              'checkpoints': checks}
    report['continuation'] = []
    current = predicted
    first_epoch = args.first_epoch or 0
    start_epoch = first_epoch + 1
    sigma_starts = ({81: 2.0, 88: 1.0, 95: 0.5} if args.hemisphere == 'lh'
                    else {77: 2.0, 84: 1.0, 91: 0.5})
    if args.resume_report is not None:
        previous = json.loads(args.resume_report.read_text())
        last = previous['continuation'][-1]
        if (last['epoch'] != args.resume_epoch or
                last['saved_surface']['exact_vertices'] != len(current) or
                previous['input_sha256'] != report['input_sha256']):
            raise ValueError('resume report does not prove an exact matching input')
        checkpoint = Path(f'{args.native_prefix}{args.resume_epoch:04d}')
        resumed, resumed_faces = fsio.read_geometry(str(checkpoint))
        resumed = resumed.astype(np.float32)
        resumed_hash = hashlib.sha256(resumed.tobytes()).hexdigest()
        if (not np.array_equal(faces, resumed_faces) or
                resumed_hash != last['saved_surface']['predicted_array_sha256'] or
                hashlib.sha256(checkpoint.read_bytes()).hexdigest() !=
                last['saved_surface']['reference_sha256']):
            raise ValueError('resume checkpoint differs from the verified Python state')
        current = torch.from_numpy(resumed)
        start_epoch = args.resume_epoch + 1
        report['resume'] = {'report_sha256': hashlib.sha256(args.resume_report.read_bytes()).hexdigest(),
                            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            'coordinate_sha256': resumed_hash, 'epoch': args.resume_epoch}
        sigma_epoch = max((number for number in sigma_starts if number <= args.resume_epoch), default=None)
        if sigma_epoch is not None:
            sigma_seed_path = Path(f'{args.native_prefix}{sigma_epoch - 1:04d}')
            sigma_seed, sigma_faces = fsio.read_geometry(str(sigma_seed_path))
            sigma_hash = hashlib.sha256(sigma_seed_path.read_bytes()).hexdigest()
            seed_row = next((row for row in previous['continuation']
                             if row['epoch'] == sigma_epoch - 1), None)
            if (seed_row is None or seed_row['saved_surface']['exact_vertices'] != len(current)
                    or seed_row['saved_surface']['reference_sha256'] != sigma_hash
                    or not np.array_equal(faces, sigma_faces)):
                raise ValueError('resume report does not prove the sigma-stage seed')
            sigma_seed = torch.from_numpy(sigma_seed.astype(np.float32))
            sigma = sigma_starts[sigma_epoch]
            source_grid = parameterize_curvature(sigma_seed, normalized)
            curvature = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
                sigma_seed, blur_atlas_frame(source_grid, sigma)))
            mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
                sigma_seed, blur_atlas_frame(raw_mean, sigma)))
            mean_grid = parameterize_curvature(sigma_seed, mean_curve)
            variance_grid = blur_atlas_frame(raw_variance, sigma)
            report['resume']['sigma_seed_sha256'] = sigma_hash
    for epoch in range(start_epoch, first_epoch + args.next_epochs + 1):
        epoch_start = perf_counter()
        if epoch in sigma_starts:
            sigma = sigma_starts[epoch]
            source_grid = parameterize_curvature(current, normalized)
            curvature = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
                current, blur_atlas_frame(source_grid, sigma)))
            mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
                current, blur_atlas_frame(raw_mean, sigma)))
            mean_grid = parameterize_curvature(current, mean_curve)
            variance_grid = blur_atlas_frame(raw_variance, sigma)
        fold_cleanup = args.hemisphere == 'lh' and epoch >= 102
        if fold_cleanup:
            l_parea = float(np.float32(np.float32(0.2) / 100))
            l_nlarea = 100.0
            l_dist = float(np.float32(np.float32(5.0) / 100))
            l_corr = float(np.float32(np.float32(0.05) / 100))
            l_spring = float(np.float32(np.float32(0.5) / 100))
        if args.hemisphere == 'lh':
            integration_start = (epoch in (59, 60, 69, 75, 78, 80)
                                 or 81 <= epoch <= 102 or 104 <= epoch <= 107)
        else:
            integration_start = epoch in (58, 59, 67, 70, 73, 75) or 77 <= epoch <= 97
        projected = project_sphere(current) if integration_start else current
        normals = sphere_vertex_normals(projected, triangles)
        distances = sphere_arc_distances(projected, neighbors, degrees)
        avg_vertex_dist = float(distances.double().sum() / degrees.sum())
        force = first_area_gradient(
            vertices, original, projected, triangles,
            first_distance_gradient(vertices, original, projected, triangles,
                                    project=False, weight=l_dist),
            project=False, l_nlarea=l_nlarea, l_parea=l_parea)
        e1, e2 = tangent_basis(normals)
        force = correlation_gradient_add(force, projected, curvature, e1, e2,
                                         mean_grid, variance_grid, avg_vertex_dist, l_corr=l_corr)
        force_seconds = perf_counter() - epoch_start
        if args.hemisphere == 'lh':
            if fold_cleanup:
                gradient_averages = (64, 64, 16, 4, 1, 0)[epoch - 102]
            elif epoch >= 81:
                gradient_averages = (1024, 256, 64, 16, 4, 1, 0)[(epoch - 81) % 7]
            else:
                gradient_averages = (1024 if epoch == 58 else 256 if epoch == 59 else
                                     64 if epoch <= 68 else 16 if epoch <= 74 else
                                     4 if epoch <= 77 else 1 if epoch <= 79 else 0)
        elif epoch >= 77:
            gradient_averages = (1024, 256, 64, 16, 4, 1, 0)[(epoch - 77) % 7]
        else:
            gradient_averages = (1024 if epoch == 57 else 256 if epoch == 58 else
                                 64 if epoch <= 66 else 16 if epoch <= 69 else
                                 4 if epoch <= 72 else 1 if epoch <= 74 else 0)
        averaged = average_gradients(force, neighbors, degrees, gradient_averages)
        force = spring_gradient_add(averaged, projected, neighbors, degrees,
                                    dist_scale, l_spring)
        average_seconds = perf_counter() - epoch_start - force_seconds
        first_sample = len(trial_terms)
        dt, samples = first_registration_line_search(projected, force, objective)
        current = apply_spherical_gradient(projected, force, dt)
        line_seconds = perf_counter() - epoch_start - force_seconds - average_seconds
        reference = Path(f'{args.native_prefix}{epoch:04d}')
        result = compare(current, reference)
        line_sample_terms = trial_terms[first_sample:]
        reference_vertices, reference_faces = fsio.read_geometry(str(reference))
        assert np.array_equal(faces, reference_faces)
        native_reference_sse = objective(torch.from_numpy(reference_vertices.astype(np.float32)))
        native_reference_terms = trial_terms[-1]
        report['continuation'].append({
            'epoch': epoch, 'dt': dt, 'gradient_averages': gradient_averages,
            'integration_start_projection': integration_start, 'line_samples': samples,
            'line_sample_terms': line_sample_terms,
            'native_reference_sse': native_reference_sse,
            'native_reference_terms': native_reference_terms,
            'seconds_excluding_io': {'force': force_seconds, 'average_and_spring': average_seconds,
                                     'line': line_seconds},
            'saved_surface': result})
        print(json.dumps({'epoch': epoch, 'dt': dt, 'saved_surface': result}), flush=True)
        if result['exact_vertices'] != len(current):
            if args.dump_mismatch is not None:
                np.savez_compressed(args.dump_mismatch,
                                    projected=projected.cpu().numpy(),
                                    gradient=force.cpu().numpy(),
                                    reference=reference_vertices.astype(np.float32))
            if not args.continue_on_mismatch:
                break
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'dt': dt, 'seconds': report['seconds_excluding_io'],
                      'checks': {name: (item['exact_vertices'], item['max_abs_error'])
                                 for name, item in checks.items()}}))


if __name__ == '__main__':
    main()
