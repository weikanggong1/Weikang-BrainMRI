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
    args = parser.parse_args()
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
    area_scale = original_area / total_area
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

    def objective(trial):
        terms = first_registration_sse(trial, triangles, neighbors, degrees,
                                       original_distances, original_areas,
                                       curvature, mean_grid, variance_grid,
                                       original_area, total_area, return_terms=True)
        trial_distances = sphere_arc_distances(trial, neighbors, degrees)
        spring_sse = area_scale * float(((trial_distances.double() ** 2) * active).sum())
        weighted = {'sse_area': terms['sse_area'], 'sse_nl_area': terms['sse_nl_area'],
                    'sse_dist': terms['sse_dist'], 'sse_corr': 0.05 * terms['sse_corr'],
                    'sse_spring': 0.5 * spring_sse}
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
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'dt': dt, 'seconds': report['seconds_excluding_io'],
                      'checks': {name: (item['exact_vertices'], item['max_abs_error'])
                                 for name, item in checks.items()}}))


if __name__ == '__main__':
    main()
