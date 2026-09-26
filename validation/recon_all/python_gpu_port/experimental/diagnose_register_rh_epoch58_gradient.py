"""Check whether the frozen RH second-step gradient differs from native geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from fnit.recon_all.mris_register_nonlinear import apply_spherical_gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    arrays = np.load(args.capture)
    start = torch.from_numpy(arrays['projected'])
    gradient = torch.from_numpy(arrays['gradient'])
    reference = arrays['reference']

    def compare(dt):
        result = apply_spherical_gradient(start, gradient, dt).numpy()
        difference = np.abs(result - reference)
        return {'exact_vertices': int(np.all(result == reference, axis=1).sum()),
                'max_abs_error_mm': float(difference.max()),
                'squared_error': float(np.square(difference.astype(np.float64)).sum())}

    fit = minimize_scalar(lambda dt: compare(dt)['squared_error'],
                          bounds=(1.5, 2.2), method='bounded',
                          options={'xatol': 1e-12})
    rounded = np.float32(fit.x)
    neighbors = [np.nextafter(rounded, np.float32(-np.inf)), rounded,
                 np.nextafter(rounded, np.float32(np.inf))]
    report = {'capture_sha256': hashlib.sha256(args.capture.read_bytes()).hexdigest(),
              'input_vertices': len(reference), 'best_fit_dt': float(fit.x),
              'best_fit': compare(fit.x),
              'float32_dt_neighbors': [{'dt': float(dt), **compare(float(dt))}
                                       for dt in neighbors],
              'python_selected': compare(1.7613636255264282),
              'native_printed': compare(1.857)}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
