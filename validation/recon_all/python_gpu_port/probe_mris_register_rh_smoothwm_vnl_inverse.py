"""One-operation replacement of the RH curvature-fit inverse at two vertices."""

import hashlib
import json
from pathlib import Path

import numpy as np

from fnit.recon_all.topology_vnl_svd import add, mul, svd_inverse_3, svdc_3


def main():
    root = Path(__file__).resolve().parent
    native = json.loads((root / "native_smoothwm_fit.json").read_text())
    raw = np.fromfile(root / "native_smoothwm_curvature.bin", dtype="<f4")
    report = {"native_fit_sha256": hashlib.sha256((root / "native_smoothwm_fit.json").read_bytes()).hexdigest(),
              "vnl_source_sha256": hashlib.sha256(
                  (root / "src/fnit/recon_all/topology_vnl_svd.py").read_bytes()).hexdigest(),
              "vertices": {}}
    for key, stage in native.items():
        gram = np.array(stage["gram"], np.float32)
        rhs = np.array(stage["rhs"], np.float32).ravel()
        inverse = svd_inverse_3(gram)
        coeff = np.zeros(3, np.float32)
        for row in range(3):
            value = np.float32(0)
            for col in range(3):
                value = add(value, mul(inverse[row, col], rhs[col]))
            coeff[row] = value
        native_inverse = np.array(stage["inverse"], np.float32)
        native_coeff = np.array(stage["coefficients"], np.float32).ravel()
        _, singular, _ = svdc_3(gram)
        trace = np.float32(coeff[0] + coeff[2])
        report["vertices"][key] = {
            "inverse_exact_elements": int(np.count_nonzero(inverse == native_inverse)),
            "inverse_max_abs_error": float(np.max(np.abs(inverse - native_inverse))),
            "coeff_exact_elements": int(np.count_nonzero(coeff == native_coeff)),
            "coeff_max_abs_error": float(np.max(np.abs(coeff - native_coeff))),
            "source_order_singular": singular.tolist(),
            "source_order_trace": float(trace),
            "native_H": float(raw[int(key)]),
            "native_coeff": native_coeff.tolist(),
            "source_order_coeff": coeff.tolist(),
        }
    (root / "rh_smoothwm_vnl_inverse.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
