"""Run SynthStrip on three public T1w images with two Python workers."""

from pathlib import Path

import pandas as pd

from freesurfer_torch import SynthStrip


ROOT = Path(__file__).resolve().parent


def main():
    table = pd.DataFrame({
        "input": [str(ROOT / "data" / f"sub-{index:02d}_T1w.nii.gz") for index in (1, 2, 3)],
        "output": [str(ROOT / "results" / "python" / f"sub-{index:02d}")
                   for index in (1, 2, 3)],
    })
    model = SynthStrip(device="cuda:0")
    for paths in model.predict_batch(table, workers=2):
        print(paths)


if __name__ == "__main__":
    main()
