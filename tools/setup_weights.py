"""Run from a Git checkout: python tools/setup_weights.py [--model MODEL]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnit.weights import main


if __name__ == "__main__":
    main()
