"""Compare numerical wmparc.stats fields with a fixed native reference."""

import argparse
import hashlib
from pathlib import Path


def _parse(path: Path):
    lines = path.read_text().splitlines()
    rows = [line.split() for line in lines if line and not line.startswith("#")]
    measures = [line for line in lines if line.startswith("# Measure ")]
    schema = [line.split() for line in lines if line.startswith(
        ("# TableCol ", "# NRows ", "# NTableCols ", "# ColHeaders "))]
    return rows, measures, schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("python", type=Path)
    args = parser.parse_args()
    official_rows, official_measures, official_schema = _parse(args.native)
    python_rows, python_measures, python_schema = _parse(args.python)
    mismatches = []
    if len(official_rows) != len(python_rows):
        mismatches.append(("row_count", len(official_rows), len(python_rows)))
    for official, candidate in zip(official_rows, python_rows):
        for column, (left, right) in enumerate(zip(official, candidate)):
            if left != right:
                mismatches.append((official[1], column, left, right))
    if official_measures != python_measures:
        mismatches.append(("measures", official_measures, python_measures))
    if official_schema != python_schema:
        mismatches.append(("schema", official_schema, python_schema))
    print("rows", len(official_rows), len(python_rows),
          "row_cells", sum(map(len, official_rows)),
          "measures", len(official_measures),
          "mismatches", len(mismatches))
    print("official_sha256", hashlib.sha256(args.native.read_bytes()).hexdigest())
    print("python_sha256", hashlib.sha256(args.python.read_bytes()).hexdigest())
    for item in mismatches[:20]:
        print("difference", item)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
