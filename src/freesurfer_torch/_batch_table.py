"""Input/output table checks shared by the per-model Python batch methods."""

from pathlib import Path


def cases_from_table(table):
    if not hasattr(table, 'columns') or set(table.columns) != {'input', 'output'} or len(table.columns) != 2:
        raise ValueError("table must be a pandas DataFrame with exactly 'input' and 'output' columns")
    cases, outputs = [], set()
    for source, destination in table[['input', 'output']].itertuples(index=False, name=None):
        if not isinstance(destination, (str, Path)):
            raise TypeError('output must be an absolute path prefix')
        path = Path(destination).expanduser()
        if not path.is_absolute():
            raise ValueError(f'output must be an absolute path prefix: {path}')
        path = path.resolve()
        if not path.name or path.is_dir() or path.name.lower().endswith(
                ('.nii', '.nii.gz', '.mgz', '.mgh', '.npz', '.csv', '.lta')):
            raise ValueError(f'output must be a basename prefix, not a directory or output file: {path}')
        if path in outputs:
            raise ValueError(f'duplicate output prefix: {path}')
        outputs.add(path)
        cases.append((source, path))
    return cases
