"""Explicit local weights; inference never downloads or imports FreeSurfer."""
import os
from pathlib import Path


def resolve_weights(filename, explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser()
        path = path / filename if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    roots = [os.environ.get('FREESURFER_TORCH_WEIGHTS'), Path.home() / '.cache' / 'freesurfer_torch']
    if os.environ.get('FREESURFER_HOME'):
        roots.append(Path(os.environ['FREESURFER_HOME']) / 'models')
    for root in roots:
        if root and (Path(root) / filename).is_file():
            return Path(root) / filename
    raise FileNotFoundError(f'{filename}: provide weights= or set FREESURFER_TORCH_WEIGHTS to the weights directory')
