"""Parallel table orchestration without checkpoint or GPU fixtures."""

from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import sys

import pytest

from freesurfer_torch._parallel_table import run_parallel


class FakeTorch:
    def __init__(self):
        self.threads = 4

    def get_num_threads(self):
        return self.threads

    def set_num_threads(self, value):
        self.threads = value


def test_two_workers_overlap_and_return_in_input_order(tmp_path):
    started, release = Event(), Event()
    torch = FakeTorch()
    calls = []

    class Runner:
        def __init__(self, **options):
            assert options == {'devices': ('cuda:0',), 'workers_per_device': 1,
                               'threads_per_worker': 1}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run(self, jobs, overwrite):
            assert overwrite is True
            started.set()
            assert release.wait(2), 'parent did not overlap worker dispatch'
            return [SimpleNamespace(error=None, outputs=job['outputs']) for job in jobs]

    def local(case):
        calls.append(case)
        assert started.wait(2), 'worker was not dispatched before local inference'
        release.set()
        return {'image': tmp_path / f'{case}.nii.gz'}

    cases = list(range(4))
    with patch.dict(sys.modules, {'torch': torch}), patch('freesurfer_torch.batch.BatchRunner', Runner):
        paths = run_parallel(cases, SimpleNamespace(device='cuda:0'), 'synthstrip',
                             2, 1, lambda case: {'outputs': {'image': tmp_path / f'{case}.nii.gz'}},
                             local)
    assert calls == [0, 2]
    assert paths == [{'image': tmp_path / f'{index}.nii.gz'} for index in cases]
    assert torch.threads == 4


def test_worker_error_has_table_row_and_restores_threads(tmp_path):
    torch = FakeTorch()

    class Runner:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run(self, jobs, overwrite):
            return [SimpleNamespace(error='FileNotFoundError: missing input', outputs={})
                    for _ in jobs]

    with patch.dict(sys.modules, {'torch': torch}), patch('freesurfer_torch.batch.BatchRunner', Runner):
        with pytest.raises(RuntimeError, match='synthsr row 1: FileNotFoundError'):
            run_parallel([0, 1], SimpleNamespace(device='cpu'), 'synthsr', 2, 1,
                         lambda case: {'outputs': {'image': tmp_path / f'{case}.nii.gz'}},
                         lambda case: {'image': tmp_path / f'{case}.nii.gz'})
    assert torch.threads == 4


def test_parent_error_is_not_hidden_by_child_failure():
    torch = FakeTorch()

    class Runner:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run(self, jobs, overwrite):
            raise RuntimeError('child failed')

    def local(case):
        raise ValueError('parent failed')

    with patch.dict(sys.modules, {'torch': torch}), patch('freesurfer_torch.batch.BatchRunner', Runner):
        with pytest.raises(ValueError, match='parent failed'):
            run_parallel([0, 1], SimpleNamespace(device='cpu'), 'synthsr', 2, 1,
                         lambda case: {'outputs': {}}, local)
    assert torch.threads == 4


@pytest.mark.parametrize('workers,threads', [(0, 1), (True, 1), (2, 0)])
def test_invalid_worker_counts_fail_before_writing(workers, threads):
    with pytest.raises(ValueError):
        run_parallel([0], SimpleNamespace(device='cpu'), 'synthstrip', workers, threads,
                     lambda case: {}, lambda case: None)
