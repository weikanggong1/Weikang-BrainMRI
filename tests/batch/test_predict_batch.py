"""Prefix validation and per-model multi-subject saving without checkpoints."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from freesurfer_torch._batch_table import cases_from_table

try:
    import pandas as pd
    from freesurfer_torch.synthstrip.pipeline import SynthStrip
    from freesurfer_torch.wmh_synthseg.pipeline import WMHSynthSeg
    from freesurfer_torch.synthsr.pipeline import SynthSR
    from freesurfer_torch.synthmorph.pipeline import SynthMorph
except ImportError:
    pd = None


class _Table:
    def __init__(self, rows, columns=('input', 'output')):
        self.rows, self.columns = rows, columns

    def __getitem__(self, columns):
        return self

    def itertuples(self, index=False, name=None):
        return iter(self.rows)


class TableValidationTests(unittest.TestCase):
    def test_absolute_unique_basename_prefixes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prefix = root / 'case.v2'
            self.assertEqual(cases_from_table(_Table([('scan.nii.gz', str(prefix))])),
                             [('scan.nii.gz', prefix)])
            with self.assertRaisesRegex(ValueError, 'absolute'):
                cases_from_table(_Table([('scan.nii.gz', 'relative')]))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                cases_from_table(_Table([('a', prefix), ('b', root / 'x' / '..' / 'case.v2')]))
            with self.assertRaisesRegex(ValueError, 'basename'):
                cases_from_table(_Table([('a', root / 'scan.nii.gz')]))
            with self.assertRaisesRegex(ValueError, 'basename'):
                cases_from_table(_Table([('a', root)]))
            with self.assertRaisesRegex(ValueError, 'exactly'):
                cases_from_table(_Table([], columns=('input', 'output', 'extra')))


class _Volume:
    def __init__(self, batch):
        self.batch = batch

    def save(self, path):
        Path(path).write_text(str(self.batch))


class _Model:
    def __init__(self, fields, mode='joint'):
        self.fields, self.model = fields, mode
        self.fixed_seen = []
        self.calls = 0

    def _result(self, batch):
        values = {field: _Volume(batch) for field in self.fields}
        values['volumes_mm3'] = {label: 1.0 for label in range(100)}
        return SimpleNamespace(**values)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if len(args) > 1:
            self.fixed_seen.append(args[1])
        return self._result(1)


@unittest.skipUnless(pd is not None, 'pandas and inference dependencies are required')
class PerModelDispatchTests(unittest.TestCase):
    def test_all_outputs_are_saved_with_sequential_single_case_calls(self):
        models = (
            (SynthStrip, {'image': '_brain.nii.gz', 'mask': '_mask.nii.gz',
                          'distance': '_sdt.nii.gz'}),
            (WMHSynthSeg, {'segmentation': '_seg.nii.gz',
                           'lesion_probability': '_lesion_probs.nii.gz',
                           'volumes_csv': '_volumes.csv'}),
            (SynthSR, {'image': '_synthsr.nii.gz'}),
            (SynthMorph, {'moved': '_moved.nii.gz', 'fixed_moved': '_fixed_moved.nii.gz',
                          'transform': '_transform.mgz', 'inverse': '_inverse.mgz'}),
        )
        with tempfile.TemporaryDirectory() as folder:
            for cls, suffixes in models:
                with self.subTest(model=cls.__name__):
                    root = Path(folder) / cls.__name__
                    prefixes = [root / f'case-{index}' for index in range(3)]
                    table = pd.DataFrame({'input': ['a', 'b', 'c'],
                                          'output': [str(prefix) for prefix in prefixes]})
                    model = _Model(tuple(name for name in suffixes if name != 'volumes_csv'))
                    kwargs = {'fixed': 'reference.nii.gz'} if cls is SynthMorph else {}
                    fixed = patch('freesurfer_torch.synthmorph.pipeline._load',
                                  return_value='reference.nii.gz') if cls is SynthMorph else nullcontext()
                    with fixed:
                        returned = cls.predict_batch(model, table, **kwargs)
                        self.assertEqual(len(returned), 3)
                        self.assertEqual(model.calls, 3)
                        for prefix, paths in zip(prefixes, returned):
                            self.assertEqual(paths, {name: Path(f'{prefix}{suffix}')
                                                     for name, suffix in suffixes.items()})
                            for name, path in paths.items():
                                if name == 'volumes_csv':
                                    content = path.read_text()
                                    self.assertIn('Input-file,Intracranial-volume', content)
                                    self.assertIn(str(paths['segmentation']), content)
                                else:
                                    self.assertEqual(path.read_text(), '1')

    def test_synthmorph_row_aligned_fixed_and_length_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = pd.DataFrame({'input': ['a', 'b', 'c'],
                                  'output': [str(root / str(i)) for i in range(3)]})
            model = _Model(('moved', 'fixed_moved', 'transform', 'inverse'), mode='affine')
            with self.assertRaisesRegex(ValueError, 'length'):
                SynthMorph.predict_batch(model, table, fixed=['one', 'two'])
            self.assertFalse(any(root.iterdir()))
            paths = SynthMorph.predict_batch(model, table, fixed=['one', 'two', 'three'])
            self.assertEqual(model.fixed_seen, ['one', 'two', 'three'])
            self.assertEqual(paths[0]['transform'].suffix, '.lta')
            self.assertEqual(paths[0]['inverse'].suffix, '.lta')

    def test_wmh_worker_csv_matches_serial_bytes(self):
        from freesurfer_torch import batch

        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / 'case'
            table = pd.DataFrame({'input': ['scan.nii.gz'], 'output': [str(prefix)]})
            model = _Model(('segmentation', 'lesion_probability'))
            paths = WMHSynthSeg.predict_batch(model, table)[0]
            expected = paths['volumes_csv'].read_bytes()
            for path in paths.values():
                path.unlink()
            key = ('wmh_synthseg', '{}')
            with patch.object(batch, '_models', {key: model}), \
                    patch.object(batch, '_device', 'cpu'), \
                    patch.object(batch, '_initialization_error', None):
                result = batch._run_job(0, {
                    'task': 'wmh_synthseg', 'model': {},
                    'kwargs': {'image': 'scan.nii.gz'},
                    'outputs': {name: str(path) for name, path in paths.items()},
                })
            self.assertTrue(result.ok, result.traceback)
            self.assertEqual(paths['volumes_csv'].read_bytes(), expected)
