"""Standalone PyTorch SynthStrip and SynthMorph inference."""
__version__ = '0.2.0'


def __getattr__(name):
    if name in ('SynthStrip', 'StripResult'):
        from . import synthstrip
        return getattr(synthstrip, name)
    if name in ('SynthMorph', 'RegistrationResult', 'apply_transform'):
        from . import synthmorph
        return getattr(synthmorph, name)
    if name in ('BatchRunner', 'BatchResult', 'run_batch'):
        from . import batch
        return getattr(batch, name)
    raise AttributeError(name)
