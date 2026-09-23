"""Standalone PyTorch brain MRI inference tools."""
__version__ = '0.4.0'


def __getattr__(name):
    if name in ('SynthStrip', 'StripResult'):
        from . import synthstrip
        return getattr(synthstrip, name)
    if name in ('SynthMorph', 'RegistrationResult', 'apply_transform'):
        from . import synthmorph
        return getattr(synthmorph, name)
    if name in ('WMHSynthSeg', 'WMHResult'):
        from . import wmh_synthseg
        return getattr(wmh_synthseg, name)
    if name in ('SynthSR', 'SynthSRResult', 'SynthSRImage'):
        from . import synthsr
        return getattr(synthsr, name)
    if name in ('BatchRunner', 'BatchResult', 'run_batch'):
        from . import batch
        return getattr(batch, name)
    raise AttributeError(name)
