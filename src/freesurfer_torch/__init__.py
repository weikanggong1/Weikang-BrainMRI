"""Standalone PyTorch brain MRI inference tools."""
__version__ = '0.8.0'


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
    if name in ('TorchFAST', 'FASTResult', 'FASTConfig', 'FASTTensorResult',
                'segment_t1'):
        from . import fast
        return getattr(fast, name)
    if name in ('FastVBM', 'FastVBMResult', 'FASTVBMResult', 'VBMRegistrationResult',
                'LinearRegistrationResult', 'FLIRTResult', 'TorchFLIRT',
                'FNIRTVBMResult', 'PyTorchFNIRTRegistration',
                'register_affine', 'register_gm', 'world_to_flirt_affine'):
        from . import fast_vbm
        return getattr(fast_vbm, name)
    if name in ('BatchRunner', 'BatchResult', 'run_batch'):
        from . import batch
        return getattr(batch, name)
    raise AttributeError(name)
