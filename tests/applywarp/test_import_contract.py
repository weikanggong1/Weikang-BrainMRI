"""Clean-interpreter import checks for the applywarp API."""

import subprocess
import sys


def test_root_class_and_subpackage_function_import_without_recursion():
    code = """
import freesurfer_torch as package
assert package.TorchApplyWarp.__name__ == 'TorchApplyWarp'
from freesurfer_torch.applywarp import applywarp
assert callable(applywarp)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
