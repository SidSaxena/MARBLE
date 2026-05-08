# setup.py
#
# The Cython HMM extension (marble.tasks.GTZANBeatTracking.madmom.hmm) was
# removed: the codebase uses hmm_numba.py (numba-JIT) instead, so the .pyx
# file is never imported.  All package metadata now lives in pyproject.toml.
# This file is kept as an empty shim so tools that expect setup.py still work.

from setuptools import setup

setup()
