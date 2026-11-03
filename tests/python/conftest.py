"""Gemeinsame pytest-Konfiguration: automatische Skips für gpu/reference-Tests."""

import importlib.util
import sys
from pathlib import Path

import pytest

# NumPy-Referenzimplementierungen (tests/reference/*_ref.py) importierbar machen.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))


def _cuda_available() -> bool:
    try:
        from biooptomd import _core
    except ImportError:
        return False
    return _core.cuda_device_count() > 0


def pytest_collection_modifyitems(config, items):
    has_gpu = _cuda_available()
    has_openmm = importlib.util.find_spec("openmm") is not None
    skip_gpu = pytest.mark.skip(reason="kein CUDA-Gerät / _core nicht gebaut")
    skip_ref = pytest.mark.skip(reason="OpenMM nicht installiert")
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "reference" in item.keywords and not has_openmm:
            item.add_marker(skip_ref)
