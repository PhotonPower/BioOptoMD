"""Konstanten in C++ und Python müssen identisch sein (AGENTS.md Regel 1)."""

import re
from pathlib import Path

import numpy as np
import pytest

from biooptomd import units

HEADER = Path(__file__).resolve().parents[2] / "include" / "biooptomd" / "units.hpp"


def test_cpp_python_constants_identical():
    text = HEADER.read_text()
    found = dict(re.findall(r"constexpr double (\w+)\s*=\s*([0-9.eE+-]+);", text))
    assert found, "keine Konstanten im Header gefunden"
    for name, value in found.items():
        assert hasattr(units, name), f"{name} fehlt in biooptomd.units"
        assert float(value) == getattr(units, name), name


def test_compiled_constants_identical():
    """Die ins Modul kompilierten Werte (units.hpp) stimmen mit biooptomd.units überein."""
    core = pytest.importorskip("biooptomd._core")
    for name in ("ONE_4PI_EPS0", "KB", "FARADAY", "DEBYE_PER_ENM", "FIXED_POINT_SCALE"):
        assert getattr(core.units, name) == getattr(units, name), name
    # EPS_R2 ist in C++ ein float.
    assert core.units.EPS_R2 == float(np.float32(units.EPS_R2))
