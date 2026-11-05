"""Konstanten in C++ und Python müssen identisch sein (AGENTS.md Regel 1)."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from biooptomd import units

HEADER = Path(__file__).resolve().parents[2] / "include" / "biooptomd" / "units.hpp"

# Erfasst `constexpr double NAME = 1.0;` und `constexpr float NAME = 1.0f;`.
# Abgeleitete Konstanten (z. B. `static_cast<float>(…)`) werden bewusst nicht erfasst.
_PATTERN = re.compile(r"constexpr\s+(double|float)\s+(\w+)\s*=\s*([0-9.eE+-]+)f?\s*;")

# float32 hat ~7 Dezimalstellen; der Python-Wert ist double.
_FLOAT_REL_TOL = 1e-6


def _header_constants():
    return [
        (m.group(1), m.group(2), float(m.group(3))) for m in _PATTERN.finditer(HEADER.read_text())
    ]


def test_cpp_python_constants_identical():
    found = _header_constants()
    assert found, "keine Konstanten im Header gefunden"
    for ctype, name, value in found:
        assert hasattr(units, name), f"{name} fehlt in biooptomd.units"
        py_value = getattr(units, name)
        if ctype == "double":
            assert value == py_value, name
        else:
            assert math.isclose(value, py_value, rel_tol=_FLOAT_REL_TOL), name


def test_float_constants_are_checked():
    """Regressionsschutz: EPS_R2 (float) muss vom Parser erfasst werden."""
    names = {name for _, name, _ in _header_constants()}
    assert "EPS_R2" in names


def test_compiled_constants_identical():
    """Die ins Modul kompilierten Werte (units.hpp) stimmen mit biooptomd.units überein."""
    core = pytest.importorskip("biooptomd._core")
    for name in ("ONE_4PI_EPS0", "KB", "FARADAY", "DEBYE_PER_ENM", "FIXED_POINT_SCALE"):
        assert getattr(core.units, name) == getattr(units, name), name
    # EPS_R2 ist in C++ ein float.
    assert core.units.EPS_R2 == float(np.float32(units.EPS_R2))
