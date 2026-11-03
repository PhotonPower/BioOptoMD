"""Konstanten in C++ und Python müssen identisch sein (AGENTS.md Regel 1)."""

import re
from pathlib import Path

from biooptomd import units

HEADER = Path(__file__).resolve().parents[2] / "include" / "biooptomd" / "units.hpp"


def test_cpp_python_constants_identical():
    text = HEADER.read_text()
    found = dict(re.findall(r"constexpr double (\w+)\s*=\s*([0-9.eE+-]+);", text))
    assert found, "keine Konstanten im Header gefunden"
    for name, value in found.items():
        assert hasattr(units, name), f"{name} fehlt in biooptomd.units"
        assert float(value) == getattr(units, name), name
