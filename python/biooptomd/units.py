"""Physikalische Konstanten. Einheitensystem: nm, ps, amu, kJ/mol, e, K.

Muss identisch zu include/biooptomd/units.hpp sein (geprüft in tests/python/test_units.py).
"""

ONE_4PI_EPS0 = 138.935456  # kJ nm / (mol e^2)
KB = 0.0083144626  # kJ / (mol K)
FARADAY = 96.4853321  # kJ / (mol V)
DEBYE_PER_ENM = 48.0320471  # D / (e nm)
FIXED_POINT_SCALE = 4294967296.0  # 2^32
EPS_R2 = 1.0e-8  # nm^2
EPS_CROSS2 = 1.0e-12  # nm^4, Untergrenze für |a x b|^2 (kollineare Geometrien)
