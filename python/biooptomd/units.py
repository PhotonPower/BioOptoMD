"""Physikalische Konstanten. Einheitensystem: nm, ps, amu, kJ/mol, e, K.

Muss identisch zu include/biooptomd/units.hpp sein (geprüft in tests/python/test_units.py).
"""

ONE_4PI_EPS0 = 138.935456  # kJ nm / (mol e^2)
KB = 0.0083144626  # kJ / (mol K)
FARADAY = 96.4853321  # kJ / (mol V)
DEBYE_PER_ENM = 48.0320471  # D / (e nm)
AVOGADRO = 6.02214076e23  # 1 / mol
AKMA_TIME_PS = 0.04888821  # ps, Zeiteinheit von CHARMM/DCD
FIXED_POINT_SCALE = 4294967296.0  # 2^32
EPS_R2 = 1.0e-8  # nm^2
EPS_CROSS2 = 1.0e-12  # nm^4, Untergrenze für |a x b|^2 (kollineare Geometrien)
C_LIGHT = 299792458.0  # m/s (SI, Optik-Kopplung)
EPS0_SI = 8.8541878128e-12  # F/m (SI, Optik-Kopplung; konsistent mit ONE_4PI_EPS0)
