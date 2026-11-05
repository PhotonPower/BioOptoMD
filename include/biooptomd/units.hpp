// BioOptoMD – physikalische Konstanten und Einheiten.
// Einheitensystem: nm, ps, amu, kJ/mol, e, K.  Siehe docs/UNITS_AND_CONVENTIONS.md.
// Diese Werte müssen identisch zu python/biooptomd/units.py sein (Test:
// tests/python/test_units.py).
#pragma once

namespace bomd::units {

inline constexpr double ONE_4PI_EPS0 = 138.935456;         // kJ nm / (mol e^2)
inline constexpr double KB = 0.0083144626;                 // kJ / (mol K)
inline constexpr double FARADAY = 96.4853321;              // kJ / (mol V)
inline constexpr double DEBYE_PER_ENM = 48.0320471;        // D / (e nm)
inline constexpr double AVOGADRO = 6.02214076e23;          // 1 / mol
inline constexpr double AKMA_TIME_PS = 0.04888821;         // ps, Zeiteinheit von CHARMM/DCD
inline constexpr double FIXED_POINT_SCALE = 4294967296.0;  // 2^32
inline constexpr float EPS_R2 = 1.0e-8f;                   // nm^2
// Untergrenze für |a × b|² (kollineare Winkel/Torsionen, Kraft dort 0), nm^4
inline constexpr float EPS_CROSS2 = 1.0e-12f;

inline constexpr float ONE_4PI_EPS0_F = static_cast<float>(ONE_4PI_EPS0);
inline constexpr float KB_F = static_cast<float>(KB);
inline constexpr float FIXED_POINT_SCALE_F = static_cast<float>(FIXED_POINT_SCALE);  // exakt

}  // namespace bomd::units
