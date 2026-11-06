# ADR 0017: Coarse-Grained-Modelle (Antikörper und Aptamere auf Oberflächen)

- **Status:** Accepted (Projektinhaber 2026-11-06; Vorschläge zu OQ-27/28/29 angenommen)
- **Datum:** 2026-11-06
- **Bezug:** ROADMAP „Coarse-Grained-Kraftfelder (MARTINI, ENM für Antikörper)“, ARCHITECTURE §6,
  ADR 0002 (Fixed-Point), ADR 0004 (implizites Lösungsmittel), ADR 0005 (Topologie je Replica),
  ADR 0007 (Wände), ADR 0013 (Bildladungen), ADR 0016 (MD ↔ FEM), OQ-01/02 (Optikparameter).

## Kontext
Die Zielsysteme des Biosensors sind Aptamere (≈ 10²–10³ Atome) und Antikörper (IgG ≈ 150 kDa,
≈ 20 000 Atome, ≈ 1 300 Residuen) auf Gold oder Graphen. Für Antikörper sind atomistische Läufe
über die relevanten Zeiten zu teuer:
- Adsorption, Orientierung und Bedeckung brauchen Mikrosekunden bis Millisekunden.
- Die Bedeckung braucht viele Moleküle je Box.
- Für die Optik (ADR 0016) zählt vor allem die Massen- bzw. Polarisierbarkeitsverteilung über
  der Oberfläche, nicht die atomare Struktur.

Randbedingungen der Engine:
- **Implizites Lösungsmittel:** Debye-Hückel bzw. GB, kein explizites Wasser, keine PME (ADR 0004).
  Die Box ist in x/y periodisch, in z nicht (ADR 0003), darunter liegen Wand und Bildladungen.
- **Batch-Replicas mit eigener Topologie** (ADR 0005), Fixed-Point-Kräfte (ADR 0002).
- **Vorhandene Bausteine:**
  - harmonische Bindungen;
  - harmonische Winkel in θ;
  - periodische Torsionen;
  - Paarliste mit eigenem σ/ε und abgeschirmter Coulomb-Kraft (heute für 1-4);
  - Nonbonded-Kernel mit voller LJ-Paartabelle (σ_ij, ε_ij) je Typpaar, ohne Mischungsregel;
  - Tile-Paarliste für große Einzelsysteme (ADR 0014).
- **Optik:** α′ und v je Optik-Typ, kalibriert an dn/dc (Phase 10.0). Eine CG-Perle braucht
  dieselben Größen.

Die verbreiteten CG-Modelle passen unterschiedlich gut zu diesen Randbedingungen:
- **MARTINI (2/3)** ist für explizites CG-Wasser parametrisiert, mit Reaction-Field und ε_r = 15.
  Eine implizite Variante gibt es als „Dry MARTINI“ (Arnarez et al., JCTC 2015, für MARTINI 2).
  Ob es für MARTINI 3 und für Nukleinsäuren eine gleichwertige implizite Parametrisierung gibt, ist
  offen (OQ-27). MARTINI-Winkel sind kosinus-harmonisch, MARTINI 3 nutzt zusätzlich virtuelle
  Stellen und Constraints in Ringen.
- **Strukturbasierte Modelle (ENM/Gō, Cα):**
  - eine Perle je Residuum;
  - harmonische Federn zwischen nahen Cα (elastisches Netz);
  - optional attraktive native Kontakte (Gō) und unspezifische Abstoßung.
  - Sie sind von Natur aus implizit und passen direkt auf die vorhandenen Kernel (Bindungen bzw.
    Paarliste).
  - Für Antikörper zählen Domänenbewegungen und die Gelenkregion, keine Faltung; dafür sind sie
    erprobt.
- **DNA-Modelle mit implizitem Salz:**
  - 3SPN.2/3SPN.2C (de Pablo-Gruppe): drei Stellen je Nukleotid, Debye-Hückel, winkelabhängige
    Basenpaarung und Stapelung.
  - oxDNA2: starre Nukleotide mit Orientierung, Debye-Hückel.
  - Beide beschreiben Hybridisierung und Haarnadeln, brauchen aber neue richtungsabhängige
    Wechselwirkungen. oxDNA braucht zusätzlich Rotationsfreiheitsgrade.
  - Die Literaturangaben sind vor der Umsetzung zu verifizieren (OQ-27).

## Entscheidung (Vorschlag)
1. **CG-Rahmen in der Engine, keine eigenen Kraftfeld-Kernel je Modell.** Ein CG-System ist ein
   `SystemSpec` mit Perlen statt Atomen:
   - Masse, Ladung und LJ-Typ je Perle; Bonded-Listen wie atomistisch.
   - Der Builder erzeugt es aus einer Abbildung atomistisch → CG.
   - Neue Kernel nur, wo die vorhandenen nicht reichen (Punkt 4).
2. **Stufe 1 – Antikörper als Cα-Netz mit Gō-Kontakten (ENM/Gō).**
   - Perle je Residuum am Cα. Masse ist die Residuenmasse, Ladung die Residuenladung bei pH 7
     (ASSUMPTION: Lys/Arg +1, Asp/Glu −1, His 0, Termini).
   - Bindungen: Kette als harmonische Bindungen.
   - Elastisches Netz: harmonische Federn für Cα-Paare unter einem Cutoff, innerhalb einer Domäne
     (Domänengrenzen aus der Struktur oder Nutzerangabe). Zwischen den Domänen (Gelenk) gibt es keine
     Federn, sondern native Kontakte, damit die Gelenkbeweglichkeit erhalten bleibt.
   - Native Kontakte zwischen den Domänen als LJ-Paarterme über die vorhandene Paarliste (Gō,
     12-6). Die Paare werden im Nonbonded-Kernel ausgeschlossen.
   - Unspezifische Wechselwirkungen: abstoßendes LJ aus der vollen Paartabelle, Debye-Hückel zwischen
     geladenen Perlen.
   - Wand: 9-3 bzw. 10-4 über die Perlen-LJ-Parameter (ADR 0007); λ kalibrieren (OQ-29).
   - Bildladungen wie atomistisch.
   - Optik: α′ und v je Perle als Summe der kalibrierten Atomwerte der Residuen (Phase 10.0). Damit
     bleiben dn/dc und die Kopplung (ADR 0016) unverändert gültig.
   - Zeitschritt 10–20 fs (Perlenmasse ≈ 110 amu, steifste Feder bestimmt die Grenze; im Test
     bestimmen).
3. **Stufe 2 – Aptamere (DNA) in CG:** Modellwahl offen (OQ-27).
   - Vorschlag: 3SPN.2C, weil es implizites Salz über Debye-Hückel nutzt und ohne
     Rotationsfreiheitsgrade auskommt.
   - Es braucht einen neuen Kernel für die winkelabhängige Basenpaarung und Kreuzstapelung über
     Kandidatenpaare (Nachbarliste, Mehrkörperterm).
   - Bis dahin bleiben Aptamere atomistisch, und gemischte Systeme werden möglich (Punkt 5).
4. **Neue Bausteine (minimal):**
   - (a) Winkelform „kosinus-harmonisch“ ½ k (cos θ − cos θ0)² als zweite Winkelart im
     `BondedForce` (MARTINI, viele ENM-Varianten).
   - (b) Paarliste mit wählbarer Form (12-6 heute; 12-10 für Gō nach Karanicolas-Brooks optional).
   - (c) Builder: `builder.coarse_grain(spec, mapping)` bzw. `builder.cg_from_pdb(pdb, model="enm-go",
     ...)`, Domänen- und Kontaktdefinition, Optiktabellen je Perle.
   - Keine Änderung am Integrator, an den Fixed-Point-Puffern oder an der Replica-Struktur.
5. **Gemischte Auflösung (optional, Stufe 1b):** atomistisches Aptamer und CG-Antikörper in einem
   Kontext.
   - LJ zwischen Atom und Perle über die volle Paartabelle; Debye-Hückel wie gehabt.
   - Physikalisch nur eine Näherung, die Kalibrierung ist offen (OQ-28). Ein gemeinsamer Zeitschritt
     richtet sich nach der atomistischen Komponente.
6. **Explizites MARTINI wird nicht eingebaut.** CG-Wasser widerspricht dem Konzept der impliziten
   Lösung und würde viele Teilchen sowie Reaction-Field-Elektrostatik bringen. Für MARTINI-Systeme
   ist GROMACS das Referenzwerkzeug. Ein Export der Konfigurationen (DCD/HDF5) reicht für den
   Abgleich.

## Alternativen
- **Explizites MARTINI 3 in der Engine:** Es gibt die meisten Parameter, auch für Lipide und
  Linker. Es passt aber nicht zur impliziten Strategie (ADR 0004) und erzeugt viele Wasserperlen,
  also geringen Durchsatz je Replica. Verworfen; GROMACS ist dafür das bessere Werkzeug.
- **Dry MARTINI für Proteine statt ENM/Gō:** Es ist physikalisch allgemeiner (Faltung, Kontakte zu
  anderen Molekülen). Es gibt aber nur MARTINI-2-Parameter, und die Tertiärstruktur von Antikörpern
  braucht ohnehin ein elastisches Netz. Als spätere Stufe möglich, wenn OQ-27 eine geeignete
  Parametrisierung findet.
- **oxDNA2 für Aptamere:** Etabliert für DNA-Nanotechnologie, mit Salzabhängigkeit. Es braucht aber
  Rotationsfreiheitsgrade (Quaternionen, Drehmomente) und damit tiefe Integrator-Änderungen.
  Zurückgestellt.
- **Reine Normalmodenanalyse (ANM) ohne MD:** Sie liefert die Fluktuationen um eine Struktur, aber
  keine Adsorption und keine Bedeckung. Sie dient als Testreferenz (Konsequenzen), nicht als Modell.

## Konsequenzen
- Vorteile:
  - Antikörper und Bedeckungen werden mit geringem Aufwand simulierbar (≈ 1 300 Perlen statt
    ≈ 20 000 Atome, Zeitschritt etwa 5–10-mal größer).
  - Der Rahmen nutzt die vorhandenen Kernel, die Optik bleibt über die Summenregel kalibriert.
- Nachteile:
  - ENM/Gō beschreibt keine Entfaltung auf der Oberfläche.
  - Die Adsorptionsstärke hängt an der Kalibrierung der Wand-λ (OQ-29).
  - Gemischte Auflösung ist nur näherungsweise korrekt.
- Abnahmetests (Vorschlag, ROADMAP Phase 11):
  - **Gegen OpenMM:** `test_cg_forces_vs_openmm`. Ein ENM/Gō-Fab-Fragment mit HarmonicBondForce,
    CustomAngleForce (kosinus-harmonisch), CustomBondForce (Gō) und CustomNonbondedForce (LJ-Tabelle
    + Debye-Hückel). Kräfte max. rel. 1e-4 zur RMS-Kraft, Energie rel. 1e-4.
  - **Gegen die Normalmodenanalyse:** `test_enm_fluctuations_vs_anm`. Für ein reines elastisches
    Netz (ohne Gō, ohne LJ) stimmen die Fluktuationen ⟨Δr_i²⟩ aus der MD mit k_B T · Spur der
    3×3-Blöcke der Pseudoinversen der Hesse-Matrix überein (Korrelation > 0.95, Mittel rel. 5 %).
  - **NVE:** `test_cg_nve` bei 10 bzw. 20 fs, Energiedrift wie atomistisch (≤ 1e-4 relativ über
    100 ps).
  - **Kosinus-harmonische Winkel:** `test_cosine_angle`. Kraft gegen die analytische Ableitung bzw.
    finite Differenzen, auch nahe θ = 0 und π.
  - **Optik:** `test_cg_optics_sum_rule`. α′ und v je Perle reproduzieren dn/dc der atomistischen
    Kalibrierung exakt (Summenregel), und n(z) einer CG-Schicht stimmt mit der atomistischen Schicht
    gleicher Konfiguration bis auf die Verschmierung überein.
  - **Beispiel:** IgG (z. B. PDB 1IGT) über Gold, Adsorption und Orientierung über einige
    Mikrosekunden mit REMD, Durchsatz dokumentiert.
- Neue offene Fragen: OQ-27 (DNA-CG-Modell, implizites MARTINI), OQ-28 (gemischte Auflösung), OQ-29
  (Wandkalibrierung für CG-Perlen).
