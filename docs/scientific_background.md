# Scientific Background: Physics & Chemistry of Tandem Mass Spectrometry (LC-MS/MS)

## 1. Mass Spectrometry Operating Principles

Mass spectrometry (MS) is an analytical technique that ionizes chemical species and sorts ions based on their mass-to-charge ratio ($m/z$).

### Ionization Modes & Adducts
Molecules must be charged to be manipulated by electric and magnetic fields. In Liquid Chromatography-Tandem Mass Spectrometry (LC-MS/MS):
- **Positive Electrospray Ionization ($ESI^+$)**:
  - $[M + H]^+$: Addition of a proton ($m/z = M + 1.007276$)
  - $[M + NH_4]^+$: Addition of ammonium ($m/z = M + 18.033823$)
  - $[M + Na]^+$: Addition of sodium ion ($m/z = M + 22.989218$)
  - $[M + K]^+$: Addition of potassium ion ($m/z = M + 38.963158$)
- **Negative Electrospray Ionization ($ESI^-$)**:
  - $[M - H]^-$: Loss of a proton ($m/z = M - 1.007276$)
  - $[M + HCOO]^-$: Formate adduct ($m/z = M + 44.998201$)
  - $[M + CH_3COO]^-$: Acetate adduct ($m/z = M + 59.013851$)
  - $[M + Cl]^-$: Chlorine adduct ($m/z = M + 34.969402$)

### Precursor Mass Resolution & Mass Accuracy
Modern high-resolution mass spectrometers (e.g., Orbitrap, timsTOF, Q-TOF) achieve parts-per-million (ppm) mass accuracy:
$$\Delta m (\text{ppm}) = \frac{|m_{\text{theoretical}} - m_{\text{observed}}|}{m_{\text{theoretical}}} \times 10^6$$
An accuracy of $<5\,\text{ppm}$ severely constrains possible molecular formulas (Seven Golden Rules of elemental composition).

---

## 2. Collision-Induced Dissociation (CID) & Fragmentation

During tandem MS (MS/MS or $MS^2$):
1. **Precursor Isolation**: Quadrupole filters out all ions except the narrow window around the precursor $m/z$.
2. **Collision Cell**: The precursor ion is accelerated into an inert collision gas (usually $N_2$ or $Ar$). Inelastic collisions convert kinetic energy into internal vibrational energy.
3. **Unimolecular Dissociation**: When vibrational energy exceeds bond dissociation energy, chemical bonds cleave according to predictable fragmentation chemistry (e.g., McLafferty rearrangement, retro-Diels-Alder, alpha-cleavage).
4. **Fragment Detection**: The $m/z$ and abundance of each fragment ion are recorded, producing the MS/MS spectrum.

### Spectrum Anatomy
- **Base Peak**: The fragment with highest relative intensity (assigned 100% or 1.0).
- **Precursor Peak**: The unfragmented parent ion peak.
- **Diagnostic Ions**: Specific $m/z$ peaks that indicate distinct functional groups (e.g., $m/z = 91.0548$ for tropylium ion $[C_7H_7]^+$, $m/z = 105.0335$ for benzoyl ion $[C_7H_5O]^+$).
- **Neutral Losses**: Difference between precursor mass and fragment mass, representing uncharged departing molecules (e.g., $-18.0106\,\text{Da}$ for $H_2O$, $-43.9898\,\text{Da}$ for $CO_2$, $-27.9949\,\text{Da}$ for $CO$).
