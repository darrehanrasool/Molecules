# Data Dictionary

| Field Name | Type | Unit / Format | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `molecule_id` | String | Identifier | Unique token identifying the test or train spectrum | `m_0014ef` |
| `precursor_mz` | Float | $m/z$ (Da) | Precursor ion mass-to-charge ratio measured by instrument | `255.0652` |
| `adduct` | String | IUPAC/MS Notation | Ion adduct acquired during ionization mode | `[M-H]-` |
| `ion_mode` | String | Categorical | Positive (`POS`) or Negative (`NEG`) ionization mode | `NEG` |
| `collision_energy` | Float | Electronvolts (eV) | Nominal fragmentation collision energy | `50.0` |
| `peaks` | List[Tuple] | `[(m/z, intensity)]` | Normalized or raw fragment peaks | `[(153.018, 0.42), (254.057, 1.0)]` |
| `smiles` | String | SMILES | Ground truth or predicted 2D chemical structure | `OC[C@H]1OC(O)...` |
| `inchikey14` | String | 14-char hash | First 14 characters of canonicalized InChIKey | `WQZGKKKJIJFFOK` |
| `neutral_mass` | Float | Daltons (Da) | Calculated uncharged monoisotopic mass | `254.0579` |
