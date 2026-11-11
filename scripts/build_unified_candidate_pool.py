#!/usr/bin/env python3
"""
Build Unified High-Coverage Candidate Pool for CASMI 2026.
Merges:
1. CASMI Training Set (275,810 molecules from data/processed/fp_database.npz)
2. COCONUT 2.0 Natural Products (738,794 molecules from data/external/coconut_csv_lite.zip)

Deduplicates by InChIKey14, sorts ascending by exact neutral mass,
and saves ready-to-upload candidate pools in kaggle_upload/ and data/processed/.
"""

import csv
import io
import os
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent.parent
CASMI_FP_PATH = ROOT / "data" / "processed" / "fp_database.npz"
CASMI_MASS_PATH = ROOT / "data" / "processed" / "candidate_masses.npy"
COCONUT_ZIP_PATH = ROOT / "data" / "external" / "coconut_csv_lite.zip"
OUT_DIR = ROOT / "kaggle_upload"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("BUILDING UNIFIED NATURAL PRODUCT CANDIDATE POOL (COCONUT + CASMI)")
    print("=" * 70)

    # 1. Load CASMI training pool
    print(f"[*] Step 1: Loading CASMI training pool from {CASMI_FP_PATH.name}...")
    t0 = time.time()
    with np.load(CASMI_FP_PATH, allow_pickle=True) as data:
        casmi_fps = data['fps']
        casmi_smiles = data['smiles']
        casmi_ik14 = data['inchikey14']
    casmi_masses = np.load(CASMI_MASS_PATH)

    print(f"    Loaded {len(casmi_smiles):,} CASMI molecules in {time.time()-t0:.1f}s.")

    # Track seen InChIKey14 for deduplication
    seen_ik14 = set(casmi_ik14)
    print(f"    Unique CASMI planar skeletons: {len(seen_ik14):,}")

    # Prepare lists for new COCONUT additions
    new_masses = []
    new_smiles = []
    new_ik14 = []
    new_fps = []

    # 2. Process COCONUT
    print(f"\n[*] Step 2: Streaming and fingerprinting COCONUT from {COCONUT_ZIP_PATH.name}...")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    t_start = time.time()
    total_coconut_rows = 0
    added_coconut = 0
    skipped_duplicates = 0
    invalid_rows = 0

    with zipfile.ZipFile(COCONUT_ZIP_PATH) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'))
            for row in reader:
                total_coconut_rows += 1
                smi = row.get('canonical_smiles')
                mass_str = row.get('exact_molecular_weight')
                ik = row.get('standard_inchi_key')

                if not smi or not mass_str:
                    invalid_rows += 1
                    continue

                try:
                    mass = float(mass_str)
                except ValueError:
                    invalid_rows += 1
                    continue

                # Filter extreme mass out of MS range (50 - 1200 Da)
                if mass < 50.0 or mass > 1200.0:
                    continue

                ik14 = ik[:14] if ik else None
                if ik14 and ik14 in seen_ik14:
                    skipped_duplicates += 1
                    continue

                m = Chem.MolFromSmiles(smi)
                if m is None:
                    invalid_rows += 1
                    continue

                # If InChIKey wasn't in CSV or incomplete, derive planar key
                if not ik14:
                    try:
                        ik14 = Chem.MolToInchiKey(m)[:14]
                    except Exception:
                        ik14 = smi[:14]

                if ik14 in seen_ik14:
                    skipped_duplicates += 1
                    continue

                fp = gen.GetFingerprintAsNumPy(m)

                new_masses.append(mass)
                new_smiles.append(smi)
                new_ik14.append(ik14)
                new_fps.append(fp)
                seen_ik14.add(ik14)
                added_coconut += 1

                if added_coconut % 50000 == 0:
                    elapsed = time.time() - t_start
                    rate = added_coconut / elapsed
                    print(f"    Processed {total_coconut_rows:,} rows | Added {added_coconut:,} novel natural products ({rate:.0f} mols/s)...")

    print(f"\n[+] COCONUT parsing complete:")
    print(f"    Total rows read:       {total_coconut_rows:,}")
    print(f"    Novel natural added:   {added_coconut:,}")
    print(f"    Duplicates skipped:    {skipped_duplicates:,}")
    print(f"    Invalid rows skipped:  {invalid_rows:,}")
    print(f"    Processing time:       {time.time()-t_start:.1f}s")

    # 3. Concatenate and sort ascending by exact mass
    print(f"\n[*] Step 3: Merging CASMI ({len(casmi_masses):,}) and COCONUT ({added_coconut:,})...")
    all_masses = np.concatenate([casmi_masses.astype(np.float32), np.array(new_masses, dtype=np.float32)])
    all_smiles = np.concatenate([casmi_smiles, np.array(new_smiles, dtype=object)])
    all_ik14 = np.concatenate([casmi_ik14, np.array(new_ik14, dtype=object)])

    # Convert new_fps to numpy uint8 matrix
    new_fps_arr = np.array(new_fps, dtype=np.uint8)
    all_fps = np.vstack([casmi_fps, new_fps_arr])

    total_pool_size = len(all_masses)
    print(f"    Combined pool size: {total_pool_size:,} candidate molecules.")

    print(f"[*] Step 4: Sorting candidate pool ascending by exact neutral mass...")
    t_sort = time.time()
    sort_idx = np.argsort(all_masses)

    all_masses = all_masses[sort_idx]
    all_smiles = all_smiles[sort_idx]
    all_ik14 = all_ik14[sort_idx]
    all_fps = all_fps[sort_idx]
    print(f"    Sorted {total_pool_size:,} candidates in {time.time()-t_sort:.1f}s.")

    # 4. Save to kaggle_upload/
    print(f"\n[*] Step 5: Exporting compressed candidate pool to {OUT_DIR}...")
    t_save = time.time()

    # Save candidate_masses.npy
    masses_file = OUT_DIR / "candidate_masses.npy"
    np.save(masses_file, all_masses)
    print(f"    Saved: {masses_file} ({masses_file.stat().st_size / 1e6:.1f} MB)")

    # Save fp_database.npz (directly compatible with existing notebook)
    fp_db_file = OUT_DIR / "fp_database.npz"
    np.savez_compressed(
        fp_db_file,
        fps=all_fps,
        smiles=all_smiles,
        inchikey14=all_ik14,
        masses=all_masses
    )
    print(f"    Saved: {fp_db_file} ({fp_db_file.stat().st_size / 1e6:.1f} MB)")

    # Also save as candidate_pool.npz
    pool_file = OUT_DIR / "candidate_pool.npz"
    if pool_file != fp_db_file:
        import shutil
        shutil.copy2(fp_db_file, pool_file)
        print(f"    Saved: {pool_file} (dual-name alias)")

    # Also copy to data/processed for local use
    shutil.copy2(masses_file, ROOT / "data" / "processed" / "candidate_masses.npy")
    shutil.copy2(fp_db_file, ROOT / "data" / "processed" / "fp_database.npz")
    print(f"    Synced to data/processed/")

    print("\n" + "=" * 70)
    print(f"SUCCESS: UNIFIED CANDIDATE POOL BUILT ({total_pool_size:,} MOLECULES)")
    print(f"Upload Directory: {OUT_DIR.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
