#!/usr/bin/env python3
"""Automated Kaggle dataset downloader for Enveda CASMI 2026."""

import os
import subprocess
import sys
from pathlib import Path

COMPETITION_SLUG = "enveda-CASMI26-molecule-id-mass-spectra"


def main():
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading dataset for competition: {COMPETITION_SLUG}")
    print(f"[*] Target directory: {raw_dir}")

    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION_SLUG, "-p", str(raw_dir)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print("[+] Download successful. Extracting files...")
            import zipfile
            for z in raw_dir.glob("*.zip"):
                with zipfile.ZipFile(z, "r") as zip_ref:
                    zip_ref.extractall(raw_dir)
                print(f"[+] Extracted: {z.name}")
        else:
            print("[!] Kaggle CLI reported an issue:")
            print(res.stderr or res.stdout)
            print("[*] Note: Ensure ~/.kaggle/kaggle.json exists with valid credentials,")
            print("    or accept the competition rules at:")
            print(f"    https://www.kaggle.com/competitions/{COMPETITION_SLUG}")
    except FileNotFoundError:
        print("[!] Kaggle CLI not found on PATH. Run 'pip install kaggle'.")


if __name__ == "__main__":
    main()
