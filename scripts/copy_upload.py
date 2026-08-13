#!/usr/bin/env python
"""Recreate upload/ — a PNG-only mirror of the main figures for quick upload.

Copies every top-level .png from the six main figure sections into
upload/<section>/ (no JSON, no _appendix). figures/ stays the source of truth.
Intended for ''.

Run:  PYTHONPATH=. python scripts/copy_upload.py
"""

import os
import shutil

SRC = os.path.join(os.path.dirname(__file__), "..", "figures")
DST = os.path.join(os.path.dirname(__file__), "..", "upload")
SECTIONS = ["01_schematics", "02_dataset", "03_leaderboard",
            "04_electricity", "05_s2_lru", "06_chaos"]


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST, exist_ok=True)
    total = 0
    for sec in SECTIONS:
        sdir = os.path.join(SRC, sec)
        if not os.path.isdir(sdir):
            continue
        ddir = os.path.join(DST, sec)
        os.makedirs(ddir, exist_ok=True)
        n = 0
        for f in sorted(os.listdir(sdir)):
            if not f.endswith(".png"):
                continue
            shutil.copy2(os.path.join(sdir, f), os.path.join(ddir, f))
            n += 1
        total += n
        print(f"  {sec}: {n} png")
    print(f"copied {total} png files (no json, no appendix) -> {DST}")


if __name__ == "__main__":
    main()