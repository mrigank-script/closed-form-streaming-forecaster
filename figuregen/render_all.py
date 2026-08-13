"""figuregen/render_all.py — regenerate every figure section + manifest.

Run from repo root:
  PYTHONPATH=. python figuregen/render_all.py
Sections that need heavy compute (s02 dataset montages) load raw files but are
fast; s06 re-integrates chaos trajectories; s07 reads timing.json (already
produced). manifest.json is rebuilt at the end with the panel totals.
"""

import importlib
import os
import sys


SECTIONS = [
    "figuregen.s01_schematics",
    "figuregen.s02_dataset",
    "figuregen.s03_leaderboard",
    "figuregen.s04_electricity",
    "figuregen.s05_s2_lru",
    "figuregen.s06_chaos",
]


def main():
    for mod in SECTIONS:
        print(f"\n=== {mod} ===")
        try:
            importlib.import_module(mod).main()
        except SystemExit:
            pass
        except Exception:
            import traceback
            traceback.print_exc()

    from figuregen import manifest
    manifest.report()          # writes figures/manifest.json + prints totals


if __name__ == "__main__":
    sys.exit(main())