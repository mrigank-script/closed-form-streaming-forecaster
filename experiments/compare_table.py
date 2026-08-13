"""Compare our protocol rows against published DSOF Table 2 cells.

Reads our rows from data/proc/results/<dataset>_<H>.json (run_table --out)
and the published anchors from data/proc/published_dsof.json, then prints a
per-cell ranking: which published teacher rows we beat/tie/lose to.

Usage:  ./run.sh experiments.compare_table
"""

import glob
import json
import os

DATA_PROC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "proc")
ROWS_DIR = os.path.join(DATA_PROC, "results")
PUB_FILE = os.path.join(DATA_PROC, "published_dsof.json")


def _load_rows():
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROWS_DIR, "*.json"))):
        fn = os.path.basename(f)
        try:
            with open(f) as fh:
                l = json.load(fh)
        except Exception as e:
            print(f"[compare_table] skip {fn}: {e}")
            continue
        if isinstance(l, dict) and "rows" in l:
            l = l["rows"]                    # run_seeds output
        if not l or not isinstance(l, list):
            continue
        ds = l[0].get("dataset")
        pl = l[0].get("pred_len")
        for r in l:
            if r.get("model") is None:
                continue
            key = (ds, pl, r["model"])
            # seeds rows (dict-with-rows) lack 'n' and would clobber the real
            # run_table row; keep whichever carries the tested count.
            if r.get("n") is not None or key not in rows or rows[key].get("n") is None:
                rows[key] = r
    return rows


def _load_pub():
    with open(PUB_FILE) as f:
        return json.load(f)


def main():
    ours = _load_rows()
    pub = _load_pub()["rows"]
    src = _load_pub()["source"]

    skipped = 0
    for (ds, pl, model), r in sorted(ours.items()):
        d = pub.get(ds)
        ws = ours.get((ds, pl, "static"))

        tested = r.get("n", 0) > 0
        if not tested:
            print(f"{ds} H{pl} {model}: untested")
            continue

        if model == "rls" and ws is not None:
            lab = f"Ours-RLS {r['mse']:.4f}"
        else:
            lab = f"Ours-static {r['mse']:.4f}" if model == "static" else f"Ours {r['mse']:.4f}"

        if d is None or str(pl) not in d.get("DLinear", {}):
            continue

        # build per-teacher verdict vs best public method value on this H
        allvals = []
        for teacher, hh in d.items():
            if str(pl) in hh:
                m = hh[str(pl)]
                allvals.append((teacher, "batch", m["batch"]))
                allvals.append((teacher, "dsof", m["dsof"]))
        if not allvals:
            continue
        best_teacher, best_mode, best = min(allvals, key=lambda t: t[2])
        n_beaten = sum(1 for *_ , v in allvals if r["mse"] < v)
        n_tie = sum(1 for *_ , v in allvals if abs(r["mse"] - v) < 1e-9)
        total = len(allvals)

        # Rows computed on spike-corrected data (clip_spikes>0) are NOT directly
        # comparable to the published rows (same DSOF-exact raw target file).
        # S2/S2RLS on the untouched data rank normally.
        if r.get("clip_spikes", 0) > 0:
            print(f"{ds:11s} H{pl:<3} {lab}  [corrected-data {r['clip_spikes']}x] "
                  f"pub-best-row={best_teacher}:{best_mode}={best}  "
                  f"ours={r['mse']:.4f}  (not in leaderboard: different loss surface)")
            continue

        print(f"{ds:11s} H{pl:<3} {lab}  vs DSOF-pub (n={len(allvals)}) "
              f"beats {n_beaten}/{total}  best-pub={best_teacher}:{best_mode}={best} "
              f"({n_beaten + n_tie}/{total} ≤ Ours)")
        skipped += 0

    print()
    print("Published source:", src["venue"], "-", src["title"])
    print("  ", src["url"])


if __name__ == "__main__":
    main()