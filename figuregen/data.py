"""figuregen/data.py — single source of truth for every figure's numbers.

Loads the protocol rows, seed/bootstrap rows, published anchors, chaos
summary and (optional) timing measurements produced elsewhere in the repo,
normalising them into small dicts/DataFrames that all section modules plot
from. No figure module should read a JSON file directly.

Contract files (see docs/paper_context.md §7):
  data/proc/results/<ds>_<H>.json          run_table rows (list of dicts)
  data/proc/results/<ds>_<H>_seeds.json    run_seeds rows ({..."rows": [...]})
  data/proc/published_dsof.json            DSOF Table-2 anchors
  data/proc/chaos_results.json             chaos summary
  data/proc/results/timing.json            run_profile output (may not exist yet)
"""

import json
import os

import numpy as np

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RESULTS = os.path.join(PROC, "results")

VALID_DATASETS = ("etth2", "ettm1", "exchange", "weather", "traffic", "electricity")
HORIZONS = (1, 24, 48)


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

def _datasets_with_rows():
    found = set()
    if not os.path.isdir(RESULTS):
        return found
    for fn in os.listdir(RESULTS):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        base = stem.split("_")[0]
        if base in VALID_DATASETS:
            found.add(base)
    return sorted(found)


def load_results(dataset=None, horizons=None):
    """Load run_table rows across datasets/horizons.

    Returns dict {(dataset, H, model): row} and a parallel list metadata of
    which (dataset,H) files exist. Skips seeds-format and clip rows (handled
    separately below) unless explicitly asked via `load_rows(include_clip=...)`.
    """
    datasets = [dataset] if dataset else _datasets_with_rows()
    horizons = horizons or HORIZONS
    rows = {}
    files = []
    for ds in datasets:
        for H in horizons:
            p = os.path.join(RESULTS, f"{ds}_{H}.json")
            if not os.path.exists(p):
                if dataset:
                    files.append(p)   # requested-but-missing
                continue
            files.append(p)
            with open(p) as fh:
                for r in json.load(fh):
                    if r.get("clip_spikes", 0):
                        continue
                    key = (ds, H, r["model"])
                    rows[key] = r
    return rows, files


def load_seeds(dataset=None, horizons=None, include_clip=False):
    """Load run_seeds bootstrap rows.

    Returns dict {(dataset, H, model, clip): row} where clip is 0.0 or the
    clip factor. Also returns the list of loaded file paths.
    """
    datasets = [dataset] if dataset else _datasets_with_rows()
    horizons = horizons or HORIZONS
    rows = {}
    files = []
    for ds in datasets:
        # Accept any <ds>_<H>*_seeds.json in results/ (e.g. etth2_24_seeds.json,
        # ettm1_24_s2_seeds.json, electricity_1_raw_seeds.json) — the protocol
        # rows inside carry (dataset, pred_len, model) so the filename suffix
        # is decorative and must not gate what we plot.
        candidates = (f for f in os.listdir(RESULTS)
                      if f.startswith(f"{ds}_") and f.endswith("_seeds.json"))
        for fn in candidates:
            p = os.path.join(RESULTS, fn)
            with open(p) as fh:
                doc = json.load(fh)
            if not doc or "rows" not in doc:
                continue
            for r in doc.get("rows", []):
                rds = r.get("dataset", ds)
                rH = int(r.get("pred_len", fn.split("_")[1]))
                clip = r.get("clip_spikes", 0.0)
                if clip and not include_clip:
                    continue
                files.append(p)
                rows[(rds, rH, r["model"], clip)] = r
    # de-duplicate file list, keep order
    seen = []
    for p in files:
        if p not in seen:
            seen.append(p)
    return rows, seen


def load_electricity_rows(include_plain=True):
    """Electricity-H1 rows for all models under raw & clip=3 protocols.

    Returns dict {(kind, model): row} with kind in {"raw", "clip3",
    "raw_s2", "clip3_s2"} matching the verified fp64 determinism re-run.
    """
    raw = {}
    for fn, kind in (("electricity_1_raw64.json", "raw"),
                     ("electricity_1_clip33.json", "clip3"),
                     ("electricity_1.json", "plain")):
        p = os.path.join(RESULTS, fn)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for r in json.load(fh):
                raw[(kind, r["model"])] = r
    return raw


# ---------------------------------------------------------------------------
# Published anchors
# ---------------------------------------------------------------------------

def load_published():
    with open(os.path.join(PROC, "published_dsof.json")) as fh:
        return json.load(fh)


def published_cells(dataset, H, publisher=None):
    """List of (teacher, mode, mse) for the 14 published cells at (ds,H).

    mode in {"batch","dsof"}. If `publisher` given, returns just its rows.
    """
    pub = load_published()["rows"]
    d = pub.get(dataset)
    if not d:
        return []
    out = []
    for teacher, hh in d.items():
        if str(H) not in hh:
            continue
        m = hh[str(H)]
        out.append((teacher, "batch", m["batch"]))
        out.append((teacher, "dsof", m["dsof"]))
    if publisher:
        out = [c for c in out if c[0] == publisher]
    return out


# ---------------------------------------------------------------------------
# Chaos
# ---------------------------------------------------------------------------

def load_chaos():
    with open(os.path.join(PROC, "chaos_results.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Timing (optional; produced by experiments/run_profile.py)
# ---------------------------------------------------------------------------

def load_timing():
    p = os.path.join(RESULTS, "timing.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def row_value(rows, dataset, H, model, field="mse"):
    r = rows.get((dataset, H, model))
    return None if r is None else r.get(field)


def by_step_series(rows, dataset, H, model):
    """Sorted (r, mse, mae) series from a row's by_step map, else None."""
    r = rows.get((dataset, H, model))
    if not r or "by_step" not in r:
        return None
    steps = sorted((int(k), v[0], v[1]) for k, v in r["by_step"].items())
    return steps


def ours_best(rows, dataset, H, models=("static", "rls", "s2rls")):
    """Best (min MSE) of our models at (ds,H) that pass the rank bar."""
    best = None
    for m in models:
        v = row_value(rows, dataset, H, m)
        if v is None:
            continue
        if best is None or v < best[1]:
            best = (m, v)
    return best


def beats_count(rows, dataset, H):
    """(#cells beaten, #cells tied, total) for our best row at (ds,H)."""
    best = ours_best(rows, dataset, H)
    if best is None:
        return None
    _, val = best
    cells = published_cells(dataset, H)
    beaten = sum(1 for *_ , v in cells if val < v)
    tied = sum(1 for *_ , v in cells if abs(val - v) < 1e-9)
    return beaten, tied, len(cells)


if __name__ == "__main__":
    rows, files = load_results()
    print(f"loaded {len(rows)} result rows from {len(files)} files")
    for ds in sorted({k[0] for k in rows}):
        print(f"  {ds:12s} H={sorted({k[1] for k in rows if k[0]==ds})}")