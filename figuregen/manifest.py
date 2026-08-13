"""Build the figure manifest: PNG count + panel totals per section.

Each multi-panel figure counts as one image per subplot (a 15-panel figure =
15 panels; a standalone plot = 1). Sections write <name>.meta.json next to
each figure with a deterministic panel count; figures without metadata fall
back to 1. Aggregates into figures/manifest.json.

Usage:  PYTHONPATH=. python figuregen/manifest.py [--rebuild]
"""

import json
import os

FIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "figures")


def _count_panels(png_path):
    """Best-effort panel count: a .meta.json next to the png, else 1."""
    meta = png_path[:-4] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            d = json.load(fh)
        n = d.get("panels")
        if isinstance(n, int) and n >= 1:
            return n, d
    return 1, {}


def build():
    manifest = {
        "count_policy": "panels (a multi-panel png counts each panel as one "
                        "image; a single-plot png counts as 1)",
        "root": os.path.abspath(FIG_ROOT),
        "sections": {},
        "totals": {"png_files": 0, "panels": 0},
    }
    for sec in sorted(os.listdir(FIG_ROOT)):
        sdir = os.path.join(FIG_ROOT, sec)
        if not os.path.isdir(sdir):
            continue
        pngs = sorted(f for f in os.listdir(sdir) if f.endswith(".png"))
        entries = []
        sec_panels = 0
        for f in pngs:
            n, meta = _count_panels(os.path.join(sdir, f))
            sec_panels += n
            entries.append({"file": f, "panels": n, "meta": meta})
        manifest["sections"][sec] = {
            "png_files": len(pngs),
            "panels": sec_panels,
            "files": entries,
        }
        manifest["totals"]["png_files"] += len(pngs)
        manifest["totals"]["panels"] += sec_panels
    out = os.path.join(FIG_ROOT, "manifest.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def report():
    m = build()
    t = m["totals"]
    print(f"total PNG files : {t['png_files']}")
    print(f"total PANELS    : {t['panels']}")
    for sec, s in m["sections"].items():
        print(f"   {sec:22s} {s['panels']:4d} panels in {s['png_files']:3d} png")
    print("main figures only; appendix files archived under <sec>/_appendix/")
    return m


if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv:
        report()
    else:
        report()