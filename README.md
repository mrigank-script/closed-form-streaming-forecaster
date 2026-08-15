# NormaStream — Closed-Form Long-Horizon Time-Series Forecasting

NormaStream is a study of whether streaming time-series forecasting can be
done effectively **without backpropagation or continual gradient-based
training**. The method initializes the model with closed-form ridge
regression, then adapts it one observation at a time with Recursive Least
Squares (RLS) via Woodbury updates. An optional fixed complex-valued LRU
reservoir — **NormaRC** — provides longer temporal memory.

The core solves ridge regression **exactly in a single algebraic step** (no
SGD, no learning rate, no weight-seed randomness), so the forecast is fully
deterministic and the uncertainty you report is the honest test-time sampling
variability, not optimizer luck.

The repo reproduces every number and every figure in the paper, from raw
official datasets down to the rendering.

---

## What the engine does

| State | Features | Update | Description |
|-------|----------|--------|-------------|
| `static` | lag + Fourier + trend | closed-form ridge fit once on TRAIN | DLinear-class solver, `core/ridge_solver.py` |
| `rls`    | same | online rank-1 Woodbury per timestep | memory-efficient online RLS, `core/woodbury.py` |
| `s2`     | `static` + LRU context | — | complex-diagonal LRU reservoir, `core/lru.py` (O(log L) associative scan) |
| `s2rls`  | `static` + LRU context | online Woodbury | RLS over the LRU-augmented feature space |

All four share the same protocol-safe pipeline (`experiments/forecaster.py`):
per-channel normalization statistics from TRAIN only, chronological
20%/5%/75% train/val/test split, and evaluation without information leakage
— the exact protocol of the online-forecasting literature (OneNet, NeurIPS
2023; DSOF, ICLR 2025).

### Headline results
- **Leaderboard**: against the published batch/dsof cells of seven methods on
  ETT h2 / ETTm1 / Exchange / Weather / Traffic, our `rls` row is **rank #1 on
  Exchange-H24**, **rank #2 on 9 of 12 (dataset, H) cells**, mean rank ≈ 2.7 / 15.
- **Electricity case study**: the published "broken-meter" spikes dominate the
  raw baseline's MSE (persistence floor 6.75 ≫ every published row). Our
  `s2rls` damps the spike blow-up **723.7 → 21.6 MSE (−97%)** while keeping MAE
  nearly constant; on the corrected protocol our MSE (0.0675) is ~30× below the
  best published row. Raw rows are honestly reported as *not rank-comparable*
  in `figures/04_electricity/`.
- **S2/LRU on clean data**: matches `rls` within block-bootstrap noise across
  datasets (no regression, no invented win) — see `figures/05_s2_lru/`.
- **Chaos track**: the same closed-form core applied to Lorenz-96 and
  NARMA-10/30 (`experiments/chaos.py`, `figures/06_chaos/`).

---

## Repository layout

```
core/                 Closed-form math: covariance, ridge solve, Woodbury RLS,
                      LRU reservoir, Hopfield working memory, IA objective.
experiments/          Protocol harness: data loaders, eval_protocol, run_table,
                      run_seeds (bootstrap SE), compare_table (vs published).
figuregen/            All paper figures (s01..s06) + shared style + manifest.
figures/              Rendered main figures; per-section _appendix/ archives
                      every per-(ds,H) variant. figures/manifest.json = counts.
data/proc/            ACTUAL RESULT JSONs (source of truth, committed):
                        results/<ds>_<H>.json          protocol rows (per-model)
                        results/<ds>_<H>_seeds.json    block-bootstrap SE rows
                        published_dsof.json            published Table-2 anchors
                        chaos_results.json             Lorenz-96/NARMA summaries
docs/paper_context.md Reproducibility narrative + honest caveats.
tests/                Unit tests (core numerics + protocol).
upload/               PNG-only mirror of the main figures (subfolders per
                      section) for quick online upload; gitignored, rebuilt
                      by scripts/copy_upload.py.
```

**Data policy.** Raw datasets are *not* committed (they are large and official).
`experiments/data.py` auto-downloads them on first use from their canonical
public sources:
ECL / Traffic / Exchange → `laiguokun/multivariate-time-series-data`,
ETT → `zhouhaoyi/ETDataset`, Weather → `thuml/Time-Series-Library`.

---

## Install & run

```bash
pip install -r requirements.txt   # jax + jaxlib, numpy/scipy, scikit-learn, matplotlib, pytest
```

The GPU run wrapper (`run.sh`) sets a memory-stabilising allocator
(`XLA_PYTHON_CLIENT_MEM_FRACTION` etc.) so a 6 GB laptop GPU does not OOM on
small ops; it forwards to `<your python> -m <module>`. Adjust the `PY` path for
your environment, or invoke the modules directly:

```bash
# 1. Reproduce protocol rows (static, rls, s2, s2rls) for a dataset
python -m experiments.run_table --dataset etth2 --pred_len 24 \
    --models static rls --out data/proc/results/etth2_24.json

# 2. Add block-bootstrap uncertainty rows
python -m experiments.run_seeds --dataset etth2 --pred_len 24 \
    --models rls --out data/proc/results/etth2_24_seeds.json

# 3. Compare our rows against published DSOF Table-2 cells
python -m experiments.compare_table

# 4. Regenerate every figure + manifest (fast)
python -m figuregen.render_all

# 5. Tests
python -m pytest tests -q
```

All steps are idempotent: the committed `data/proc` JSONs are the ground truth,
and the figure suite re-renders purely from them, so the paper's numbers never
drift from the code.

### Quick figure upload copy

```bash
PYTHONPATH=. python scripts/copy_upload.py   # rebuild upload/ from figures/
```

---

## Reproducibility notes (honest by construction)

- The head is a **closed-form linear solve** → deterministic across devices;
  no seed-seed averaging is needed or claimed.
- Uncertainty = **block bootstrap over test chunks**, reported as
  [`p5`, `p95`] / SE (see `experiments/run_seeds.py`). This is what we plot on
  our bars and what you should quote in the paper.
- Published rows are single reported values (the leaderboard does not publish
  std) — the main comparison table therefore shows published points without
  error bars, and our rows with their bootstrap intervals.
- Electricity raw rows are **not** rank-valid (spike artifact). They are kept
  strictly in `figures/04_electricity/` as a case study, never in the
  leaderboard table.

---

## License

MIT — Copyright (c) 2026 Mrigank. You are free to use, modify, and distribute
this code (including commercially) provided you retain the copyright notice.
See [LICENSE](LICENSE).
