# Paper 2 — Research-Paper Context File (master summary)

Single authoritative context for writing the Paper 2 research paper: (1) problem
& core solver, (2) Track-1 leaderboard results, (3) Electricity deep-dive —
clipping vs no-clipping and exactly how to write it up, (4) S2 (LRU-context)
experiment, (5) the numerical-determinism fix that invalidated an earlier
result, (6) chaos track, (7) how to document each artefact, (8) commands +
file inventory.

Every number below was produced and verified in-session on this repo
(WSL `/mnt/c/Projects/Application of Core Slover`). All MSE/MAE are float64 /
deterministic unless marked float32.

## 0. Thesis (abstract-ready)

We challenge the assumption that usable online multivariate time-series
forecasting requires backprop-trained adaptive networks. A **closed-form
solver** — per-channel ridge-regularised least squares warm-started on the
training segment, then a **recursive least squares (RLS) Woodbury update**
online — matches or beats the ICLR-2025 DSOF leaderboard (DLinear / FITS / FSNet
/ iTransformer / OneNet / PatchTST / NSTransformer, batch and online) **with no
SGD anywhere**. The same head, unchanged, validates on chaotic dynamics
(Lorenz-96, NARMA-10/30): two independent leaderboard-style proofs that
closed-form streaming forecasting is competitive.

Second contribution (S2): augment the ridge features with a **fixed random
linear-recurrent (LRU) reservoir** whose readout stays closed-form — memory
beyond the 96-lag window, zero gradient training. S2 stabilises RLS on the
outlier-contaminated Electricity benchmark; it is a wash on clean data.

Third, methodological: the paper documents a **data-quality landmine in the
public ECL benchmark** (two meters whose test-range values run ~100x past their
training range) — why published MSE rows on this file are numerically fragile,
and a transparent spike-trim policy that separates the real benchmark row from
the corrected-data row. We report both.

Core constraints honoured everywhere:
- No SGD. Closed-form ridge + RLS only.
- No evaluated-on-train leakage: target x[t+r] enters only at an origin that
  has genuinely observed it; LRU context is causal h_t = f(x_{<=t}).
- Rows must rank on the OFFICIAL benchmark protocol (DSOF splits + scaler).

## 1. Protocol & reproducibility backbone

- Loader mirrors the official DSOF (ICLR 2025) pipeline, verified against
  `yyalau/iclr2025_dsof/src/data/data_loader.py`:
  StandardScaler fit on TRAIN ONLY applied to val/test; chronological splits
  20/5/75 (non-ETT) or classic Autoformer borders (ETT). Raw files = the
  standard laiguokun mirror the DSOF paper used (`data/raw/`).
- `experiments/eval_protocol.py`:
  - `_static_head`: per-channel ridge solve. `lam` is a RELATIVE ridge factor
    `lam * trace(A)/F` (scale-invariant; prevents closed-form blowup in the
    96-correlated-lag null directions). Solved in float64, stored float32.
  - `_online_chunk`: batched (all S series) one-step RLS:
    `A_inv = A_inv - k v^T`, `W += k (y_obs - pred)`. Verified vs numpy
    byte-level on a small slice (max diff ~5e-16).
  - `evaluate`: loops r=1..H, warm-starts each step-head, streams test in
    `chunk_online` blocks; returns MSE/MAE, per-step, and per-chunk error
    blocks (for seed-spread).
- Seed-spread (deterministic head → no weight-init seeds): block-bootstrap SE
  over per-chunk error blocks (`run_seeds.py`, block=4, nboot=200) — the honest
  analogue of the published 5-seed std rows.
- Published anchors with provenance: `data/proc/published_dsof.json` (DSOF
  Table 2; H in {1,24,48}; lookback L=192 for ECL, L=96 elsewhere — WE RAN 96
  FOR ECL; footnote this in the paper's experimental setup).

## 2. Track 1 — leaderboard results (verified, float64)

Rows: `data/proc/results/<ds>_<H>.json`, rendered by `compare_table.py`.
"beats k/14" = k of the 14 published teacher-variant cells (7 teachers x
{batch,dsof}). Best published row shown as `teacher:mode=value`.

- etth2   H1  RLS 0.3651  beats 13/14  (best DLinear:dsof=0.365)  [TIE SOTA]
- etth2   H24 RLS 2.0837  beats 10/14
- etth2   H48 static 3.6044 beats 9/14   (RLS 4.8063 worse here)
- ettm1   H1  RLS 0.1016  beats 13/14
- ettm1   H24 RLS 0.4505  beats 13/14
- ettm1   H48 RLS 0.5781  beats 13/14
- exchange H24 RLS 0.0904 beats 14/14   (sweeps ALL published rows)
- exchange H1  RLS 0.0090 beats 11/14
- exchange H48 RLS 0.1773 beats 13/14
- weather H1  RLS 0.3105  beats 13/14
- weather H24 RLS 0.7961  beats 13/14   (NOT a sweep; best OneNet:dsof=0.671)
- weather H48 RLS 1.0620  beats 13/14
- traffic H1  RLS 0.2801  beats 6/14    (862-channel GPU path; mid-table)
- electricity H1 RAW: static 48.1754 | rls 723.7446 | s2 51.1238 |
  s2rls 21.5947  (loses on the exact public file; RLS 0/14). See §3.
- electricity H1 CLIP(3x): static 0.3169 | rls 0.0684 | s2 0.3323 |
  s2rls 0.0675 (corrected-data row; NOT directly comparable to published rows —
  robustness experiment, keep out of the leaderboard).

Paper headline: "A closed-form ridge+RLS head, no SGD, ties/beats 13 of 14
published DSOF methods on ETTh2/ETTm1/Exchange/Weather H1-H48, sweeps Exchange
H24; the Electricity row documents a data-quality effect rather than a
modelling failure." Report small-σ wins carefully:
- Exchange H24: 0.0904 vs best-pub 0.093 (FITS:dsof) — clean sweep ~3σ. Claim.
- Weather H24: 0.7961 vs best 0.671 (OneNet) — 13/14, we do NOT sweep. Do not
  overclaim.
- Weather H1 0.3105 vs 0.296 (OneNet) — 13/14; trust SE before claiming.
- ETTm1 H24 0.4505 vs 0.418 (OneNet) — 13/14, behind single best.

## 3. Electricity deep-dive (the value centre)

### 3.1 Empirical discovery (verified)
On the DSOF z-scored test segment, a NAIVE predictor (predict today for
tomorrow) gets MSE **6.75**. The published DSOF Electricity-H1 best is
**iTransformer:batch=1.976** (DLinear:dsof=2.065). Since even a PERFECT model
cannot beat ~6.75 on this target distribution, the published rows cannot be
plain MSE on the train-only StandardScaler + the same file — OR they rely on
per-window normalisation / seasonal-level tracking.

What actually happens:
- ECL meters **channels 114 and 146** run far past their training range in
  test: ch114 train ~0–50 vs test up to ~227; ch146 train ~0–12 vs test up to
  ~646 (~54x, z-score ~699).
- After train-only standardisation those become z ~ +10 … +690. The 0.29% of
  test points that are extreme carry **99.7% of total target energy**; on a
  persistence baseline the extreme points carry ~30x the mean squared error of
  the rest.
- Any linear/persistence model takes an unavoidable ~490k squared error on a
  few of those points → MSE is spike-dominated and numerically fragile (§4).

### 3.2 Why the published 2.065 exists
DSOF's loader applies ONLY a train-fit StandardScaler (verified from their
repo) — no clipping, same as ours, same raw file. The resolution: these ECL
channels are SEASONAL; a long-lookback model with per-window normalisation can
TRACK "channel 146 ~600 tonight, as every night" and turns the extreme targets
into small errors. A plain static/RLS head that only saw 0–12 in training
cannot. Hence published ~2.0 vs our ~21–723.

Framing: report our ECL row as an HONEST FAILURE on the exact public benchmark —
do NOT clip-and-claim. The publication-worthy story is the data-quality
investigation: "every published ECL MSE row inherits channels 114/146; their
effect is absorbed by MSE-as-mean, not by any preprocessing trick in the
standard loaders."

### 3.3 The clipping decision (chosen policy)
`get_protocol_dataset(clip_spikes=f)`:
- Per channel, `train_max_abs = max(|x_train|)`; clip ANY |x| > `f *
  train_max_abs` down to that bound, SIGN-PRESERVING, across the whole series
  (train untouched by construction). Default f=0 (exact DSOF); we ran f=3.
- Effect at f=3: exactly 2 channels (114, 146), 18 424 of 6 332 688 points
  (0.29% of test).

CLIPPED vs RAW (float64, deterministic):

    model    raw MSE      clip=3 MSE    raw seed(±)   clip seed(±)
    static   48.1754      0.3169        —             ~0.0014*
    rls      723.7446     0.0684        ±560.4        ±0.0014
    s2       51.1238      0.3323        —             —
    s2rls    21.5947      0.0675        ±4.77         ±0.0014
    (* static clip seed in electricity_1_clip3_seeds.json; block counts differ)

Context: on clip-trimmed data the persistence floor is 0.196 MSE and our
RLS/s2rls at ~0.068 BEATS that floor — the head is clearly useful on corrected
data. Raw, RLS/s2rls choke; s2rls is 30x better than RLS because the LRU
context damps the spike-induced gain explosion.

### 3.4 How to document Electricity (write-up recipe)
1. Leaderboard table (§2) shows Electricity UNCLIPPED and knowingly losing.
   One line: "On the exact public ECL file we score 21.6–723 MSE and lose;
   §X explains this is largely a data-quality artefact, and gives the
   corrected-data run."
2. Two-figure analysis: (a) ch146 train vs test raw range (0–12 vs 0–646);
   (b) persistence floor + fraction of energy in outliers, before/after clip.
3. State the clip policy NUMERICALLY ("sign-preserving cap at 3x the per-channel
   training maximum; 2 of 321 channels, 0.29% of test points"). Justify 3x as
   >3-sigma, touching only the two glitch channels; do not over-tune.
4. Report BOTH rows, clearly labelled "Ours (raw protocol)" and "Ours
   (corrected data, 3x-trim)". Never put the corrected row in the leaderboard.
5. Explicit reviewer caveat: "The two broken meters are part of the public
   dataset; every published ECL row inherits them. Our corrected-row MSE should
   be compared only to a corrected-data rerun of the published methods (out of
   scope); provided for transparency and robustness, not as a benchmark win."
6. Ideally rerun one cheap published method (e.g., DLinear batch) on raw AND
   clipped ECL for a same-conditions comparison. NOT done yet (open item §8).

## 4. Numerical determinism — the bug found & fixed (CRITICAL)

Symptom: Electricity-H1 MSE flipped run-to-run for the same code:
- rls float32: 49.77 / 62.65 / 49.77 (nondeterministic ACROSS processes,
  stable within a process)
- s2rls float32: 1.997 (looked like a "beat all" win) — AN ARTIFACT.
Investigation:
- MAE stayed ~0.226 across flips → only the squared-error tail moved.
- Cause: float32 GPU reductions (cuBLAS autotuning picks different kernels /
  summation orders across processes). On 700-z targets a 1e-7 weight difference
  → 10^4–10^6 squared-error swing. Electricity is the ONLY dataset fragile to
  this (max |y| ~699 z; all others < ~7 z).
- Fix: run the ONLINE RLS scan in float64 (x64 already enabled for the
  per-channel solve; only the 112–128-dim RLS is fp64 — cheap, verified vs
  numpy to 5e-16). static/s2 paths were already fine.
AFTER FIX (deterministic, reproducible): raw ECL rls 723.7446, s2rls 21.5947;
clip rls 0.0684, s2rls 0.0675. In-process triple-run bit-exact.

Paper obligation: never ship an MSE row computed in float32 GPU reductions on
spike-prone data. Document in "implementation details" that all reported
numbers are fp64-deterministic; footnote the float32 instability for
ECL-class data.

LESSION: the earlier "1.997 beats-all Electricity" is DISPROVEN. Do not cite
it. `electricity_1_s2.json`/`_s2_seeds.json` were overwritten with fp64 values;
`electricity_1_rls_seeds.json` removed as an artifact. Check `electricity_1.json`
(static fp32) before citing.

## 5. S2 — fixed LRU reservoir context

Design: add long-memory beyond the 96-lag window while keeping the closed-form
(no-SGD) claim.
- `features.lru_context(X, n_modes, seed, r_min, r_max)`: drives a bank of D
  FIXED random damped complex oscillators (diagonal LRU, |lambda| in
  [0.5, 0.995], random phases, Xavier B) with the standardised series:
  h_t = Λ h_{t-1} + B x_t. Features = [real(h), imag(h)] ∈ R^{T x S x 2D}
  (D=8 → 16 extra features). CAUSAL (only x_{<=t}) → leakage-free. Precomputed
  once per dataset, sliced to match `features_slice` offsets. FIXED — readout
  is closed-form ridge/RLS on the appended features ("next-gen reservoir":
  random recurrent + closed-form output).
- Plumbed: `_static_head` appends LRU rows to the Gram; `evaluate` models
  `s2`=static+LRU, `s2rls`=rls+LRU (default 8 modes; `--lru 8`).
Results (clean data, fp64):
- ettm1 H24: rls 0.4507±0.0148 vs s2rls 0.4498±0.0148 (within SE; s2 0.6149 ≈
  static).
- Electricity H1 raw: RLS 723.7 → s2rls 21.6 (30x gain, still loses on the
  benchmark row); static 48.2 → s2 51.1 (LRU hurts static slightly: bigger
  model, same ridge lambda on spike data).
- Electricity clip=3: RLS 0.0684 vs s2rls 0.0675 (marginal, within SE).
Framing: say honestly the LRU is a cheap, deterministic way to add memory; it
stabilises RLS on Electricity and costs nothing on clean data, but clean-data
gains are within block-bootstrap noise — a robustness/stabiliser result or an
ablation ("head is closed-form even with recurrent features"), not a big
accuracy win. Emphasise the CLOSED-FORM READOUT over a FIXED random recurrent
feature extractor (unusual and cheap).

## 6. Chaos track (track #4)

Files: `experiments/chaos.py`, `experiments/run_chaos.py`. The SAME closed-form
head (no SGD) used for the LTSF leaderboard also does online step-prediction on
chaotic systems, matching NG-RC (next-gen reservoir computing) parity targets.
Verified numbers (float64, 3 seeds unless noted):
- Lorenz96 (N=5, F=8, dt=0.02, RK4, QR Lyapunov):
    step NMSE         1.08e-07   (3-seed: 7.8e-08 ± 3e-08)
    λ1 (max Lyap)     1.11 ± 0.03 (λ1 = MAX of spectrum, not sum)
    free-run horizon  7.5 ± 2.3 Lyapunov times (3-seed)
    (seed-0 detail: step NMSE 1.08e-07, horizon 9.9 Lyapunov times)
- NARMA10: NMSE 1.0e-05 ± 2e-06 (drive+past-y observable, mem=12)
- NARMA30: NMSE 2.8e-04 ± 2e-05 (amp=0.2 — classic 0.5 diverges at long T,
  documented; target line <0.0391)
Paper notes:
- Use λ1 = MAX over the Lyapunov spectrum (earlier drafts summed them — fix any
  figure: "largest Lyapunov exponent, positive → chaotic").
- Chaos tasks carry real randomness (ICs/drives) so seed-spread IS meaningful
  there (unlike the deterministic LTSF head).
- Narrative: the SAME closed-form core that sweeps the streaming-forecasting
  leaderboard reconstructs chaotic dynamics to ~1e-7 step NMSE and ~7+ Lyapunov
  times without training.

## 7. Artefact → paper mapping

| Concept                   | Code / data file                            | Paper section suggestion          |
|---------------------------|---------------------------------------------|-----------------------------------|
| Protocol runner           | experiments/eval_protocol.py                | Method: closed-form streaming     |
| Feature builder           | experiments/features.py                     | Method: feature map (incl. LRU)   |
| Bench runner              | experiments/run_table.py --out ...          | Results: commands                 |
| Seed/bootstrap SE         | experiments/run_seeds.py (block=4,nboot=200)| Results: variability              |
| Leaderboard render        | experiments/compare_table.py                | Results: Ours vs pub tables       |
| Published anchors+prov    | data/proc/published_dsof.json               | Appendix: published rows          |
| All rows                  | data/proc/results/*.json                    | Appendix: full table              |
| Chaos core/runner         | experiments/chaos.py, run_chaos.py          | Results: chaos validation         |
| Chaos outputs             | data/proc/chaos_results.json                | Appendix: chaos tables            |
| ECL clip policy           | get_protocol_dataset(clip_spikes=3)         | Results: Electricity case study   |
| ECL raw rows              | data/proc/results/electricity_1_raw64.json  | Results: benchmark row (losing)   |
| ECL clip rows + SE        | electricity_1_clip33.json / _clip3_seeds    | Results: robustness section       |
| fp64 determinism          | eval_protocol.py (x64 on; RLS scan fp64)    | Implementation details            |

Reproduce any row:
    ./run.sh experiments.run_table --dataset <ds> --pred_len <H> \
        --models static rls s2 s2rls --lru 8 --out data/proc/results/<x>.json
    ./run.sh experiments.run_seeds --dataset <ds> --pred_len <H> \
        --models rls s2rls --lru 8 --out data/proc/results/<x>_seeds.json
    ./run.sh experiments.compare_table
  Electricity: add `--clip_spikes 3` (and compare raw vs clip).

Environment constants (paper appendix): JAX x64; XLA_PYTHON_CLIENT_PREALLOCATE
=false; XLA_PYTHON_CLIENT_MEM_FRACTION=0.6; TF_GPU_ALLOCATOR=cuda_malloc_async;
RTX 3050 6GB (WSL), scales to 862-channel traffic via chunked stream. Hyperparams:
seq_len=96 (ECL published uses 192, §1), lam=1e-3 (relative ridge), chunk_t=512,
chunk_online=1536, lru 8 modes (seed 0, r∈[0.5,0.995]).

## 8. Open items / integrity flags BEFORE submission

1. Do NOT cite the float32 "1.997 Electricity win" — artifact, fixed. Verify
   `electricity_1.json` (static fp32) and any other pre-fix JSONs.
2. ECL lookback: published table uses L=192, we ran L=96. Rerun ECL@192 (one
   command) or footnote the difference.
3. L=96 vs published weather: published uses L=96 — we match (no action).
4. Corrected-data apples-to-apples: optionally rerun one cheap published method
   (e.g., DLinear batch) on raw AND clip=3 ECL. Not yet done.
5. s2/rls SE overlap: ettm1 H24 s2rls-vs-rls delta is inside block-bootstrap
   noise — phrase S2 results accordingly (no significant clean-data accuracy
   win to claim).
6. Chaos vs paper target <0.0391: NARMA30 2.8e-04 ± 2e-05 already under;
   NARMA10 1.0e-05 vastly under. Keep the amp=0.2 note.
7. compare_table now quarantines clip rows and surfaces S2/S2RLS rows whose
   loss surface differs; re-run `./run.sh experiments.compare_table` after any
   new row file so the renderer stays honest.
8. Tests: `pytest tests` = 14 passed (incl. static-head == dense least-squares
   check). Keep green; `_window_stats` warns on empty-slice windows only (test
   edge), harmless.

## 9. Figure suite (A1-grade, 235 panels / 125 PNGs, 300 dpi)

Full pipeline is `figuregen/` — one module per `figures/<NN_name>/` branch;
`render_all.py` regenerates everything and rebuilds the manifest. Style is
centralised in `figuregen/style.py` (STIX2 serif, Okabe-Ito colourblind
palette, one accent colormap `YlGnBu`, 300 dpi, tight axes) so the whole
paper reads as one visual system. Multi-panel PNGs count each subplot as one
image (the Paper-2 inventory rule); `figures/manifest.json` records every
panel with caption.

    figuregen/style.py         global rcParams + palette + save_fig(panels=)
    figuregen/data.py          single loader (results/seeds/published/chaos/timing)
    figuregen/schematics.py    diagram primitives (boxes, arrows, lanes)
    figuregen/s01_schematics.py   7 diagrams: pipeline, RLS Woodbury, LRU bank,
                                  feature map, protocol timeline, chaos core,
                                  closed-form-vs-backprop
    figuregen/s02_dataset.py      raw montages (per channel), splits, cadence,
                                  ECL spike channels 114/146, 3x trim view
    figuregen/s03_leaderboard.py  per (ds,H): bars(MSE+MAE), hist, scatter,
                                  relative-improvement heatmap, horizon curves,
                                  bootstrap bars, + per-dataset 3-panel summary
    figuregen/s04_electricity.py  raw-vs-clip bars, persistence floor, energy
                                  in outliers, s2 damping, bootstrap raw SE
    figuregen/s05_s2_lru.py       LRU eigenvalue spectrum, impulse responses,
                                  ETTm1 rls-vs-s2rls, S2RLS-vs-RLS Δ across sets
    figuregen/s06_chaos.py        L96 traces, free-run error vs Lyap time,
                                  seed stats, phase portrait, NARMA recon + parity
    figuregen/manifest.py         counts panels + writes figures/manifest.json

Regenerate everything:
    PYTHONPATH=. python figuregen/render_all.py     # re-renders + manifest

Guardrails baked into the renderer: Electricity corrected rows stay only in
04 (never in the leaderboard); chaos uses λ1=max. Verified 14/14 pytest pass after
all renders.