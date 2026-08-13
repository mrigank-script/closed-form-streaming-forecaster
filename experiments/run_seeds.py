"""Block-bootstrap variability for our protocol rows.

The closed-form head is deterministic (no weight-init randomness), so the
honest equivalent of published 5-seed averages is sampling variability over
test chunks: the online sweep emits roughly-iid error blocks per (dataset, H,
model). We report the block-bootstrap distribution of aggregate MSE/MAE (and
a jackknife-style SE) — what a reviewer can compare against published std rows.

Usage:  ./run.sh experiments.run_seeds --dataset weather --pred_len 24
                                      --models rls --nboot 200
"""

import argparse
import json
import os

import numpy as np

from experiments.data import get_protocol_dataset
from experiments.eval_protocol import evaluate


def block_bootstrap(ce, nboot=200, seed=0, block=1):
    """Block-bootstrap MSE/MAE from per-chunk (sq, ab, n) tuples.

    block=1 resamples single chunks; block>1 resamples contiguous runs
    (dependent-data safe). Returns mean, se, and 5/95 percentiles of the
    bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    sq = np.array([c[0] for c in ce])
    ab = np.array([c[1] for c in ce])
    n = np.array([c[2] for c in ce])
    K = len(sq)
    nB = int(np.ceil(K / block))
    mse_hat = sq.sum() / n.sum()
    mae_hat = ab.sum() / n.sum()

    mse_b, mae_b = [], []
    for _ in range(nboot):
        idx = rng.integers(0, K - block + 1, size=nB)
        # take block contiguous chunks starting at idx
        blocks = (idx[:, None] + np.arange(block)) % K
        sel = blocks.ravel()
        mse_b.append(sq[sel].sum() / n[sel].sum())
        mae_b.append(ab[sel].sum() / n[sel].sum())
    mse_b = np.array(mse_b)
    mae_b = np.array(mae_b)
    return {
        "mse": float(mse_hat), "mae": float(mae_hat),
        "mse_se": float(mse_b.std(ddof=1)), "mae_se": float(mae_b.std(ddof=1)),
        "mse_p5": float(np.percentile(mse_b, 5)), "mse_p95": float(np.percentile(mse_b, 95)),
        "mae_p5": float(np.percentile(mae_b, 5)), "mae_p95": float(np.percentile(mae_b, 95)),
        "nboot": int(nboot), "block": int(block), "nchunks": int(K),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--models", nargs="+", default=["static", "rls"])
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--lru", type=int, default=0,
                   help="S2: number of LRU reservoir modes to append (0=off)")
    p.add_argument("--lru_seed", type=int, default=0)
    p.add_argument("--clip_spikes", type=float, default=0.0,
                   help="robust-trim: clip |x| > factor*train_max_abs per channel (ECL broken meters)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ds = get_protocol_dataset(a.dataset, clip_spikes=a.clip_spikes)
    print(f"{a.dataset}: X{ds['X'].shape}  test={ds['borders']['test']}")

    out_rows = []
    for m in a.models:
        r = evaluate(ds, a.pred_len, model=m, return_by_step=False,
                     return_chunk_errs=True,
                     n_lru_modes=a.lru, lru_seed=a.lru_seed)
        ce = r["chunk_errs"]
        bs = block_bootstrap(ce, a.nboot, block=a.block)
        row = {"dataset": a.dataset, "pred_len": a.pred_len, "model": m,
               "mse": r["mse"], "mae": r["mae"], **bs,
               "clip_spikes": a.clip_spikes}
        out_rows.append(row)
        print(f"  {m:<7} MSE {row['mse']:.4f} ± {row['mse_se']:.4f} "
              f"[{row['mse_p5']:.4f}, {row['mse_p95']:.4f}]   "
              f"MAE {row['mae']:.4f} ± {row['mae_se']:.4f}   "
              f"(chunks={row['nchunks']})")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"dataset": a.dataset, "pred_len": a.pred_len,
                       "nboot": a.nboot, "block": a.block, "rows": out_rows},
                      f, indent=2)
        print(f"[run_seeds] -> {a.out}")


if __name__ == "__main__":
    main()