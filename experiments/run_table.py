"""experiments/run_table.py — reproduce official-protocol rows for a dataset.

Usage:  ./run.sh experiments.run_table --dataset etth1 --pred_len 24
                                       --models static rls --seq_len 96

Protocol: DSOF (ICLR 2025) / OneNet (NeurIPS 2023) — DLinear-class static,
and online RLS (S1) scored with cumulative MSE/MAE over the test segment,
comparable to DSOF Table 2.
"""

import argparse
import json
import os

from experiments.data import get_protocol_dataset
from experiments.eval_protocol import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--models", nargs="+", default=["static", "rls"])
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--chunk_t", type=int, default=512)
    p.add_argument("--chunk_online", type=int, default=1536)
    p.add_argument("--lru", type=int, default=0,
                   help="S2: number of LRU reservoir modes to append (0=off)")
    p.add_argument("--lru_seed", type=int, default=0)
    p.add_argument("--clip_spikes", type=float, default=0.0,
                   help="robust-trim: clip |x| > factor*train_max_abs per channel (ECL broken meters)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ds = get_protocol_dataset(a.dataset, seq_len=a.seq_len,
                              clip_spikes=a.clip_spikes)
    print(f"{a.dataset}: X{ds['X'].shape}  borders={ds['borders']}  meta={ds['meta']}")

    rows = []
    for m in a.models:
        r = evaluate(ds, a.pred_len, model=m, lam=a.lam, seq_len=a.seq_len,
                     chunk_t=a.chunk_t, chunk_online=a.chunk_online,
                     n_lru_modes=a.lru, lru_seed=a.lru_seed)
        r["clip_spikes"] = a.clip_spikes
        rows.append(r)
        print(f"  pred_len={r['pred_len']:<3} {r['model']:<7} "
              f"MSE {r['mse']:.4f}   MAE {r['mae']:.4f}   (n={r['n']})")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[run_table] rows -> {a.out}")


if __name__ == "__main__":
    main()