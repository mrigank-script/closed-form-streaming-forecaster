"""Chaos track driver.

Usage:  ./run.sh experiments.run_chaos [--seeds N]

With --seeds N each task runs over N random initial conditions / drives and
reports mean ± std — the honest seed spread for the chaotic tasks, which do
have genuine randomness in ICs and drives.
"""

import argparse
import json
import os

import numpy as np

from experiments.chaos import (
    benchmark_lorenz96,
    benchmark_narma,
)


def summarize(vals, name):
    v = np.asarray(vals, dtype=np.float64)
    return {f"{name}_mean": float(v.mean()), f"{name}_std": float(v.std(ddof=1)),
            f"{name}_n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    n_seeds = a.seeds

    out = os.path.join("data", "proc", "chaos_results.json")
    results = {}

    print("=== Lorenz96 (N=5, F=8, dt=0.02) — NG-RC parity ===")
    l96 = []
    for seed in range(n_seeds):
        try:
            r = benchmark_lorenz96(n_train=5000, n_test=2000, seed=seed)
            l96.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Lorenz96 seed {seed}: FAILED {e}")
    if l96:
        r0 = l96[0]
        results["lorenz96"] = r0
        results["lorenz96_seed_stats"] = {
            "nmse_step": summarize([r["nmse_step"] for r in l96], "nmse_step"),
            "lyapunov_lam1": summarize([r["lyapunov_lam1"] for r in l96], "lyapunov_lam1"),
            "error_crossed_std_lyap": summarize(
                [r["error_crossed_std_lyap"] for r in l96], "error_crossed_std_lyap"),
        }
        print(f"  step NMSE      : {results['lorenz96_seed_stats']['nmse_step']}")
        print(f"  Lyapunov λ1    : {results['lorenz96_seed_stats']['lyapunov_lam1']}")
        print(f"  error crosses σ: "
              f"{results['lorenz96_seed_stats']['error_crossed_std_lyap']} Lyapunov times")
        print(f"  (seed 0 detail: {r0['nmse_step']:.2e}, "
              f"λ1={r0['lyapunov_lam1']:.3f}, "
              f"horizon={r0['error_crossed_std_lyap']:.2f})")

    print("=== NARMA echo-state tasks ===")
    for n, mem in ((10, 12), (30, 35)):
        try:
            r0 = benchmark_narma(n_tau=n, mem=mem, n_train=5000, n_test=2000,
                                 amp=(0.5 if n == 10 else 0.2), seed=0)
            results[f"narma{n}"] = r0
            if n_seeds > 1:
                stats = summarize(
                    [benchmark_narma(n_tau=n, mem=mem, n_train=5000, n_test=2000,
                                     amp=(0.5 if n == 10 else 0.2), seed=s)["nmse"]
                     for s in range(n_seeds)], "nmse")
                results[f"narma{n}_seed_stats"] = stats
                print(f"  NARMA{n:>2}: NMSE {stats['nmse_mean']:.6f} ± "
                      f"{stats['nmse_std']:.6f}  (features {r0['n_features']})")
            else:
                print(f"  NARMA{n:>2}: NMSE {r0['nmse']:.6f}  (features {r0['n_features']})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  NARMA{n}: FAILED {e}")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[run_chaos] -> {out}")


if __name__ == "__main__":
    main()