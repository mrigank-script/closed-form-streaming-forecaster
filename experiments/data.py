"""experiments/data.py — public multivariate time-series loaders + splits.

Sources (public, reproducible):
  ECL (electricity), Traffic, Exchange-rate : laiguokun/multivariate-time-series-data
  ETT (ETTh1/ETTm1/ETTh2)                   : zhouhaoyi/ETDataset
  Weather                                   : Time-Series-Library (thuml)

Splits follow the no-information-leakage protocol of the online-forecasting
literature (OneNet/DSOF): chronological 20% train / 5% validation / 75% test,
with per-series normalization statistics computed on the TRAIN segment only.
"""

import os
import gzip
import urllib.request
import zipfile
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "proc")

_URLS = {
    "electricity": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
    "traffic": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
    "exchange": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
    "etth1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "etth2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ettm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "weather": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv",
}


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)


def available_datasets():
    return sorted(_URLS)


def download(name: str, force: bool = False) -> str:
    """Download a raw dataset to data/raw if absent. Returns local path."""
    ensure_dirs()
    if name not in _URLS:
        raise ValueError(f"unknown dataset {name!r}; available: {available_datasets()}")
    gz = name not in ("ett" + "h1", "etth2", "ettm1") and name != "weather"
    ext = ".csv" if name.startswith("ett") or name == "weather" else ".txt"
    dest = os.path.join(RAW_DIR, name + ext)
    if force or not os.path.exists(dest):
        url = _URLS[name]
        print(f"[data] downloading {name} <- {url}")
        gz_path = dest + ".gz" if gz else dest
        urllib.request.urlretrieve(url, gz_path)
        if gz:
            print(f"[data] gunzipping {gz_path}")
            with gzip.open(gz_path, "rb") as fin:
                with open(dest, "wb") as fout:
                    fout.write(fin.read())
            os.remove(gz_path)
        print(f"[data] saved {dest}")
    else:
        print(f"[data] cached {dest}")
    return dest


def load_array(name: str) -> np.ndarray:
    """Return the numeric matrix for a dataset: (T, num_series) float64."""
    ensure_dirs()
    path = download(name)
    if name.startswith("ett"):
        import pandas as pd
        df = pd.read_csv(path)
        return df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if name == "weather":
        try:
            raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
            return raw[:, 1:].astype(np.float64)
        except ValueError:
            # weather.csv has an object first column in some mirrors
            import pandas as pd
            df = pd.read_csv(path)
            num = df.select_dtypes(include=[np.number])
            return num.to_numpy(dtype=np.float64)
    # generic multivariate files (electricity/traffic/exchange): auto-delimiter
    with open(path, "r") as f:
        first = f.readline()
    delim = "," if "," in first else None
    return np.loadtxt(path, delimiter=delim, dtype=np.float64)


def get_dataset(name: str, train_frac: float = 0.2, val_frac: float = 0.05,
                test_horizons=(1, 24, 48)):
    """Load + normalize (train-only stats) + chronological split.

    Returns dict with 'X_all' (normalized, with train/val/test index ranges),
    'mean', 'std', 'meta'.
    """
    X = load_array(name)
    X = np.asarray(X, dtype=np.float32)
    T, D = X.shape
    t_train = int(T * train_frac)
    t_val = int(T * (train_frac + val_frac))

    mean = X[:t_train].mean(axis=0, keepdims=True)
    std = X[:t_train].std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mean) / std

    return {
        "name": name,
        "X": Xn,                     # (T, D) normalized
        "mean": mean.reshape(-1),
        "std": std.reshape(-1),
        "T": T, "D": D,
        "t_train": t_train, "t_val": t_val,
        "test_horizons": test_horizons,
        "meta": {"n_train": t_train, "n_val": t_val - t_train, "n_test": T - t_val},
    }


def protocol_borders(name: str, len_df: int, seq_len: int = 96) -> dict:
    """Chronological split borders replicating the DSOF/FSNet official loader.

    See src/data/data_loader.py of yyalau/iclr2025_dsof. ETT uses the classic
    Autoformer borders (with the seq_len ~lookback offset baked into val/test
    starts); the others use 20% train / 5% val / 75% test fractions. This is
    the exact protocol our published rows must be comparable against.

    Returns dict of (start, end) per phase.
    """
    if name in ("etth1", "etth2"):
        return {
            "train": (0, 4 * 30 * 24),
            "val": (4 * 30 * 24 - seq_len, 5 * 30 * 24),
            "test": (5 * 30 * 24 - seq_len, 20 * 30 * 24),
        }
    if name in ("ettm1", "ettm2"):
        return {
            "train": (0, 4 * 30 * 24 * 4),
            "val": (4 * 30 * 24 * 4 - seq_len, 5 * 30 * 24 * 4),
            "test": (5 * 30 * 24 * 4 - seq_len, 20 * 30 * 24 * 4),
        }
    tr = int(len_df * 0.2)
    te = int(len_df * 0.25)
    return {"train": (0, tr), "val": (tr, te), "test": (te, len_df)}


def get_protocol_dataset(name: str, seq_len: int = 96, clip_spikes: float = 0.0) -> dict:
    """DSOF-protocol-exact dataset: raw -> trim -> StandardScaler(train) -> borders.

    Mirrors the official (ICLR 2025) DSOF pipeline so our metrics are directly
    comparable to its Table 2 (batch learning DLinear/FITS/FSNet/OneNet/etc.).

    `clip_spikes` (0 = off): robust-trim flag. The public ECL has a few broken
    meters whose TEST values run ~100x past their TRAIN range (e.g. channel 146
    trains 0-12, test reaches 646; z-scored ~690). Every method, DSOF included,
    scores a fixed catastrophic MSE on those few points (a perfect predictor
    still loses ~(690)^2 there), and RLS diverges on them in float64. With
    clip_spikes>0 we sign-preserving clip ANY value in the FULL series that
    exceeds clip_spikes * train_max_abs(channel) down to that bound, which
    removes the glitch tail without touching the train distribution. The number
    of touched channels/points is reported in meta so the paper can disclose it.

    Returns dict with 'X' (scaled on train-only stats), 'borders', 'meta'.
    """
    DSOF_TABLE1 = {
        "electricity": (321, 26304), "traffic": (862, 17544),
        "exchange": (8, 7396), "weather": (21, 52696),
        "etth1": (7, 14400), "etth2": (7, 14400), "ettm1": (7, 57600),
    }
    X = load_array(name)
    X = np.asarray(X, dtype=np.float64)
    T, S = X.shape
    exp_features, exp_T = DSOF_TABLE1[name]

    # FSNet/DSOF trim leading rows for Exchange; ETT trimmed to 20 months.
    if name == "exchange":
        X = X[-exp_T:]
    elif name in ("etth1", "etth2", "ettm1"):
        X = X[:exp_T]
    assert X.shape[1] == exp_features, f"{name}: got {X.shape[1]} series, DSOF uses {exp_features}"
    assert X.shape[0] == exp_T, f"{name}: got T={X.shape[0]}, DSOF uses {exp_T}"

    borders = protocol_borders(name, exp_T, seq_len)
    tr_start, tr_end = borders["train"]

    trim = {"clip_factor": 0.0, "channels": 0, "points": 0}
    if clip_spikes > 0:
        train_max_abs = np.abs(X[tr_start:tr_end]).max(axis=0, keepdims=True)
        cap = clip_spikes * train_max_abs
        cap = np.where(cap < 1e-9, np.inf, cap)     # constant channels untouched
        over = np.abs(X) > cap
        clipped = np.sign(X) * np.minimum(np.abs(X), cap)
        trim = {"clip_factor": float(clip_spikes),
                "channels": int((over.any(axis=0)).sum()),
                "points": int(over.sum())}
        X = np.where(over, clipped, X)

    mean = X[tr_start:tr_end].mean(axis=0, keepdims=True)
    std = X[tr_start:tr_end].std(axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)        # constant channels -> identity
    Xn = (X - mean) / std

    meta = {
        "T": exp_T, "S": exp_features,
        "n_train": tr_end - tr_start,
        "n_val": borders["val"][1] - borders["val"][0],
        "n_test": borders["test"][1] - borders["test"][0],
        "clip_spikes": trim,
    }
    return {"name": name, "X": Xn, "borders": borders, "meta": meta,
            "mean": mean.reshape(-1), "std": std.reshape(-1)}


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["electricity"]
    ensure_dirs()
    for n in names:
        d = get_protocol_dataset(n)
        print(f"{n}: X{d['X'].shape}  borders train/val/test = "
              f"{d['borders']}  meta={d['meta']}")