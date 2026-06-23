"""Regenerate mms1_b_gse (chunked) and rebuild the IMAOC7 dataset with NaN gaps.

Run: .venv/bin/python regenerate.py
"""
import pickle
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

import speasy as spz
from speasy.signal.resampling import generate_time_vector

from dataset_tools import fetch_resampled_chunked, interpolate_with_gaps

START, STOP, INTERVAL = "2024/01/01", "2026/01/01", 60. * 5


def log(msg):
    print(msg, flush=True)


def regenerate_mms1_pickle():
    log("[1/3] Re-fetching mms1_b_gse in 7-day chunks (FGM survey ~16 Hz)...")
    b_gse = fetch_resampled_chunked("amda/mms1_b_gse", START, STOP, INTERVAL, chunk_days=7)
    if b_gse is None or not len(b_gse.time):
        log("ERROR: mms1_b_gse fetch returned nothing"); sys.exit(1)
    log(f"      mms1_b_gse: N={len(b_gse.time)} "
        f"range {b_gse.time.min()} -> {b_gse.time.max()}")

    with open("mms1_5min.pkl", "rb") as f:
        old = pickle.load(f)
    kept = [v for v in old if v.name != "mms1_b_gse"]
    log(f"      keeping from old pickle: {[v.name for v in kept]}")

    with open("mms1_5min.pkl", "wb") as f:
        pickle.dump([b_gse] + kept, f)
    log("      wrote mms1_5min.pkl")


def rebuild_dataset():
    log("[2/3] Rebuilding combined dataframe with NaN-gap interpolation...")
    time_vector = generate_time_vector(START, STOP, INTERVAL)
    df = pd.DataFrame()
    for fname in glob("*_5min.pkl"):
        with open(fname, "rb") as f:
            dataset = pickle.load(f)
        for v in dataset:
            vi = interpolate_with_gaps(time_vector, v)
            _df = vi.to_dataframe()
            if p := vi.name:
                _df.columns = [f"{p}_{l}" for l in _df.columns]
            df = pd.concat([df, _df], axis=1)
        log(f"      merged {fname}")

    log("[3/3] Writing IMAOC7_summer_school_dataset.pkl / .csv ...")
    df.to_pickle("IMAOC7_summer_school_dataset.pkl")
    df.to_csv("IMAOC7_summer_school_dataset.csv")
    mms1_b = [c for c in df.columns if c.startswith("mms1_b_gse")]
    log("      mms1_b_gse NaN counts: "
        + ", ".join(f"{c}={int(df[c].isna().sum())}" for c in mms1_b))
    log("      done.")


if __name__ == "__main__":
    regenerate_mms1_pickle()
    rebuild_dataset()
