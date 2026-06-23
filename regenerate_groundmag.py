"""Regenerate ground_mag_5min.pkl (with download retries) and rebuild the dataset.

Run: .venv/bin/python regenerate_groundmag.py
"""
import pickle
from pathlib import Path

from speasy.signal.resampling import resample as spz_resample

from dataset_tools import load_ground_mag
from regenerate import rebuild_dataset, log, START, STOP, INTERVAL

STATIONS = ("TAM", "SOK", "EDA", "CLF", "KOU", "IPM", "PPT")


def regenerate_ground_mag_pickle():
    log("[1/3] Re-fetching ground-mag stations (one day-file per UTC day, with retries)...")
    variables = []
    for st in STATIONS:
        r = load_ground_mag(START, STOP, st)
        if r is None or not len(r.time):
            log(f"      {st}: NO DATA"); continue
        log(f"      {st}: N={len(r.time)} range {r.time.min()} -> {r.time.max()}")
        variables.append(spz_resample(r, INTERVAL))
    with open("ground_mag_5min.pkl", "wb") as f:
        pickle.dump(variables, f)
    log("      wrote ground_mag_5min.pkl")


if __name__ == "__main__":
    regenerate_ground_mag_pickle()
    rebuild_dataset()
