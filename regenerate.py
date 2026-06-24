"""Regenerate every *_5min.pkl with NaN-aware 5-min bin-averaging, then rebuild
the IMAOC7 dataset and overview plots.

Bin-averaging (dataset_tools.bin_average) replaces interpolation-based resampling so
that scattered sub-bin gaps are averaged away (not propagated) and real multi-sample
gaps become NaN (not bridged by a ramp). Every source lands on the same global grid.

Run: .venv/bin/python regenerate.py
"""
import pickle
from glob import glob

import numpy as np
import pandas as pd

from speasy.products import Dataset
from speasy.products.variable import merge
from speasy.signal.resampling import generate_time_vector

from dataset_tools import (bin_average, fetch_resampled_chunked, fetch_mms1_fgm_b_gse,
                           interpolate_with_gaps, load_ground_mag)
import make_overview_plots

START, STOP, INTERVAL = "2024/01/01", "2026/01/01", 60. * 5
GRID_ORIGIN = START
AMDA_FGM_END = "2025-09-04"   # AMDA mms1_b_gse stops 2025-09-03; CDAWeb covers beyond
GROUND_STATIONS = ("TAM", "SOK", "EDA", "CLF", "KOU", "IPM", "PPT")


def log(msg):
    print(msg, flush=True)


def _bin_fetch(uid, chunk_days):
    return fetch_resampled_chunked(uid, START, STOP, INTERVAL, chunk_days=chunk_days)


def build_omni():
    log("[OMNI] fetch 1-min, mask FILLVAL, 5-min bin-average...")
    vs = [_bin_fetch(f"cda/OMNI_HRO2_1MIN/{p}", 30) for p in
          ("proton_density", "Pressure", "T", "Vx", "Vy", "Vz", "BX_GSE", "BY_GSE", "BZ_GSE")]
    pickle.dump(vs, open("omni_5min.pkl", "wb"))


def build_themis(sc):
    log(f"[THEMIS-{sc.upper()}] bin-average bs_gsm / v_i / n_i...")
    vs = [_bin_fetch(f"amda/th{sc}_{p}", 30) for p in ("bs_gsm", "v_i", "n_i")]
    pickle.dump(vs, open(f"themis_{sc}_5min.pkl", "wb"))


def build_mms1():
    log("[MMS1] FGM B (AMDA -> 2025-09 + CDAWeb tail) and FPI moments, bin-averaged...")
    amda_b = fetch_resampled_chunked("amda/mms1_b_gse", START, AMDA_FGM_END, INTERVAL, chunk_days=7)
    tail = fetch_mms1_fgm_b_gse("2025-09-03", STOP, INTERVAL, chunk_days=7)
    tail = tail[amda_b.time.max() + np.timedelta64(1, "s"):]
    b = merge([amda_b, tail])
    log(f"       b_gse: {b.time.min()} -> {b.time.max()} (N={len(b.time)})")
    fpi = [_bin_fetch("amda/mms1_dis_vgse", 30), _bin_fetch("amda/mms1_dis_ni", 30)]
    pickle.dump([b] + fpi, open("mms1_5min.pkl", "wb"))


def build_ground_mag():
    log("[GROUND] fetch BCMT day-files (with retries) and 5-min bin-average...")
    vs = []
    for st in GROUND_STATIONS:
        r = load_ground_mag(START, STOP, st)
        if r is None or not len(r.time):
            log(f"       {st}: NO DATA"); continue
        vs.append(bin_average(r, INTERVAL, GRID_ORIGIN))
        log(f"       {st}: ok")
    pickle.dump(vs, open("ground_mag_5min.pkl", "wb"))


def rebuild_dataset():
    log("[ASSEMBLE] align every source onto the common 5-min grid...")
    time_vector = generate_time_vector(START, STOP, INTERVAL)
    df = pd.DataFrame()
    for fname in glob("*_5min.pkl"):
        for v in pickle.load(open(fname, "rb")):
            vi = interpolate_with_gaps(time_vector, v)
            _df = vi.to_dataframe()
            if p := vi.name:
                _df.columns = [f"{p}_{l}" for l in _df.columns]
            df = pd.concat([df, _df], axis=1)
        log(f"       merged {fname}")
    df.to_pickle("IMAOC7_summer_school_dataset.pkl")
    df.to_csv("IMAOC7_summer_school_dataset.csv")
    log("       wrote IMAOC7_summer_school_dataset.{pkl,csv}")


if __name__ == "__main__":
    build_omni()
    build_themis("a")
    build_themis("b")
    build_mms1()
    build_ground_mag()
    rebuild_dataset()
    log("[PLOTS] regenerating overview plots...")
    make_overview_plots.main()
    log("done.")
