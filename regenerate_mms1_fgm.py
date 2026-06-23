"""Extend MMS1 B-field coverage to 2026 using CDAWeb FGM, rebuild dataset + plots.

AMDA's mms1_b_gse stops 2025-09-03; CDAWeb's MMS1_FGM_SRVY_L2 (same L2 data) runs to
2026. The existing pickle already holds good B up to the AMDA cutoff, so we only fetch
the CDAWeb tail beyond it and merge — avoiding a multi-year full-resolution re-download.

Run: .venv/bin/python regenerate_mms1_fgm.py
"""
import pickle

import numpy as np
from speasy.products.variable import merge

from dataset_tools import fetch_mms1_fgm_b_gse
from regenerate import rebuild_dataset, log, STOP, INTERVAL
import make_overview_plots


def extend_mms1_b_to_2026():
    with open("mms1_5min.pkl", "rb") as f:
        ds = pickle.load(f)
    by_name = {v.name: v for v in ds}
    existing_b = by_name.pop("mms1_b_gse")
    cutover = existing_b.time.max()
    log(f"[1/3] existing b_gse ends {cutover}; fetching CDAWeb FGM beyond it...")

    tail = fetch_mms1_fgm_b_gse("2025-09-03", STOP, INTERVAL, chunk_days=7)
    tail = tail[cutover + np.timedelta64(1, "s"):]
    log(f"      CDA tail: {tail.time.min()} -> {tail.time.max()} (N={len(tail.time)})")

    full_b = merge([existing_b, tail])
    log(f"      merged b_gse: {full_b.time.min()} -> {full_b.time.max()} (N={len(full_b.time)})")

    with open("mms1_5min.pkl", "wb") as f:
        pickle.dump([full_b] + list(by_name.values()), f)
    log("      wrote mms1_5min.pkl")


if __name__ == "__main__":
    extend_mms1_b_to_2026()
    rebuild_dataset()
    log("Regenerating overview plots...")
    make_overview_plots.main()
