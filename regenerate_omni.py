"""Regenerate omni_5min.pkl with FILLVAL masking, rebuild the dataset, redo plots.

Run: .venv/bin/python regenerate_omni.py
"""
import pickle
from pathlib import Path

import speasy as spz
from speasy.products import Dataset
from speasy.signal.resampling import resample as spz_resample

from regenerate import rebuild_dataset, log, START, STOP, INTERVAL
import make_overview_plots

OMNI = (
    ("cda/OMNI_HRO2_1MIN/proton_density", "ni"),
    ("cda/OMNI_HRO2_1MIN/Pressure", "Pdyn"),
    ("cda/OMNI_HRO2_1MIN/T", "T"),
    ("cda/OMNI_HRO2_1MIN/Vx", "Vx"),
    ("cda/OMNI_HRO2_1MIN/Vy", "Vy"),
    ("cda/OMNI_HRO2_1MIN/Vz", "Vz"),
    ("cda/OMNI_HRO2_1MIN/BX_GSE", "BX_GSE"),
    ("cda/OMNI_HRO2_1MIN/BY_GSE", "BY_GSE"),
    ("cda/OMNI_HRO2_1MIN/BZ_GSE", "BZ_GSE"),
)


def regenerate_omni_pickle():
    log("[1/3] Re-fetching OMNI with FILLVAL -> NaN masking...")
    variables = {}
    for uid, param in OMNI:
        v = spz.get_data(uid, START, STOP)
        variables[param] = spz_resample(v.replace_fillval_by_nan(), INTERVAL)
        log(f"      {param}: ok")
    with open("omni_5min.pkl", "wb") as f:
        pickle.dump(list(variables.values()), f)
    log("      wrote omni_5min.pkl")


if __name__ == "__main__":
    regenerate_omni_pickle()
    rebuild_dataset()
    log("Regenerating overview plots...")
    make_overview_plots.main()
