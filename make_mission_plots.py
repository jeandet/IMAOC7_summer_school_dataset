"""One stacked time-series figure per mission: each physical quantity in its own
panel, sharing the time axis. Gaps (NaN) show as breaks in the lines.

Run: .venv/bin/python make_mission_plots.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

# Per mission: list of panels (ylabel, {series label: column}, logy).
MISSIONS = {
    "OMNI": [
        ("B GSE (nT)", {"Bx": "omni_bx_gse", "By": "omni_by_gse", "Bz": "omni_bz_gse"}, False),
        ("V GSE (km/s)", {"Vx": "omni_vx_gse", "Vy": "omni_vy_gse", "Vz": "omni_vz_gse"}, False),
        ("ni (cm⁻³)", {"ni": "omni_n"}, False),
        ("Pdyn (nPa)", {"Pdyn": "omni_pdyn"}, False),
        ("T (K)", {"T": "omni_t"}, True),
    ],
    "THEMIS-A": [
        ("B GSM (nT)", {"Bx": "tha_bx_gsm", "By": "tha_by_gsm", "Bz": "tha_bz_gsm"}, False),
        ("Vi GSE (km/s)", {"Vx": "tha_vx_gse", "Vy": "tha_vy_gse", "Vz": "tha_vz_gse"}, False),
        ("ni (cm⁻³)", {"ni": "tha_n"}, False),
    ],
    "THEMIS-B": [
        ("B GSM (nT)", {"Bx": "thb_bx_gsm", "By": "thb_by_gsm", "Bz": "thb_bz_gsm"}, False),
        ("Vi GSE (km/s)", {"Vx": "thb_vx_gse", "Vy": "thb_vy_gse", "Vz": "thb_vz_gse"}, False),
        ("ni (cm⁻³)", {"ni": "thb_n"}, False),
    ],
    "MMS1": [
        ("B GSE (nT)", {"Bx": "mms1_bx_gse", "By": "mms1_by_gse", "Bz": "mms1_bz_gse"}, False),
        ("Vi GSE (km/s)", {"Vx": "mms1_vx_gse", "Vy": "mms1_vy_gse", "Vz": "mms1_vz_gse"}, False),
        ("ni (cm⁻³)", {"ni": "mms1_n"}, False),
    ],
    "GROUND": [
        (f"{st} (nT)", {"X": f"{st.lower()}_x", "Y": f"{st.lower()}_y", "Z": f"{st.lower()}_z"}, False)
        for st in ("TAM", "SOK", "EDA", "CLF", "KOU", "IPM", "PPT")
    ],
}


def make_mission_figure(h, name, panels, detrend=False):
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.0 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (ylabel, series, logy) in zip(axes, panels):
        for label, col in series.items():
            if col in h:
                y = h[col]
                if detrend:                 # show variations, not absolute baseline
                    y = y - y.median()
                ax.plot(h.index, y.values, lw=0.5, label=label)
        ax.set_ylabel(("Δ " if detrend else "") + ylabel, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(ncol=len(series), fontsize=7, loc="upper right")
    axes[0].set_title(f"IMAOC7 dataset — {name} (hourly means; gaps = no data)")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out = PLOTS / f"mission_{name.replace('-', '').lower()}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    df = pd.read_pickle("IMAOC7_summer_school_dataset.pkl")
    h = df.resample("1h").mean()   # lighten plots; NaN gaps preserved
    for name, panels in MISSIONS.items():
        out = make_mission_figure(h, name, panels, detrend=(name == "GROUND"))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
