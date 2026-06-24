"""Overview plots for the IMAOC7 dataset: data-coverage heatmap + signal overview.

Run: .venv/bin/python make_overview_plots.py
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

# (group label, [(column, short label), ...]) — drives both figures.
GROUPS = [
    ("OMNI", [("omni_n", "ni"), ("omni_pdyn", "Pdyn"), ("omni_t", "T"),
              ("omni_vx_gse", "Vx"), ("omni_vy_gse", "Vy"), ("omni_vz_gse", "Vz"),
              ("omni_bx_gse", "Bx"), ("omni_by_gse", "By"), ("omni_bz_gse", "Bz")]),
    ("THEMIS-A", [("tha_bx_gsm", "Bx"), ("tha_by_gsm", "By"), ("tha_bz_gsm", "Bz"),
                  ("tha_vx_gse", "Vx"), ("tha_vy_gse", "Vy"), ("tha_vz_gse", "Vz"),
                  ("tha_n", "ni")]),
    ("THEMIS-B", [("thb_bx_gsm", "Bx"), ("thb_by_gsm", "By"), ("thb_bz_gsm", "Bz"),
                  ("thb_vx_gse", "Vx"), ("thb_vy_gse", "Vy"), ("thb_vz_gse", "Vz"),
                  ("thb_n", "ni")]),
    ("MMS1", [("mms1_bx_gse", "Bx"), ("mms1_by_gse", "By"), ("mms1_bz_gse", "Bz"),
              ("mms1_vx_gse", "Vx"), ("mms1_vy_gse", "Vy"), ("mms1_vz_gse", "Vz"),
              ("mms1_n", "ni")]),
]
GROUPS += [(st, [(f"{st.lower()}_{c}", c.upper()) for c in "xyzf"])
           for st in ("TAM", "SOK", "EDA", "CLF", "KOU", "IPM", "PPT")]


def coverage_heatmap(df):
    cols = [c for _, items in GROUPS for c, _ in items]
    labels = [f"{g}:{lbl}" for g, items in GROUPS for _, lbl in items]
    daily = df[cols].notna().groupby(df.index.normalize()).mean()  # days x cols
    cov = daily.values.T  # cols x days
    days = daily.index

    fig, ax = plt.subplots(figsize=(15, 16))
    x0, x1 = mdates.date2num(days[0]), mdates.date2num(days[-1])
    im = ax.imshow(cov, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1,
                   extent=[x0, x1, len(cols), 0], interpolation="nearest")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_yticks(np.arange(len(cols)) + 0.5)
    ax.set_yticklabels(labels, fontsize=7)
    row = 0
    for _, items in GROUPS[:-1]:
        row += len(items)
        ax.axhline(row, color="white", lw=1.5)
    ax.set_title("IMAOC7 dataset — daily data coverage (fraction of valid 5-min samples per day)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="valid fraction")
    fig.tight_layout()
    fig.savefig(PLOTS / "coverage_heatmap.png", dpi=130)
    plt.close(fig)
    return daily


def magnitude(df, cols):
    return np.sqrt((df[cols] ** 2).sum(axis=1, min_count=len(cols)))


def overview_timeseries(df):
    h = df.resample("1h").mean()  # lighten the plot; NaN gaps preserved
    panels = [
        ("OMNI |B| (nT)", magnitude(h, ["omni_bx_gse", "omni_by_gse", "omni_bz_gse"]), False),
        ("THEMIS-A |B| (nT)", magnitude(h, ["tha_bx_gsm", "tha_by_gsm", "tha_bz_gsm"]), False),
        ("THEMIS-B |B| (nT)", magnitude(h, ["thb_bx_gsm", "thb_by_gsm", "thb_bz_gsm"]), False),
        ("MMS1 |B| (nT, log)", magnitude(h, ["mms1_bx_gse", "mms1_by_gse", "mms1_bz_gse"]), True),
        ("MMS1 ion density (cm⁻³)", h["mms1_n"], False),
    ]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(15, 14), sharex=True)
    for ax, (label, series, logy) in zip(axes, panels):
        ax.plot(series.index, series.values, lw=0.4)
        ax.set_ylabel(label, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    ax = axes[-1]
    for st in ("TAM", "SOK", "EDA", "CLF", "KOU", "IPM", "PPT"):
        ax.plot(h.index, h[f"{st.lower()}_f"].values, lw=0.4, label=st)
    ax.set_ylabel("Ground |F| (nT)", fontsize=9)
    ax.legend(ncol=7, fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    axes[0].set_title("IMAOC7 dataset — signal overview (hourly means; gaps = no data)")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS / "overview_timeseries.png", dpi=130)
    plt.close(fig)


def main():
    df = pd.read_pickle("IMAOC7_summer_school_dataset.pkl")
    coverage_heatmap(df)
    overview_timeseries(df)
    overall = df.notna().mean().rename("coverage")
    print("Overall coverage by group:")
    for g, items in GROUPS:
        cols = [c for c, _ in items]
        print(f"  {g:9s} {overall[cols].mean():5.1%}")
    print("Wrote plots/coverage_heatmap.png and plots/overview_timeseries.png")


if __name__ == "__main__":
    main()
