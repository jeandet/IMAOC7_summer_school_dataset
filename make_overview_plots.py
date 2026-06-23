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
    ("OMNI", [("proton_density_Proton density", "ni"), ("Pressure_Flow pressure", "Pdyn"),
              ("T_temperature", "T"), ("Vx_Vx Velocity, GSE", "Vx"),
              ("Vy_Vy Velocity, GSE", "Vy"), ("Vz_Vz Velocity, GSE", "Vz"),
              ("BX_GSE_Bx, GSE", "Bx"), ("BY_GSE_By, GSE", "By"), ("BZ_GSE_Bz, GSE", "Bz")]),
    ("THEMIS-A", [("tha_bs_gsm_bx", "Bx"), ("tha_bs_gsm_by", "By"), ("tha_bs_gsm_bz", "Bz"),
                  ("tha_v_i_vx", "Vx"), ("tha_v_i_vy", "Vy"), ("tha_v_i_vz", "Vz"),
                  ("tha_n_i_ion density", "ni")]),
    ("THEMIS-B", [("thb_bs_gsm_bx", "Bx"), ("thb_bs_gsm_by", "By"), ("thb_bs_gsm_bz", "Bz"),
                  ("thb_v_i_vx", "Vx"), ("thb_v_i_vy", "Vy"), ("thb_v_i_vz", "Vz"),
                  ("thb_n_i_ion density", "ni")]),
    ("MMS1", [("mms1_b_gse_bx", "Bx"), ("mms1_b_gse_by", "By"), ("mms1_b_gse_bz", "Bz"),
              ("mms1_dis_vgse_vx", "Vx"), ("mms1_dis_vgse_vy", "Vy"),
              ("mms1_dis_vgse_vz", "Vz"), ("mms1_dis_ni_density", "ni")]),
]
GROUPS += [(st, [(f"{st}{c}", c) for c in "XYZF"])
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
    # group separators
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
        ("OMNI |B| (nT)", magnitude(h, ["BX_GSE_Bx, GSE", "BY_GSE_By, GSE", "BZ_GSE_Bz, GSE"]), False),
        ("THEMIS-A |B| (nT)", magnitude(h, ["tha_bs_gsm_bx", "tha_bs_gsm_by", "tha_bs_gsm_bz"]), False),
        ("THEMIS-B |B| (nT)", magnitude(h, ["thb_bs_gsm_bx", "thb_bs_gsm_by", "thb_bs_gsm_bz"]), False),
        ("MMS1 |B| (nT, log)", magnitude(h, ["mms1_b_gse_bx", "mms1_b_gse_by", "mms1_b_gse_bz"]), True),
        ("MMS1 ion density (cm⁻³)", h["mms1_dis_ni_density"], False),
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
        ax.plot(h.index, h[f"{st}F"].values, lw=0.4, label=st)
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
    daily = coverage_heatmap(df)
    overview_timeseries(df)
    overall = df.notna().mean().rename("coverage")
    print("Overall coverage by group:")
    for g, items in GROUPS:
        cols = [c for c, _ in items]
        print(f"  {g:9s} {overall[cols].mean():5.1%}")
    print("Wrote plots/coverage_heatmap.png and plots/overview_timeseries.png")


if __name__ == "__main__":
    main()
