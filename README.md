# IMAOC7 Summer School Dataset Repository

This repository contains the code and utilities required to assemble a multi‑series time
series dataset for the IMAOC7 summer school (themes: ionosphere, magnetosphere, solar wind).

The dataset covers January 2024 through January 2026 and will combine data from:

- solar wind monitors (OMNI, ARTEMIS, etc.)
- ground magnetometers (DST/SYM‑H, AE, IAGA‑2002 stations)
- magnetospheric spacecraft (THEMIS, MMS, Artemis when applicable)

Key features of the project:

1. data ingestion and format conversion (ASCII IAGA‑2002 → internal binary/CDF)
2. common one‑minute temporal resolution and handling of missing values (NaN/fill)
3. use of SPEASY-compatible binary storage for efficient volume and access
4. scripts to generate daily/monthly files and to merge subsets by location

The resulting dataset will be used during the summer school for machine learning
exercises (supervised classification and unsupervised event detection) and as a
teaching example of database construction.

All code, documentation, and utilities necessary to build, preprocess, and
store the data are tracked in this repository. Contributions are welcome via
issues and pull requests.

## Dataset columns

The assembled file (`IMAOC7_summer_school_dataset.csv` / `.pkl`) is indexed by UTC
timestamp on a **5-minute grid**; each 5-minute value is the NaN-aware mean of the
underlying high-rate samples (an empty bin is left as NaN — gaps are real, not bridged).
Column names are lowercase and contain no spaces or commas, so they load in any CSV reader.

| Source | Columns | Units | Notes |
|---|---|---|---|
| **OMNI** (solar wind at the bow-shock nose) | `omni_n`, `omni_pdyn`, `omni_t`, `omni_vx_gse`, `omni_vy_gse`, `omni_vz_gse`, `omni_bx_gse`, `omni_by_gse`, `omni_bz_gse` | cm⁻³, nPa, K, km/s (GSE), nT (GSE) | OMNI_HRO2 1-min; plasma ~37% gappy |
| **THEMIS-A** (`tha_…`) | `tha_bx_gsm`, `tha_by_gsm`, `tha_bz_gsm`, `tha_vx_gse`, `tha_vy_gse`, `tha_vz_gse`, `tha_n` | nT (GSM), km/s (GSE), cm⁻³ | near-Earth magnetosphere |
| **THEMIS-B** (`thb_…`) | same fields as THEMIS-A | — | ARTEMIS, lunar orbit (clean solar-wind \|B\| ~5–10 nT) |
| **MMS1** (`mms1_…`) | `mms1_bx_gse`, `mms1_by_gse`, `mms1_bz_gse`, `mms1_vx_gse`, `mms1_vy_gse`, `mms1_vz_gse`, `mms1_n` | nT (GSE), km/s (GSE), cm⁻³ | B from CDAWeb FGM survey; large \|B\| at perigee is real; FPI moments ~47% gappy |
| **Ground magnetometers** | `{station}_x`, `{station}_y`, `{station}_z`, `{station}_f` for `tam sok eda clf kou ipm ppt` | nT | BCMT IAGA-2002 1-min; coverage varies by station (EDA sparsest) |

Missing data is represented as empty/NaN. A companion `catalog.txt` lists magnetic-storm
intervals (Dst ≤ −50 nT).

---

*Last updated: June 2026*
