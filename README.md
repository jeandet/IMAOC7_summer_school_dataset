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

---

*Last updated: March 2026*
