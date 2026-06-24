import time
import urllib.error
import urllib.request
import warnings
from datetime import timedelta
from io import StringIO

import numpy as np
import pandas as pd

import speasy as spz
from speasy.core import AnyDateTimeType, make_utc_datetime
from speasy.products import SpeasyVariable, DataContainer, VariableTimeAxis
from speasy.products.variable import merge
from speasy.signal.resampling import interpolate as spz_interpolate


# Clean, CSV-safe column names (no spaces/commas) for the assembled dataset.
# The raw speasy-derived names (e.g. "Vx_Vx Velocity, GSE") embed commas, which
# break naive CSV parsers. Frames: THEMIS/MMS velocities are GSE, THEMIS B is GSM.
COLUMN_RENAME = {
    "proton_density_Proton density": "omni_n", "Pressure_Flow pressure": "omni_pdyn",
    "T_temperature": "omni_t",
    "Vx_Vx Velocity, GSE": "omni_vx_gse", "Vy_Vy Velocity, GSE": "omni_vy_gse",
    "Vz_Vz Velocity, GSE": "omni_vz_gse",
    "BX_GSE_Bx, GSE": "omni_bx_gse", "BY_GSE_By, GSE": "omni_by_gse",
    "BZ_GSE_Bz, GSE": "omni_bz_gse",
    "tha_bs_gsm_bx": "tha_bx_gsm", "tha_bs_gsm_by": "tha_by_gsm", "tha_bs_gsm_bz": "tha_bz_gsm",
    "tha_v_i_vx": "tha_vx_gse", "tha_v_i_vy": "tha_vy_gse", "tha_v_i_vz": "tha_vz_gse",
    "tha_n_i_ion density": "tha_n",
    "thb_bs_gsm_bx": "thb_bx_gsm", "thb_bs_gsm_by": "thb_by_gsm", "thb_bs_gsm_bz": "thb_bz_gsm",
    "thb_v_i_vx": "thb_vx_gse", "thb_v_i_vy": "thb_vy_gse", "thb_v_i_vz": "thb_vz_gse",
    "thb_n_i_ion density": "thb_n",
    "mms1_b_gse_bx": "mms1_bx_gse", "mms1_b_gse_by": "mms1_by_gse", "mms1_b_gse_bz": "mms1_bz_gse",
    "mms1_dis_vgse_vx": "mms1_vx_gse", "mms1_dis_vgse_vy": "mms1_vy_gse",
    "mms1_dis_vgse_vz": "mms1_vz_gse", "mms1_dis_ni_density": "mms1_n",
    **{f"{st}{c}": f"{st.lower()}_{c.lower()}"
       for st in ("TAM", "SOK", "EDA", "CLF", "KOU", "IPM", "PPT") for c in "XYZF"},
}


def interpolate_with_gaps(time_vector: np.ndarray, var: SpeasyVariable,
                          max_gap: np.timedelta64 = np.timedelta64(1, "h")) -> SpeasyVariable:
    """Interpolate ``var`` onto ``time_vector`` while preserving missing coverage.

    ``numpy.interp`` (used by speasy's interpolate) silently bridges missing data:
    outside ``[t0, t1]`` it holds the first/last sample constant (flat lines), and
    across an internal time-axis gap it draws a straight linear ramp. Both hide the
    absence of data. We mask back to NaN any output sample that is outside the
    variable's coverage or farther than ``max_gap`` from the nearest real sample,
    so a multi-day gap shows as NaN rather than a fake ramp. Short dropouts (up to
    ``max_gap``) are still bridged for usability.
    """
    src = var.time
    out = spz_interpolate(time_vector, var)
    idx = np.searchsorted(src, time_vector)
    left = np.clip(idx - 1, 0, len(src) - 1)
    right = np.clip(idx, 0, len(src) - 1)
    dist = np.minimum(np.abs(time_vector - src[left]), np.abs(time_vector - src[right]))
    mask = (time_vector < src[0]) | (time_vector > src[-1]) | (dist > max_gap)
    out.values[mask] = np.nan
    return out


def bin_average(var: SpeasyVariable, interval_s: float,
                origin: AnyDateTimeType) -> SpeasyVariable:
    """Downsample to a regular ``interval_s`` grid by the NaN-aware mean of the samples
    in each ``[t, t+interval)`` bin, aligned to ``origin``. An empty bin becomes NaN.

    Unlike interpolation-based resampling this neither propagates scattered short gaps
    (which would over-count missing data, e.g. OMNI's 1-min plasma holes) nor bridges
    real multi-sample gaps with a ramp (which would hide them, e.g. MMS FPI orbit gaps).
    Bins are aligned to ``origin`` so every source lands on the same global grid.
    """
    times = pd.to_datetime(var.time)
    origin_ts = pd.Timestamp(origin)
    if origin_ts.tzinfo is not None:          # index is tz-naive UTC; match it
        origin_ts = origin_ts.tz_localize(None)
    df = pd.DataFrame(np.asarray(var.values, dtype=float), index=times)
    binned = df.resample(f"{int(interval_s)}s", origin=origin_ts).mean()
    return SpeasyVariable(
        axes=[VariableTimeAxis(binned.index.values.astype("datetime64[ns]"))],
        values=DataContainer(binned.values, name=var.name),
        columns=list(var.columns),
    )


def fetch_resampled_chunked(uid: str, start: AnyDateTimeType, stop: AnyDateTimeType,
                            interval: float, chunk_days: int = 7,
                            mask_fill: bool = True) -> SpeasyVariable | None:
    """Fetch a high-rate product in time chunks, resampling each chunk to ``interval``
    before fetching the next, so the full-resolution data is never held whole.

    Needed for products like the FGM survey (~16 Hz) where a single multi-year request
    is too large and can be silently truncated. ``mask_fill`` replaces declared FILLVAL
    sentinels with NaN *before* resampling, so they do not poison the chunk averages.
    """
    start, stop = make_utc_datetime(start), make_utc_datetime(stop)
    chunks = []
    t = start
    while t < stop:
        t_next = min(t + timedelta(days=chunk_days), stop)
        r = spz.get_data(uid, t, t_next)
        if r is not None and len(r.time):
            if mask_fill:
                r = r.replace_fillval_by_nan()
            chunks.append(bin_average(r, interval, start))
        t = t_next
    return merge(chunks) if chunks else None


def fetch_mms1_fgm_b_gse(start: AnyDateTimeType, stop: AnyDateTimeType,
                         interval: float, chunk_days: int = 7) -> SpeasyVariable | None:
    """MMS1 FGM survey B in GSE from CDAWeb, resampled to ``interval``.

    Uses CDAWeb (covers to 2026) rather than AMDA's ``mms1_b_gse`` (stops 2025-09-03).
    Keeps only Bx/By/Bz (drops the |B| 4th column) and names the variable ``mms1_b_gse``
    with ``bx/by/bz`` columns so the assembled dataset keeps its ``mms1_b_gse_*`` schema.
    """
    raw = fetch_resampled_chunked("cda/MMS1_FGM_SRVY_L2/mms1_fgm_b_gse_srvy_l2",
                                  start, stop, interval, chunk_days)
    if raw is None:
        return None
    return SpeasyVariable(
        axes=[VariableTimeAxis(raw.time)],
        values=DataContainer(np.asarray(raw.values)[:, :3], name="mms1_b_gse"),
        columns=["bx", "by", "bz"],
    )


def _parse_ground_mag(text: str) -> SpeasyVariable:
    """Parse one BCMT IAGA-2002 minute day-file into a SpeasyVariable."""
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("DATE"))
    cols = lines[hdr].split()
    if cols[-1] == "|":
        cols = cols[:-1]
    df = pd.read_csv(StringIO("\n".join(lines[hdr + 1:])), sep=r"\s+",
                     header=None, names=cols)
    df["DATETIME"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
    df = df.set_index("DATETIME").drop(columns=["DATE", "TIME", "DOY"])
    df = df.replace({99999.00: np.nan, 88888.00: np.nan})  # IAGA-2002 fill values
    times = np.array([np.datetime64(int(d.timestamp() * 1e9), "ns") for d in df.index])
    return SpeasyVariable(axes=[VariableTimeAxis(times)],
                          values=DataContainer(df.values), columns=list(df.columns))


def _bcmt_url(station: str, day) -> str:
    s = station.lower()
    return (f"https://www.bcmt.fr/DATABANK/VARIATION/{s}/min/"
            f"{day:%Y}/{s}{day:%Y%m%d}vmin.min")


def _fetch_ground_mag_day(url: str, retries: int = 3, backoff: float = 2.0):
    """Fetch one day-file, retrying transient errors. Returns None for a genuine
    404 (day not published) or after exhausting retries on transient failures."""
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            return _parse_ground_mag(text)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # genuinely missing day, not worth retrying
            last = e
        except Exception as e:  # timeout / connection reset / parse error
            last = e
        time.sleep(backoff * (attempt + 1))
    warnings.warn(f"giving up on {url}: {last}")
    return None


def load_ground_mag(start: AnyDateTimeType, stop: AnyDateTimeType, station: str):
    """Load BCMT ground-magnetometer minute data for one station over [start, stop).

    One file per UTC day; transient download failures are retried so a flaky day
    no longer silently truncates a station's coverage.
    """
    start, stop = make_utc_datetime(start), make_utc_datetime(stop)
    d0, d1 = start.date(), stop.date()
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

    variables = [v for day in days
                 if (v := _fetch_ground_mag_day(_bcmt_url(station, day))) is not None]
    if not variables:
        return None
    return merge(variables)[start:stop]
