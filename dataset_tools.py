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
from speasy.signal.resampling import resample as spz_resample, interpolate as spz_interpolate


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
            chunks.append(spz_resample(r, interval))
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
