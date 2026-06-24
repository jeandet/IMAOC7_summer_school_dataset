import numpy as np

from speasy.products import SpeasyVariable, DataContainer, VariableTimeAxis
from speasy.signal.resampling import generate_time_vector, interpolate as spz_interpolate

from dataset_tools import interpolate_with_gaps


def _short_variable() -> SpeasyVariable:
    """A variable covering only 2024-06-01 .. 2024-06-10, hourly."""
    times = np.arange("2024-06-01", "2024-06-10", np.timedelta64(1, "h"),
                      dtype="datetime64[ns]")
    values = np.tile(np.arange(len(times), dtype=float)[:, None], (1, 3))
    return SpeasyVariable(axes=[VariableTimeAxis(times)],
                          values=DataContainer(values),
                          columns=["x", "y", "z"])


def test_plain_interpolate_flatlines_outside_coverage():
    # Documents the bug: np.interp holds edge values constant outside [t0, t1].
    var = _short_variable()
    tv = generate_time_vector("2024/01/01", "2025/01/01", 60. * 5)
    out = spz_interpolate(tv, var)
    before = tv < var.time.min()
    assert np.isfinite(out.values[before]).all()
    assert np.unique(out.values[before, 0]).size == 1  # frozen flat line


def test_interpolate_with_gaps_nans_outside_coverage():
    var = _short_variable()
    tv = generate_time_vector("2024/01/01", "2025/01/01", 60. * 5)
    out = interpolate_with_gaps(tv, var)

    outside = (tv < var.time.min()) | (tv > var.time.max())
    inside = ~outside

    assert np.isnan(out.values[outside]).all()
    assert np.isfinite(out.values[inside]).all()


def test_interpolate_with_gaps_nans_internal_gap():
    # Two data islands with a multi-day hole between them must NOT be bridged.
    a = np.arange("2024-06-01", "2024-06-03", np.timedelta64(1, "h"), dtype="datetime64[ns]")
    b = np.arange("2024-06-20", "2024-06-22", np.timedelta64(1, "h"), dtype="datetime64[ns]")
    times = np.concatenate([a, b])
    values = np.ones((len(times), 1))
    var = SpeasyVariable(axes=[VariableTimeAxis(times)],
                         values=DataContainer(values), columns=["x"])
    tv = generate_time_vector("2024/06/01", "2024/06/22", 60. * 5)
    out = interpolate_with_gaps(tv, var)

    in_gap = (tv > a[-1] + np.timedelta64(2, "h")) & (tv < b[0] - np.timedelta64(2, "h"))
    on_data = (tv >= a[0]) & (tv <= a[-1])
    assert np.isnan(out.values[in_gap]).all()      # hole stays empty, no ramp
    assert np.isfinite(out.values[on_data]).all()   # real data preserved


def test_bin_average_nan_aware_and_grid_aligned():
    from dataset_tools import bin_average
    times = np.arange("2024-01-01", "2024-01-01T00:20", np.timedelta64(1, "m"),
                      dtype="datetime64[ns]")  # 20 one-minute samples -> four 5-min bins
    vals = np.arange(20, dtype=float)[:, None]
    vals[2] = np.nan        # scattered hole in bin 0 -> averaged away, bin stays valid
    vals[5:10] = np.nan     # bin 1 entirely empty -> NaN, not bridged
    var = SpeasyVariable(axes=[VariableTimeAxis(times)],
                         values=DataContainer(vals), columns=["x"])
    out = bin_average(var, 300., "2024-01-01")
    o = np.asarray(out.values).flatten()

    assert len(o) == 4
    assert np.isclose(o[0], np.mean([0, 1, 3, 4]))   # NaN-aware mean
    assert np.isnan(o[1])                            # empty bin stays NaN
    assert np.isclose(o[2], np.mean([10, 11, 12, 13, 14]))
    assert str(out.time[0]) == "2024-01-01T00:00:00.000000000"  # grid-aligned


def test_replace_fillval_masks_sentinels():
    # OMNI CDFs declare FILLVAL (e.g. 9999.99) that must become NaN, not a real number.
    times = np.arange("2024-01-01", "2024-01-01T05", np.timedelta64(1, "h"),
                      dtype="datetime64[ns]")
    values = np.array([1.0, 9999.99, 2.0, 9999.99, 3.0])[:, None]
    var = SpeasyVariable(axes=[VariableTimeAxis(times)],
                         values=DataContainer(values, meta={"FILLVAL": [9999.99]}),
                         columns=["b"])
    out = var.replace_fillval_by_nan()
    v = np.asarray(out.values).flatten()
    assert np.isnan(v[[1, 3]]).all()
    assert np.allclose(v[[0, 2, 4]], [1.0, 2.0, 3.0])
