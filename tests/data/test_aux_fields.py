"""Change-log invariants for the aux-field extraction on synthetic frames."""
from datetime import date, datetime, timedelta

import polars as pl

from trackfm.data.aux_fields import OUTPUT_SCHEMA, build_change_log, map_aux_columns

T0 = datetime(2024, 7, 15, 0, 30, 0)


def reports(mmsi, rows):
    """rows: list of dicts with minute offset + any aux field overrides."""
    base = {
        "nav_status": "Under way using engine",
        "rot": 0.0,
        "draught": 7.5,
        "destination": "AARHUS",
        "eta": "15/07/2024 18:00:00",
        "width": 25.0,
        "length": 180.0,
        "cargo_type": "Category X",
    }
    return pl.DataFrame({
        "timestamp": [T0 + timedelta(minutes=r["m"]) for r in rows],
        "mmsi": [mmsi] * len(rows),
        **{
            f: [r.get(f, base[f]) for r in rows]
            for f in base
        },
    })


def test_constant_fields_yield_single_row():
    df = reports(219000001, [{"m": i} for i in range(5)])
    out = build_change_log(df)
    assert out.height == 1
    assert out["timestamp"][0] == T0  # first report of the day


def test_draught_change_yields_two_rows():
    df = reports(219000001, [{"m": 0}, {"m": 10}, {"m": 20, "draught": 9.1}, {"m": 30, "draught": 9.1}])
    out = build_change_log(df)
    assert out.height == 2
    assert out["draught"].to_list() == [7.5, pl.Series([9.1], dtype=pl.Float32)[0]]
    assert out["timestamp"].to_list() == [T0, T0 + timedelta(minutes=20)]


def test_per_vessel_isolation_and_destination_normalization():
    a = reports(219000001, [{"m": 0}, {"m": 5, "destination": " rotterdam "}])
    b = reports(219000002, [{"m": 1}, {"m": 6}])
    out = build_change_log(pl.concat([a, b]))
    assert out.filter(pl.col("mmsi") == 219000001).height == 2
    assert out.filter(pl.col("mmsi") == 219000002).height == 1
    assert "ROTTERDAM" in out["destination"].to_list()


def test_implausible_values_nulled_and_day_filter():
    df = reports(219000001, [
        {"m": 0, "draught": 45.0, "width": 800.0},           # implausible -> null
        {"m": 24 * 60 + 5},                                   # next day, filtered out
    ])
    out = build_change_log(df, day=date(2024, 7, 15))
    assert out.height == 1
    assert out["draught"][0] is None
    assert out["width"][0] is None
    assert dict(out.schema) == dict(OUTPUT_SCHEMA)


def test_map_aux_columns_handles_spelling_drift():
    dma_2023 = ["# Timestamp", "MMSI", "Navigational status", "ROT", "Width",
                "Length", "Draught", "Destination", "ETA", "Cargo type"]
    renamed = ["Timestamp", "MMSI", "Nav Status", "Rate of turn", "Width",
               "Length", "Draft", "Destination", "ETA", "Cargo_type"]
    for header in (dma_2023, renamed):
        mapping = map_aux_columns(header)
        assert set(mapping.values()) == {
            "timestamp", "mmsi", "nav_status", "rot", "width", "length",
            "draught", "destination", "eta", "cargo_type",
        }
