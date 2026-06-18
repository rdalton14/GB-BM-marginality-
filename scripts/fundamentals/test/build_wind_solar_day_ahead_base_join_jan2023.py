from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "day_ahead_generation_wind_solar"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "forecast_base_joins_jan2023"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

START = "2023-01-01"
END = "2023-01-31"
PSR_TYPES = ["Wind Onshore", "Wind Offshore", "Solar"]


def files_for_window() -> list[Path]:
    files: list[Path] = []
    date = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while date <= end:
        path = RAW_DIR / f"{date.strftime('%Y-%m-%d')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            date += pd.Timedelta(days=1)
            continue
        files.append(path)
        date += pd.Timedelta(days=1)
    return files


def read_raw() -> pd.DataFrame:
    usecols = [
        "publishTime",
        "processType",
        "businessType",
        "psrType",
        "startTime",
        "settlementDate",
        "settlementPeriod",
        "quantity",
    ]
    raw = pd.concat((pd.read_csv(path, usecols=usecols) for path in files_for_window()), ignore_index=True)
    raw = raw[raw["psrType"].isin(PSR_TYPES)].copy()
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="raise").astype("int16")
    raw["targetStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")
    return raw


def build_expected_spine() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    day = pd.Timestamp(START, tz="UTC")
    end = pd.Timestamp(END, tz="UTC")
    while day <= end:
        for sp in range(1, 49):
            records.append(
                {
                    "settlementDate": day.strftime("%Y-%m-%d"),
                    "settlementPeriod": sp,
                    "targetStartTime": day + pd.Timedelta(minutes=30 * (sp - 1)),
                }
            )
        day += pd.Timedelta(days=1)
    spine = pd.DataFrame(records)
    spine["settlementPeriod"] = spine["settlementPeriod"].astype("int16")
    return spine


def build_base_join(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["settlementDate", "settlementPeriod", "targetStartTime"]
    selected = (
        raw.sort_values(keys + ["psrType", "publishTime"])
        .drop_duplicates(keys + ["psrType"], keep="last")
    )

    values = selected.pivot_table(index=keys, columns="psrType", values="quantity", aggfunc="first").reset_index()
    values = values.rename(
        columns={
            "Wind Onshore": "windOnshoreForecast",
            "Wind Offshore": "windOffshoreForecast",
            "Solar": "solarForecast",
        }
    )
    for col in ["windOnshoreForecast", "windOffshoreForecast", "solarForecast"]:
        if col not in values.columns:
            values[col] = np.nan

    values["windForecast"] = values["windOnshoreForecast"] + values["windOffshoreForecast"]

    publish = selected.pivot_table(index=keys, columns="psrType", values="publishTime", aggfunc="first").reset_index()
    publish = publish.rename(
        columns={
            "Wind Onshore": "windOnshoreForecast_publishTime",
            "Wind Offshore": "windOffshoreForecast_publishTime",
            "Solar": "solarForecast_publishTime",
        }
    )

    process = selected.pivot_table(index=keys, columns="psrType", values="processType", aggfunc="first").reset_index()
    process = process.rename(
        columns={
            "Wind Onshore": "windOnshoreForecast_processType",
            "Wind Offshore": "windOffshoreForecast_processType",
            "Solar": "solarForecast_processType",
        }
    )

    out = build_expected_spine().merge(values, on=keys, how="left", validate="one_to_one")
    out = out.merge(publish, on=keys, how="left", validate="one_to_one")
    out = out.merge(process, on=keys, how="left", validate="one_to_one")
    return out.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)


def build_quality(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    duplicate_component_keys = int(
        raw.duplicated(["settlementDate", "settlementPeriod", "targetStartTime", "psrType"], keep=False).sum()
    )
    expected_rows = int(pd.date_range(START, END, freq="D").size * 48)
    all_publish_times_missing = base[
        ["windOnshoreForecast_publishTime", "windOffshoreForecast_publishTime", "solarForecast_publishTime"]
    ].isna().all(axis=1)
    matching_publish_times = all_publish_times_missing | (
        (base["windOnshoreForecast_publishTime"] == base["windOffshoreForecast_publishTime"])
        & (base["windOnshoreForecast_publishTime"] == base["solarForecast_publishTime"])
    ).fillna(False)
    mismatched_publish_times = int((~matching_publish_times).sum())

    checks = [
        ("raw_rows", len(raw)),
        ("expected_sp_rows", expected_rows),
        ("actual_sp_rows", len(base)),
        ("missing_sp_rows", expected_rows - len(base)),
        ("duplicate_component_key_rows", duplicate_component_keys),
        ("missing_wind_onshore_rows", int(base["windOnshoreForecast"].isna().sum())),
        ("missing_wind_offshore_rows", int(base["windOffshoreForecast"].isna().sum())),
        ("missing_wind_total_rows", int(base["windForecast"].isna().sum())),
        ("missing_solar_rows", int(base["solarForecast"].isna().sum())),
        ("negative_wind_onshore_rows", int((base["windOnshoreForecast"] < 0).sum())),
        ("negative_wind_offshore_rows", int((base["windOffshoreForecast"] < 0).sum())),
        ("negative_wind_total_rows", int((base["windForecast"] < 0).sum())),
        ("negative_solar_rows", int((base["solarForecast"] < 0).sum())),
        ("mismatched_component_publish_time_rows", mismatched_publish_times),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_raw()
    base = build_base_join(raw)
    quality = build_quality(raw, base)

    base_path = OUT_DIR / "wind_solar_day_ahead_base_join_jan2023"
    base.to_csv(base_path.with_suffix(".csv"), index=False)
    base.to_parquet(base_path.with_suffix(".parquet"), index=False)

    quality.to_csv(DIAG_DIR / "wind_solar_day_ahead_base_join_quality_jan2023.csv", index=False)
    (
        raw.groupby(["psrType", "processType"], as_index=False)
        .size()
        .to_csv(DIAG_DIR / "wind_solar_day_ahead_base_join_process_types_jan2023.csv", index=False)
    )

    print(f"Wrote {len(base)} SP rows to {base_path.with_suffix('.csv')}")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
