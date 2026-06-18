from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "forecast_base_joins_jan2023"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

WIND_SOLAR_PATH = BASE_DIR / "wind_solar_day_ahead_base_join_jan2023.csv"
DEMAND_PATH = BASE_DIR / "demand_aligned_to_wind_solar_publish_time_jan2023.csv"
OUT_PATH = BASE_DIR / "day_ahead_weather_demand_base_join_jan2023"

START = "2023-01-01"
END = "2023-01-31"


def read_csv_with_times(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="raise").astype("int16")
    for col in [c for c in df.columns if c.endswith("Time")]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def assert_unique_keys(df: pd.DataFrame, name: str, keys: list[str]) -> None:
    duplicate_rows = int(df.duplicated(keys, keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"{name} has {duplicate_rows} duplicate key rows on {keys}")


def build_join() -> pd.DataFrame:
    keys = ["settlementDate", "settlementPeriod", "targetStartTime"]
    wind_solar = read_csv_with_times(WIND_SOLAR_PATH)
    demand = read_csv_with_times(DEMAND_PATH)
    assert_unique_keys(wind_solar, "wind_solar", keys)
    assert_unique_keys(demand, "demand", keys)

    demand_cols = keys + [
        "windSolarPublishTime",
        "demandForecast",
        "transmissionDemandForecast",
        "demandForecast_publishTime",
        "demandForecastAgeMinutesAtWindSolarPublish",
        "windSolarLeadHoursToDelivery",
    ]
    out = wind_solar.merge(demand[demand_cols], on=keys, how="inner", validate="one_to_one")
    wind_solar_publish_cols = [
        "windOnshoreForecast_publishTime",
        "windOffshoreForecast_publishTime",
        "solarForecast_publishTime",
    ]
    all_wind_solar_publish_times_missing = out[wind_solar_publish_cols].isna().all(axis=1)
    out["windSolarComponentPublishTimesMatch"] = all_wind_solar_publish_times_missing | (
        (out["windOnshoreForecast_publishTime"] == out["windSolarPublishTime"])
        & (out["windOffshoreForecast_publishTime"] == out["windSolarPublishTime"])
        & (out["solarForecast_publishTime"] == out["windSolarPublishTime"])
    )
    out["demandPublishAfterWindSolarCutoff"] = out["demandForecast_publishTime"] > out["windSolarPublishTime"]
    out["windForecastRecomputed"] = out["windOnshoreForecast"] + out["windOffshoreForecast"]
    out["windForecastRecomputeDiff"] = out["windForecast"] - out["windForecastRecomputed"]
    return out.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)


def build_quality(joined: pd.DataFrame) -> pd.DataFrame:
    expected_rows = int(pd.date_range(START, END, freq="D").size * 48)
    checks = [
        ("expected_sp_rows", expected_rows),
        ("actual_sp_rows", len(joined)),
        ("missing_sp_rows", expected_rows - len(joined)),
        ("missing_demand_rows", int(joined["demandForecast"].isna().sum())),
        ("missing_wind_rows", int(joined["windForecast"].isna().sum())),
        ("missing_solar_rows", int(joined["solarForecast"].isna().sum())),
        ("negative_demand_rows", int((joined["demandForecast"] < 0).sum())),
        ("negative_wind_rows", int((joined["windForecast"] < 0).sum())),
        ("negative_solar_rows", int((joined["solarForecast"] < 0).sum())),
        (
            "component_publish_time_mismatch_rows",
            int((~joined["windSolarComponentPublishTimesMatch"].fillna(False)).sum()),
        ),
        ("demand_publish_after_wind_solar_cutoff_rows", int(joined["demandPublishAfterWindSolarCutoff"].sum())),
        (
            "max_abs_wind_forecast_recompute_diff",
            float(np.nanmax(np.abs(joined["windForecastRecomputeDiff"].to_numpy()))),
        ),
        (
            "median_wind_solar_lead_hours_to_delivery",
            float(joined["windSolarLeadHoursToDelivery"].median()),
        ),
        (
            "median_demand_age_minutes_at_wind_solar_cutoff",
            float(joined["demandForecastAgeMinutesAtWindSolarPublish"].median()),
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    joined = build_join()
    quality = build_quality(joined)

    joined.to_csv(OUT_PATH.with_suffix(".csv"), index=False)
    joined.to_parquet(OUT_PATH.with_suffix(".parquet"), index=False)
    quality.to_csv(DIAG_DIR / "day_ahead_weather_demand_base_join_quality_jan2023.csv", index=False)

    print(f"Wrote {len(joined)} SP rows to {OUT_PATH.with_suffix('.csv')}")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
