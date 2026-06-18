from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DEMAND_DIR = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "day_ahead_demand_forecast_evolution"
WIND_SOLAR_BASE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "forecast_base_joins_jan2023"
    / "wind_solar_day_ahead_base_join_jan2023.csv"
)
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "forecast_base_joins_jan2023"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

START = "2023-01-01"
END = "2023-01-31"


def files_for_window() -> list[Path]:
    files: list[Path] = []
    date = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while date <= end:
        path = RAW_DEMAND_DIR / f"{date.strftime('%Y-%m-%d')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Empty raw file: {path}")
        files.append(path)
        date += pd.Timedelta(days=1)
    return files


def read_wind_solar_base() -> pd.DataFrame:
    base = pd.read_csv(WIND_SOLAR_BASE)
    base["settlementPeriod"] = pd.to_numeric(base["settlementPeriod"], errors="raise").astype("int16")
    base["targetStartTime"] = pd.to_datetime(base["targetStartTime"], utc=True, errors="coerce")
    publish_cols = [
        "windOnshoreForecast_publishTime",
        "windOffshoreForecast_publishTime",
        "solarForecast_publishTime",
    ]
    for col in publish_cols:
        base[col] = pd.to_datetime(base[col], utc=True, errors="coerce")

    all_publish_times_missing = base[publish_cols].isna().all(axis=1)
    matching_publish_times = all_publish_times_missing | (
        (base["windOnshoreForecast_publishTime"] == base["windOffshoreForecast_publishTime"])
        & (base["windOnshoreForecast_publishTime"] == base["solarForecast_publishTime"])
    ).fillna(False)
    if not matching_publish_times.all():
        bad = int((~matching_publish_times).sum())
        raise ValueError(f"Wind/solar component publish times do not match for {bad} SP rows")

    base["windSolarPublishTime"] = base["windOnshoreForecast_publishTime"]
    return base[["settlementDate", "settlementPeriod", "targetStartTime", "windSolarPublishTime"]]


def read_demand_raw() -> pd.DataFrame:
    usecols = [
        "startTime",
        "settlementDate",
        "settlementPeriod",
        "boundary",
        "publishTime",
        "transmissionSystemDemand",
        "nationalDemand",
    ]
    raw = pd.concat((pd.read_csv(path, usecols=usecols) for path in files_for_window()), ignore_index=True)
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="raise").astype("int16")
    raw["targetStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["demandForecast_publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["demandForecast"] = pd.to_numeric(raw["nationalDemand"], errors="coerce")
    raw["transmissionDemandForecast"] = pd.to_numeric(raw["transmissionSystemDemand"], errors="coerce")
    raw = raw[raw["boundary"].eq("N")].copy()
    raw = raw[raw["demandForecast"].fillna(0) != 0]
    return raw[
        [
            "settlementDate",
            "settlementPeriod",
            "targetStartTime",
            "demandForecast_publishTime",
            "demandForecast",
            "transmissionDemandForecast",
        ]
    ]


def align_demand_to_wind_solar_publish_time(wind_solar: pd.DataFrame, demand_raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["settlementDate", "settlementPeriod", "targetStartTime"]
    candidates = wind_solar.merge(demand_raw, on=keys, how="left", validate="one_to_many")
    candidates = candidates[candidates["demandForecast_publishTime"] <= candidates["windSolarPublishTime"]]
    selected = (
        candidates.sort_values(keys + ["demandForecast_publishTime"])
        .drop_duplicates(keys, keep="last")
        .copy()
    )
    selected["demandForecastAgeMinutesAtWindSolarPublish"] = (
        selected["windSolarPublishTime"] - selected["demandForecast_publishTime"]
    ).dt.total_seconds() / 60.0
    selected["windSolarLeadHoursToDelivery"] = (
        selected["targetStartTime"] - selected["windSolarPublishTime"]
    ).dt.total_seconds() / 3600.0

    out = wind_solar.merge(
        selected[
            keys
            + [
                "demandForecast",
                "transmissionDemandForecast",
                "demandForecast_publishTime",
                "demandForecastAgeMinutesAtWindSolarPublish",
                "windSolarLeadHoursToDelivery",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    return out.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)


def build_quality(aligned: pd.DataFrame) -> pd.DataFrame:
    leakage = int(
        (
            aligned["demandForecast_publishTime"].notna()
            & (aligned["demandForecast_publishTime"] > aligned["windSolarPublishTime"])
        ).sum()
    )
    checks = [
        ("expected_sp_rows", int(pd.date_range(START, END, freq="D").size * 48)),
        ("actual_sp_rows", len(aligned)),
        ("missing_demand_rows", int(aligned["demandForecast"].isna().sum())),
        ("publish_after_wind_solar_cutoff_rows", leakage),
        (
            "min_demand_age_minutes_at_cutoff",
            aligned["demandForecastAgeMinutesAtWindSolarPublish"].min(),
        ),
        (
            "median_demand_age_minutes_at_cutoff",
            aligned["demandForecastAgeMinutesAtWindSolarPublish"].median(),
        ),
        (
            "max_demand_age_minutes_at_cutoff",
            aligned["demandForecastAgeMinutesAtWindSolarPublish"].max(),
        ),
        ("min_wind_solar_lead_hours_to_delivery", aligned["windSolarLeadHoursToDelivery"].min()),
        ("median_wind_solar_lead_hours_to_delivery", aligned["windSolarLeadHoursToDelivery"].median()),
        ("max_wind_solar_lead_hours_to_delivery", aligned["windSolarLeadHoursToDelivery"].max()),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    wind_solar = read_wind_solar_base()
    demand_raw = read_demand_raw()
    aligned = align_demand_to_wind_solar_publish_time(wind_solar, demand_raw)
    quality = build_quality(aligned)

    out_path = OUT_DIR / "demand_aligned_to_wind_solar_publish_time_jan2023"
    aligned.to_csv(out_path.with_suffix(".csv"), index=False)
    aligned.to_parquet(out_path.with_suffix(".parquet"), index=False)
    quality.to_csv(
        DIAG_DIR / "demand_aligned_to_wind_solar_publish_time_quality_jan2023.csv",
        index=False,
    )

    print(f"Wrote {len(aligned)} SP rows to {out_path.with_suffix('.csv')}")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
