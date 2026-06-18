from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_LOLP_DRM_DIR = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "lolpdrm"
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "forecast_base_joins_jan2023"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

WEATHER_DEMAND_PATH = BASE_DIR / "day_ahead_weather_demand_base_join_jan2023.csv"
LOLP_DRM_2H_PATH = BASE_DIR / "lolp_drm_2h_base_join_jan2023"
OUT_PATH = BASE_DIR / "weather_demand_lolp_drm_2h_base_join_jan2023"

START = "2023-01-01"
END = "2023-01-31"
FORECAST_HORIZON = 2


def files_for_window() -> list[Path]:
    files: list[Path] = []
    date = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while date <= end:
        path = RAW_LOLP_DRM_DIR / f"{date.strftime('%Y-%m-%d')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Empty raw file: {path}")
        files.append(path)
        date += pd.Timedelta(days=1)
    return files


def read_base() -> pd.DataFrame:
    base = pd.read_csv(WEATHER_DEMAND_PATH)
    base["settlementPeriod"] = pd.to_numeric(base["settlementPeriod"], errors="raise").astype("int16")
    for col in [c for c in base.columns if c.endswith("Time")]:
        base[col] = pd.to_datetime(base[col], utc=True, errors="coerce")
    return base


def build_lolp_drm_2h() -> pd.DataFrame:
    raw = pd.concat((pd.read_csv(path) for path in files_for_window()), ignore_index=True)
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="raise").astype("int16")
    raw["forecastHorizon"] = pd.to_numeric(raw["forecastHorizon"], errors="raise").astype("int16")
    raw = raw[raw["forecastHorizon"].eq(FORECAST_HORIZON)].copy()
    raw["targetStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["lolpDrmPublishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["lolpDrmPublishingPeriodCommencingTime"] = pd.to_datetime(
        raw["publishingPeriodCommencingTime"], utc=True, errors="coerce"
    )
    raw["lossOfLoadProbability"] = pd.to_numeric(raw["lossOfLoadProbability"], errors="coerce")
    raw["deratedMargin"] = pd.to_numeric(raw["deratedMargin"], errors="coerce")
    raw["lolpDrmActualLeadHoursToDelivery"] = (
        raw["targetStartTime"] - raw["lolpDrmPublishTime"]
    ).dt.total_seconds() / 3600.0
    raw["lolpDrmPublishLagMinutes"] = (
        raw["lolpDrmPublishTime"] - raw["lolpDrmPublishingPeriodCommencingTime"]
    ).dt.total_seconds() / 60.0

    keys = ["settlementDate", "settlementPeriod", "targetStartTime"]
    duplicate_rows = int(raw.duplicated(keys, keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"LOLP/DRM 2h has {duplicate_rows} duplicate key rows")

    cols = keys + [
        "forecastHorizon",
        "lossOfLoadProbability",
        "deratedMargin",
        "lolpDrmPublishTime",
        "lolpDrmPublishingPeriodCommencingTime",
        "lolpDrmActualLeadHoursToDelivery",
        "lolpDrmPublishLagMinutes",
    ]
    return raw[cols].sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)


def build_quality(joined: pd.DataFrame) -> pd.DataFrame:
    expected_rows = int(pd.date_range(START, END, freq="D").size * 48)
    checks = [
        ("expected_sp_rows", expected_rows),
        ("actual_sp_rows", len(joined)),
        ("missing_sp_rows", expected_rows - len(joined)),
        ("missing_lolp_rows", int(joined["lossOfLoadProbability"].isna().sum())),
        ("missing_drm_rows", int(joined["deratedMargin"].isna().sum())),
        ("negative_lolp_rows", int((joined["lossOfLoadProbability"] < 0).sum())),
        ("lolp_above_one_rows", int((joined["lossOfLoadProbability"] > 1).sum())),
        ("negative_drm_rows", int((joined["deratedMargin"] < 0).sum())),
        (
            "lolp_drm_publish_after_wind_solar_cutoff_rows",
            int((joined["lolpDrmPublishTime"] > joined["windSolarPublishTime"]).sum()),
        ),
        ("median_lolp_drm_actual_lead_hours", float(joined["lolpDrmActualLeadHoursToDelivery"].median())),
        ("median_lolp_drm_publish_lag_minutes", float(joined["lolpDrmPublishLagMinutes"].median())),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    base = read_base()
    lolp_drm = build_lolp_drm_2h()

    keys = ["settlementDate", "settlementPeriod", "targetStartTime"]
    joined = base.merge(lolp_drm, on=keys, how="inner", validate="one_to_one")
    quality = build_quality(joined)

    lolp_drm.to_csv(LOLP_DRM_2H_PATH.with_suffix(".csv"), index=False)
    lolp_drm.to_parquet(LOLP_DRM_2H_PATH.with_suffix(".parquet"), index=False)
    joined.to_csv(OUT_PATH.with_suffix(".csv"), index=False)
    joined.to_parquet(OUT_PATH.with_suffix(".parquet"), index=False)
    quality.to_csv(DIAG_DIR / "weather_demand_lolp_drm_2h_base_join_quality_jan2023.csv", index=False)

    print(f"Wrote LOLP/DRM 2h rows to {LOLP_DRM_2H_PATH.with_suffix('.csv')}")
    print(f"Wrote merged rows to {OUT_PATH.with_suffix('.csv')}")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
