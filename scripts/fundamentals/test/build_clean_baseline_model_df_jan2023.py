from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "forecast_base_joins_jan2023"
RAW_SYSTEM_PRICE_DIR = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "system_price_niv"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

INPUT_PATH = BASE_DIR / "weather_demand_lolp_drm_realtime_lag4_jan2023.csv"
OUT_PATH = BASE_DIR / "clean_baseline_model_df_jan2023"

START = "2023-01-01"
END = "2023-01-31"


def files_for_window() -> list[Path]:
    files: list[Path] = []
    day = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while day <= end:
        path = RAW_SYSTEM_PRICE_DIR / f"{day.strftime('%Y-%m-%d')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Empty raw file: {path}")
        files.append(path)
        day += pd.Timedelta(days=1)
    return files


def read_system_price() -> pd.DataFrame:
    usecols = ["settlementDate", "settlementPeriod", "startTime", "systemSellPrice", "systemBuyPrice"]
    raw = pd.concat((pd.read_csv(path, usecols=usecols) for path in files_for_window()), ignore_index=True)
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="raise").astype("int16")
    raw["targetStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["systemSellPrice"] = pd.to_numeric(raw["systemSellPrice"], errors="coerce")
    raw["systemBuyPrice"] = pd.to_numeric(raw["systemBuyPrice"], errors="coerce")
    raw["systemPrice"] = raw["systemBuyPrice"].where(raw["systemBuyPrice"].eq(raw["systemSellPrice"]))
    out = raw[["settlementDate", "settlementPeriod", "targetStartTime", "systemPrice"]].copy()
    duplicate_rows = int(out.duplicated(["settlementDate", "settlementPeriod", "targetStartTime"], keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"System price source has {duplicate_rows} duplicate SP rows")
    return out


def build_clean_df() -> pd.DataFrame:
    base = pd.read_csv(INPUT_PATH)
    base["settlementPeriod"] = pd.to_numeric(base["settlementPeriod"], errors="raise").astype("int16")
    base["targetStartTime"] = pd.to_datetime(base["targetStartTime"], utc=True, errors="coerce")

    keep = [
        "settlementDate",
        "settlementPeriod",
        "targetStartTime",
        "solarForecast",
        "windForecast",
        "demandForecast",
        "lossOfLoadProbability",
        "deratedMargin",
        "nivLag4",
        "interconnectorFlowLag4",
    ]
    clean = base[keep].copy()
    system_price = read_system_price()
    clean = clean.merge(
        system_price,
        on=["settlementDate", "settlementPeriod", "targetStartTime"],
        how="left",
        validate="one_to_one",
    )
    return clean.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)


def build_quality(clean: pd.DataFrame) -> pd.DataFrame:
    expected_rows = int(pd.date_range(START, END, freq="D").size * 48)
    checks = [
        ("expected_sp_rows", expected_rows),
        ("actual_sp_rows", len(clean)),
        ("missing_sp_rows", expected_rows - len(clean)),
        ("missing_system_price_rows", int(clean["systemPrice"].isna().sum())),
        ("missing_solar_forecast_rows", int(clean["solarForecast"].isna().sum())),
        ("missing_wind_forecast_rows", int(clean["windForecast"].isna().sum())),
        ("missing_demand_forecast_rows", int(clean["demandForecast"].isna().sum())),
        ("missing_lolp_rows", int(clean["lossOfLoadProbability"].isna().sum())),
        ("missing_derated_margin_rows", int(clean["deratedMargin"].isna().sum())),
        ("missing_niv_lag4_rows", int(clean["nivLag4"].isna().sum())),
        ("missing_interconnector_lag4_rows", int(clean["interconnectorFlowLag4"].isna().sum())),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    clean = build_clean_df()
    quality = build_quality(clean)

    clean.to_csv(OUT_PATH.with_suffix(".csv"), index=False)
    clean.to_parquet(OUT_PATH.with_suffix(".parquet"), index=False)
    quality.to_csv(DIAG_DIR / "clean_baseline_model_df_quality_jan2023.csv", index=False)

    print(f"Wrote clean baseline model DF to {OUT_PATH.with_suffix('.csv')}")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
