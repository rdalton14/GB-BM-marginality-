from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "fundamentals"
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "forecast_base_joins_jan2023"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

BASE_PATH = BASE_DIR / "weather_demand_lolp_drm_2h_base_join_jan2023.csv"
REALTIME_PATH = BASE_DIR / "realtime_niv_interconnector_lag4_jan2023"
OUT_PATH = BASE_DIR / "weather_demand_lolp_drm_realtime_lag4_jan2023"

START = "2023-01-01"
END = "2023-01-31"
FORECAST_HORIZON_HOURS = 2
LAG_SETTLEMENT_PERIODS = 4


def files_for_window(raw_name: str) -> list[Path]:
    files: list[Path] = []
    date = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while date <= end:
        path = RAW_ROOT / raw_name / f"{date.strftime('%Y-%m-%d')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Empty raw file: {path}")
        files.append(path)
        date += pd.Timedelta(days=1)
    return files


def read_base() -> pd.DataFrame:
    base = pd.read_csv(BASE_PATH)
    base["settlementPeriod"] = pd.to_numeric(base["settlementPeriod"], errors="raise").astype("int16")
    for col in [c for c in base.columns if c.endswith("Time")]:
        base[col] = pd.to_datetime(base[col], utc=True, errors="coerce")
    base["modelAsOfTime"] = base["targetStartTime"] - pd.to_timedelta(FORECAST_HORIZON_HOURS, unit="h")
    return base


def read_niv() -> pd.DataFrame:
    usecols = [
        "settlementDate",
        "settlementPeriod",
        "startTime",
        "netImbalanceVolume",
    ]
    raw = pd.concat(
        (pd.read_csv(path, usecols=usecols) for path in files_for_window("system_price_niv")),
        ignore_index=True,
    )
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="raise").astype("int16")
    raw["sourceStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["nivLag4"] = pd.to_numeric(raw["netImbalanceVolume"], errors="coerce")
    raw = raw[["sourceStartTime", "nivLag4"]]
    duplicate_rows = int(raw.duplicated(["sourceStartTime"], keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"NIV source has {duplicate_rows} duplicate sourceStartTime rows")
    return raw


def read_interconnector_flow() -> pd.DataFrame:
    usecols = [
        "publishTime",
        "startTime",
        "settlementDate",
        "settlementPeriod",
        "interconnectorName",
        "generation",
    ]
    raw = pd.concat(
        (pd.read_csv(path, usecols=usecols) for path in files_for_window("interconnector_flows")),
        ignore_index=True,
    )
    raw["sourceStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["generation"] = pd.to_numeric(raw["generation"], errors="coerce")
    duplicate_component_rows = int(raw.duplicated(["sourceStartTime", "interconnectorName"], keep=False).sum())
    if duplicate_component_rows:
        raise ValueError(f"Interconnector source has {duplicate_component_rows} duplicate component rows")

    grouped = (
        raw.groupby("sourceStartTime", as_index=False)
        .agg(
            interconnectorFlowLag4=("generation", "sum"),
            interconnectorLag4PublishTime=("publishTime", "max"),
            interconnectorLag4ComponentCount=("interconnectorName", "nunique"),
        )
        .sort_values("sourceStartTime")
    )
    return grouped


def build_realtime_block(base: pd.DataFrame) -> pd.DataFrame:
    keys = ["settlementDate", "settlementPeriod", "targetStartTime", "modelAsOfTime"]
    block = base[keys].copy()
    block["lag4SourceStartTime"] = block["targetStartTime"] - pd.to_timedelta(LAG_SETTLEMENT_PERIODS * 30, unit="m")
    block["lag4SourceSettlementDate"] = block["lag4SourceStartTime"].dt.strftime("%Y-%m-%d")
    block["lag4SourceSettlementPeriod"] = (
        block["lag4SourceStartTime"].dt.hour * 2
        + (block["lag4SourceStartTime"].dt.minute // 30)
        + 1
    ).astype("int16")

    niv = read_niv()
    interconnector = read_interconnector_flow()
    block = block.merge(niv, left_on="lag4SourceStartTime", right_on="sourceStartTime", how="left", validate="many_to_one")
    block = block.drop(columns=["sourceStartTime"])
    block = block.merge(
        interconnector,
        left_on="lag4SourceStartTime",
        right_on="sourceStartTime",
        how="left",
        validate="many_to_one",
    )
    block = block.drop(columns=["sourceStartTime"])
    return block.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)


def build_quality(joined: pd.DataFrame) -> pd.DataFrame:
    expected_rows = int(pd.date_range(START, END, freq="D").size * 48)
    checks = [
        ("expected_sp_rows", expected_rows),
        ("actual_sp_rows", len(joined)),
        ("missing_sp_rows", expected_rows - len(joined)),
        ("missing_niv_lag4_rows", int(joined["nivLag4"].isna().sum())),
        ("missing_interconnector_lag4_rows", int(joined["interconnectorFlowLag4"].isna().sum())),
        ("lag_settlement_periods", LAG_SETTLEMENT_PERIODS),
        ("lag_hours_from_target_start_time", LAG_SETTLEMENT_PERIODS * 0.5),
        ("median_interconnector_component_count", float(joined["interconnectorLag4ComponentCount"].median())),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    base = read_base()
    realtime = build_realtime_block(base)
    joined = base.merge(
        realtime,
        on=["settlementDate", "settlementPeriod", "targetStartTime", "modelAsOfTime"],
        how="left",
        validate="one_to_one",
    )
    quality = build_quality(joined)

    realtime.to_csv(REALTIME_PATH.with_suffix(".csv"), index=False)
    realtime.to_parquet(REALTIME_PATH.with_suffix(".parquet"), index=False)
    joined.to_csv(OUT_PATH.with_suffix(".csv"), index=False)
    joined.to_parquet(OUT_PATH.with_suffix(".parquet"), index=False)
    quality.to_csv(DIAG_DIR / "weather_demand_lolp_drm_realtime_lag4_quality_jan2023.csv", index=False)

    print(f"Wrote realtime lag4 block to {REALTIME_PATH.with_suffix('.csv')}")
    print(f"Wrote merged rows to {OUT_PATH.with_suffix('.csv')}")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
