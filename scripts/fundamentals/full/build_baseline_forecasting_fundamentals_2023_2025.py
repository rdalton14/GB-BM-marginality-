from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "fundamentals"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "full_2023_2025"
BROAD_PROCESSED_DIR = PROCESSED_ROOT / "fundamentals"
BASELINE_DIR = PROCESSED_ROOT / "baseline_forecasting_fundamentals"
DIAGNOSTICS_ROOT = PROJECT_ROOT / "data" / "diagnostics"
BASELINE_DIAG_DIR = DIAGNOSTICS_ROOT / "baseline_forecasting_fundamentals"

START = "2023-01-01"
END = "2025-12-31"
HORIZONS = [1, 2, 4, 8, 12]
EXPECTED_ROWS = 52_608
RAW_READ_ISSUES: list[dict[str, object]] = []

LINEAGE_CSV = BASELINE_DIAG_DIR / "baseline_fundamentals_source_lineage_2023_2025.csv"
AVAILABILITY_CSV = BASELINE_DIAG_DIR / "baseline_fundamentals_horizon_availability_2023_2025.csv"
LEAKAGE_CSV = BASELINE_DIAG_DIR / "baseline_fundamentals_leakage_flags_2023_2025.csv"
SUMMARY_MD = BASELINE_DIAG_DIR / "baseline_fundamentals_quality_summary_2023_2025.md"


def expected_spine() -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D")
    spine = pd.MultiIndex.from_product(
        [dates.strftime("%Y-%m-%d"), range(1, 49)],
        names=["settlementDate", "settlementPeriod"],
    ).to_frame(index=False)
    spine["settlementPeriod"] = spine["settlementPeriod"].astype("int16")
    return spine


def read_raw_dir(name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    files = sorted((RAW_ROOT / name).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw files found for {name}")
    frames: list[pd.DataFrame] = []
    for path in files:
        if path.stat().st_size == 0:
            RAW_READ_ISSUES.append(
                {
                    "rawFolder": name,
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "issue": "empty_raw_file",
                    "detail": "Skipped empty cached CSV; downstream coverage audit should show missing SPs.",
                }
            )
            continue
        frames.append(pd.read_csv(path, usecols=usecols))
    if not frames:
        raise ValueError(f"All raw files were empty for {name}")
    return pd.concat(frames, ignore_index=True)


def latest_nonempty_archive() -> Path | None:
    archive_parent = PROCESSED_ROOT / "archive"
    if not archive_parent.exists():
        return None
    candidates = [path for path in archive_parent.glob("baseline_fundamentals_purge_*") if any(path.rglob("*.*"))]
    return sorted(candidates)[-1] if candidates else None


def archive_broad_outputs() -> tuple[Path | None, int]:
    files_to_archive = [
        PROCESSED_ROOT / "system_fundamentals_2023_2025.csv",
        PROCESSED_ROOT / "system_fundamentals_2023_2025.parquet",
        PROCESSED_ROOT / "system_fundamentals_model_2023_2025.csv",
        PROCESSED_ROOT / "system_fundamentals_model_2023_2025.parquet",
    ]
    non_baseline_processed = [
        BROAD_PROCESSED_DIR / "actual_total_load_2023_2025.csv",
        BROAD_PROCESSED_DIR / "initial_demand_outturn_2023_2025.csv",
        BROAD_PROCESSED_DIR / "actual_generation_wind_solar_2023_2025.csv",
        BROAD_PROCESSED_DIR / "apx_mid_2023_2025.csv",
    ]
    broad_diagnostics = []
    for subdir in ["audits", "coverage", "missingness"]:
        diag_dir = DIAGNOSTICS_ROOT / subdir
        if diag_dir.exists():
            broad_diagnostics.extend(
                path
                for path in diag_dir.glob("*.csv")
                if any(token in path.name for token in ["system_fundamentals", "actual_total", "actual_generation", "apx_mid"])
            )

    existing = [src for src in [*files_to_archive, *non_baseline_processed, *broad_diagnostics] if src.exists()]
    if not existing:
        return latest_nonempty_archive(), 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_root = PROCESSED_ROOT / "archive" / f"baseline_fundamentals_purge_{stamp}"
    archive_root.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src in existing:
        if not src.exists():
            continue
        dst = archive_root / src.relative_to(PROJECT_ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1

    return archive_root, moved


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["settlementDate"])
    out["hour"] = ((out["settlementPeriod"].astype(int) - 1) // 2).astype("int8")
    out["dayOfWeek"] = dt.dt.dayofweek.astype("int8")
    out["isWeekend"] = out["dayOfWeek"].isin([5, 6]).astype("int8")
    out["month"] = dt.dt.month.astype("int8")
    out["quarter"] = dt.dt.quarter.astype("int8")
    out["year"] = dt.dt.year.astype("int16")
    radians = 2 * np.pi * ((out["settlementPeriod"].astype(int) - 1) / 48.0)
    out["settlementPeriod_sin"] = np.sin(radians)
    out["settlementPeriod_cos"] = np.cos(radians)
    return out


def basic_availability(
    variable: str,
    horizon: int,
    df: pd.DataFrame,
    value_cols: list[str],
    eligible: bool,
    notes: str,
) -> dict[str, object]:
    key_cols = ["settlementDate", "settlementPeriod"]
    duplicate_keys = int(df.loc[df.duplicated(key_cols, keep=False), key_cols].drop_duplicates().shape[0]) if not df.empty else 0
    missing = int(df[value_cols].isna().any(axis=1).sum()) if value_cols and not df.empty else EXPECTED_ROWS
    leakage = int((df["publishTime"] > df["asOfTime"]).sum()) if {"publishTime", "asOfTime"}.issubset(df.columns) else 0
    negative = int((df[value_cols] < 0).any(axis=1).sum()) if value_cols and not df.empty else 0
    return {
        "variable": variable,
        "horizonHours": horizon,
        "expectedRows": EXPECTED_ROWS,
        "selectedRows": int(len(df)),
        "missingRows": missing,
        "duplicateKeys": duplicate_keys,
        "publishAfterAsOfRows": leakage,
        "negativeValueRows": negative,
        "eligibleForForecastDataset": bool(eligible and leakage == 0 and duplicate_keys == 0),
        "notes": notes,
    }


def latest_asof(raw: pd.DataFrame, horizon: int, value_cols: list[str], extra_sort: list[str] | None = None) -> pd.DataFrame:
    df = raw.copy()
    df["asOfTime"] = df["targetStartTime"] - pd.to_timedelta(horizon, unit="h")
    df = df[df["publishTime"] <= df["asOfTime"]]
    sort_cols = ["settlementDate", "settlementPeriod", *(extra_sort or []), "publishTime"]
    return df.sort_values(sort_cols).drop_duplicates(["settlementDate", "settlementPeriod", *(extra_sort or [])], keep="last")


def build_system_target(spine: pd.DataFrame) -> pd.DataFrame:
    raw = read_raw_dir(
        "system_price_niv",
        ["settlementDate", "settlementPeriod", "startTime", "createdDateTime", "systemSellPrice", "systemBuyPrice", "netImbalanceVolume"],
    )
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["targetStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["createdDateTime"] = pd.to_datetime(raw["createdDateTime"], utc=True, errors="coerce")
    raw["systemSellPrice"] = pd.to_numeric(raw["systemSellPrice"], errors="coerce")
    raw["systemBuyPrice"] = pd.to_numeric(raw["systemBuyPrice"], errors="coerce")
    raw["systemPrice"] = np.where(np.isclose(raw["systemSellPrice"], raw["systemBuyPrice"], equal_nan=False), raw["systemBuyPrice"], np.nan)
    raw["netImbalanceVolume"] = pd.to_numeric(raw["netImbalanceVolume"], errors="coerce")
    target = spine.merge(
        raw[["settlementDate", "settlementPeriod", "systemPrice", "netImbalanceVolume", "createdDateTime"]],
        on=["settlementDate", "settlementPeriod"],
        how="left",
        validate="one_to_one",
    )
    target = target.merge(
        raw[["settlementDate", "settlementPeriod", "targetStartTime"]],
        on=["settlementDate", "settlementPeriod"],
        how="left",
        validate="one_to_one",
    )
    return target


def prepare_demand_raw(spine: pd.DataFrame) -> pd.DataFrame:
    raw = read_raw_dir(
        "day_ahead_demand_forecast_evolution",
        ["settlementDate", "settlementPeriod", "startTime", "publishTime", "nationalDemand"],
    )
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["nationalDemand"] = pd.to_numeric(raw["nationalDemand"], errors="coerce")
    raw = raw[raw["nationalDemand"].fillna(0) != 0]
    return raw.merge(spine[["settlementDate", "settlementPeriod", "targetStartTime"]], on=["settlementDate", "settlementPeriod"], how="inner")


def select_demand(raw: pd.DataFrame, spine: pd.DataFrame, horizon: int) -> pd.DataFrame:
    selected = latest_asof(raw, horizon, ["nationalDemand"])
    selected = selected.rename(columns={"nationalDemand": "demandForecast"})
    return spine.merge(
        selected[["settlementDate", "settlementPeriod", "demandForecast", "publishTime"]],
        on=["settlementDate", "settlementPeriod"],
        how="left",
        validate="one_to_one",
    ).assign(asOfTime=lambda d: d["targetStartTime"] - pd.to_timedelta(horizon, unit="h"))


def prepare_wind_solar_raw(spine: pd.DataFrame) -> pd.DataFrame:
    raw = read_raw_dir(
        "day_ahead_generation_wind_solar",
        ["settlementDate", "settlementPeriod", "startTime", "publishTime", "processType", "psrType", "quantity"],
    )
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")
    raw = raw[raw["psrType"].isin(["Wind Onshore", "Wind Offshore", "Solar"])]
    return raw.merge(spine[["settlementDate", "settlementPeriod", "targetStartTime"]], on=["settlementDate", "settlementPeriod"], how="inner")


def select_wind_solar(raw: pd.DataFrame, spine: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = latest_asof(raw, horizon, ["quantity"], extra_sort=["psrType"])
    pivot = selected.pivot_table(
        index=["settlementDate", "settlementPeriod"],
        columns="psrType",
        values="quantity",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "Wind Onshore": "windOnshoreForecast",
            "Wind Offshore": "windOffshoreForecast",
            "Solar": "solarForecast",
        }
    )
    for col in ["windOnshoreForecast", "windOffshoreForecast", "solarForecast"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["windForecast"] = pivot["windOnshoreForecast"] + pivot["windOffshoreForecast"]

    publish = selected.pivot_table(
        index=["settlementDate", "settlementPeriod"],
        columns="psrType",
        values="publishTime",
        aggfunc="first",
    ).reset_index()
    publish["publishTime"] = publish[[c for c in publish.columns if c not in ["settlementDate", "settlementPeriod"]]].max(axis=1)

    process_counts = (
        selected.groupby(["processType", "psrType"]).size().reset_index(name="rows").assign(horizonHours=horizon)
    )
    out = spine.merge(pivot[["settlementDate", "settlementPeriod", "windForecast", "solarForecast"]], on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
    out = out.merge(publish[["settlementDate", "settlementPeriod", "publishTime"]], on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
    out["asOfTime"] = out["targetStartTime"] - pd.to_timedelta(horizon, unit="h")
    return out, process_counts


def prepare_lolp_raw(spine: pd.DataFrame) -> pd.DataFrame:
    raw = read_raw_dir(
        "lolpdrm",
        ["settlementDate", "settlementPeriod", "startTime", "publishTime", "forecastHorizon", "lossOfLoadProbability", "deratedMargin"],
    )
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["forecastHorizon"] = pd.to_numeric(raw["forecastHorizon"], errors="coerce").astype("Int64")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["lossOfLoadProbability"] = pd.to_numeric(raw["lossOfLoadProbability"], errors="coerce")
    raw["deratedMargin"] = pd.to_numeric(raw["deratedMargin"], errors="coerce")
    return raw.merge(spine[["settlementDate", "settlementPeriod", "targetStartTime"]], on=["settlementDate", "settlementPeriod"], how="inner")


def select_lolp(raw: pd.DataFrame, spine: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = raw[raw["forecastHorizon"] == horizon]
    selected = latest_asof(df, horizon, ["deratedMargin", "lossOfLoadProbability"])
    out = spine.merge(
        selected[["settlementDate", "settlementPeriod", "deratedMargin", "lossOfLoadProbability", "publishTime"]],
        on=["settlementDate", "settlementPeriod"],
        how="left",
        validate="one_to_one",
    )
    out["asOfTime"] = out["targetStartTime"] - pd.to_timedelta(horizon, unit="h")
    return out


def audit_actual_only_baseline(spine: pd.DataFrame, target: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "variable": "netImbalanceVolume",
            "horizonHours": "all",
            "expectedRows": EXPECTED_ROWS,
            "selectedRows": int(target["netImbalanceVolume"].notna().sum()),
            "missingRows": int(target["netImbalanceVolume"].isna().sum()),
            "duplicateKeys": 0,
            "publishAfterAsOfRows": "not_applicable_actual_target_sp",
            "negativeValueRows": int((target["netImbalanceVolume"] < 0).sum()),
            "eligibleForForecastDataset": False,
            "notes": "Actual target-SP NIV; audited as baseline variable but excluded from strict ahead-of-time forecast datasets.",
        }
    )

    raw = read_raw_dir("interconnector_flows", ["settlementDate", "settlementPeriod", "publishTime", "interconnectorName", "generation"])
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["generation"] = pd.to_numeric(raw["generation"], errors="coerce")
    latest = raw.sort_values("publishTime").drop_duplicates(["settlementDate", "settlementPeriod", "interconnectorName"], keep="last")
    flow = latest.groupby(["settlementDate", "settlementPeriod"], as_index=False).agg(netInterconnectorFlow=("generation", "sum"))
    merged = spine.merge(flow, on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
    rows.append(
        {
            "variable": "netInterconnectorFlow",
            "horizonHours": "all",
            "expectedRows": EXPECTED_ROWS,
            "selectedRows": int(merged["netInterconnectorFlow"].notna().sum()),
            "missingRows": int(merged["netInterconnectorFlow"].isna().sum()),
            "duplicateKeys": 0,
            "publishAfterAsOfRows": "not_applicable_actual_outturn",
            "negativeValueRows": int((merged["netInterconnectorFlow"] < 0).sum()),
            "eligibleForForecastDataset": False,
            "notes": "Current source is interconnector outturn; excluded from strict ahead-of-time forecast datasets unless later converted to lagged flow.",
        }
    )
    return rows


def write_lineage() -> None:
    rows = [
        ["systemPrice", "target", "/balancing/settlement/system-prices/{date}", "data/raw/fundamentals/system_price_niv", "createdDateTime", "", "Target only"],
        ["netImbalanceVolume", "baseline_audit_only", "/balancing/settlement/system-prices/{date}", "data/raw/fundamentals/system_price_niv", "createdDateTime", "", "Actual target-SP NIV; excluded from strict forecast datasets"],
        ["deratedMargin/lossOfLoadProbability", "forecast_feature", "/forecast/system/loss-of-load", "data/raw/fundamentals/lolpdrm", "publishTime", "forecastHorizon", "Use matching horizons 1/2/4/8/12"],
        ["demandForecast", "forecast_feature", "/forecast/demand/day-ahead/evolution", "data/raw/fundamentals/day_ahead_demand_forecast_evolution", "publishTime", "", "Latest nonzero forecast available by as-of time"],
        ["windForecast/solarForecast", "forecast_feature", "/forecast/generation/wind-and-solar/day-ahead?processType=all", "data/raw/fundamentals/day_ahead_generation_wind_solar", "publishTime", "processType", "Latest forecast by as-of time across available process types"],
        ["netInterconnectorFlow", "baseline_audit_only", "/generation/outturn/interconnectors", "data/raw/fundamentals/interconnector_flows", "publishTime", "", "Outturn source; excluded unless converted to lagged flow"],
    ]
    pd.DataFrame(rows, columns=["variable", "role", "endpoint", "rawFolder", "publishTimestampField", "horizonOrProcessField", "selectionRule"]).to_csv(LINEAGE_CSV, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text_df = df.astype(str)
    cols = list(text_df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(row[col].replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIAG_DIR.mkdir(parents=True, exist_ok=True)
    archive_root, archived_count = archive_broad_outputs()

    spine = expected_spine()
    target = build_system_target(spine)
    target = add_calendar(target)
    time_spine = target[["settlementDate", "settlementPeriod", "targetStartTime"]].copy()

    write_lineage()
    demand_raw = prepare_demand_raw(time_spine)
    wind_solar_raw = prepare_wind_solar_raw(time_spine)
    lolp_raw = prepare_lolp_raw(time_spine)

    availability: list[dict[str, object]] = []
    leakage: list[dict[str, object]] = []
    process_counts: list[pd.DataFrame] = []
    built_datasets: list[dict[str, object]] = []

    availability.extend(audit_actual_only_baseline(spine, target))

    for horizon in HORIZONS:
        demand = select_demand(demand_raw, time_spine, horizon)
        wind_solar, process_count = select_wind_solar(wind_solar_raw, time_spine, horizon)
        lolp = select_lolp(lolp_raw, time_spine, horizon)
        process_counts.append(process_count)

        availability.append(basic_availability("demandForecast", horizon, demand, ["demandForecast"], True, "Latest demand forecast at or before as-of time."))
        availability.append(basic_availability("windForecast/solarForecast", horizon, wind_solar, ["windForecast", "solarForecast"], True, "Latest wind/solar forecast at or before as-of time."))
        availability.append(basic_availability("deratedMargin/lossOfLoadProbability", horizon, lolp, ["deratedMargin", "lossOfLoadProbability"], True, "Matching LOLP/DRM forecastHorizon selected."))

        model = target[
            [
                "settlementDate",
                "settlementPeriod",
                "targetStartTime",
                "systemPrice",
                "hour",
                "dayOfWeek",
                "isWeekend",
                "month",
                "quarter",
                "year",
                "settlementPeriod_sin",
                "settlementPeriod_cos",
            ]
        ].copy()
        model["forecastHorizonHours"] = horizon
        model["asOfTime"] = model["targetStartTime"] - pd.to_timedelta(horizon, unit="h")
        model = model.merge(demand[["settlementDate", "settlementPeriod", "demandForecast"]], on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
        model = model.merge(wind_solar[["settlementDate", "settlementPeriod", "windForecast", "solarForecast"]], on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
        model = model.merge(lolp[["settlementDate", "settlementPeriod", "deratedMargin", "lossOfLoadProbability"]], on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")

        required = ["systemPrice", "demandForecast", "windForecast", "solarForecast", "deratedMargin", "lossOfLoadProbability"]
        before = len(model)
        model = model.dropna(subset=required).reset_index(drop=True)
        csv_path = BASELINE_DIR / f"baseline_system_price_forecast_{horizon}h_2023_2025.csv"
        parquet_path = BASELINE_DIR / f"baseline_system_price_forecast_{horizon}h_2023_2025.parquet"
        model.to_csv(csv_path, index=False)
        model.to_parquet(parquet_path, index=False)
        built_datasets.append(
            {
                "horizonHours": horizon,
                "rowsBeforeCompleteCase": before,
                "rows": len(model),
                "droppedRows": before - len(model),
                "csv": str(csv_path.relative_to(PROJECT_ROOT)),
                "parquet": str(parquet_path.relative_to(PROJECT_ROOT)),
            }
        )

        for variable, df, value_cols in [
            ("demandForecast", demand, ["demandForecast"]),
            ("windForecast/solarForecast", wind_solar, ["windForecast", "solarForecast"]),
            ("deratedMargin/lossOfLoadProbability", lolp, ["deratedMargin", "lossOfLoadProbability"]),
        ]:
            leak_rows = df[df["publishTime"].notna() & (df["publishTime"] > df["asOfTime"])]
            leakage.append(
                {
                    "variable": variable,
                    "horizonHours": horizon,
                    "leakageRows": int(len(leak_rows)),
                    "eligibleForForecastDataset": int(len(leak_rows)) == 0,
                    "missingRows": int(df[value_cols].isna().any(axis=1).sum()),
                }
            )

    availability_df = pd.DataFrame(availability)
    leakage_df = pd.DataFrame(leakage)
    availability_df.to_csv(AVAILABILITY_CSV, index=False)
    leakage_df.to_csv(LEAKAGE_CSV, index=False)
    if RAW_READ_ISSUES:
        pd.DataFrame(RAW_READ_ISSUES).to_csv(BASELINE_DIAG_DIR / "baseline_raw_read_issues_2023_2025.csv", index=False)
    if process_counts:
        pd.concat(process_counts, ignore_index=True).to_csv(BASELINE_DIAG_DIR / "baseline_wind_solar_selected_process_type_counts_2023_2025.csv", index=False)
    pd.DataFrame(built_datasets).to_csv(BASELINE_DIAG_DIR / "baseline_forecast_dataset_build_audit_2023_2025.csv", index=False)

    summary = [
        "# Baseline Forecasting Fundamentals Quality Summary",
        "",
        f"Archive root: `{archive_root.relative_to(PROJECT_ROOT) if archive_root else 'none'}`",
        f"Broad artefacts archived on this run: `{archived_count}`",
        "",
        "## Model Dataset Builds",
        "",
        markdown_table(pd.DataFrame(built_datasets)),
        "",
        "## Forecast-Eligible Feature Audit",
        "",
        markdown_table(availability_df[availability_df["eligibleForForecastDataset"] == True]),  # noqa: E712
        "",
        "## Baseline Variables Excluded From Forecast Datasets",
        "",
        markdown_table(availability_df[availability_df["eligibleForForecastDataset"] == False]),  # noqa: E712
        "",
        "NIV and interconnector flow are audited but excluded because the available sources are target-SP actual/outturn values, not ex-ante forecasts.",
    ]
    SUMMARY_MD.write_text("\n".join(summary), encoding="utf-8")

    print(f"Archived broad outputs under {archive_root}; moved {archived_count} files")
    print(f"Wrote baseline diagnostics to {BASELINE_DIAG_DIR}")
    print(f"Wrote baseline forecast datasets to {BASELINE_DIR}")
    print(pd.DataFrame(built_datasets).to_string(index=False))


if __name__ == "__main__":
    main()
