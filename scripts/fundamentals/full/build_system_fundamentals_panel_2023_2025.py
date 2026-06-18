from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FUNDAMENTALS_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "fundamentals"
RAW_FUNDAMENTALS_DIR = PROJECT_ROOT / "data" / "raw" / "fundamentals"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025"
DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "audits"
START = date(2023, 1, 1)
END = date(2025, 12, 31)


OUTPUT_CSV = PROCESSED_DIR / "system_fundamentals_2023_2025.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "system_fundamentals_2023_2025.parquet"
MODEL_OUTPUT_CSV = PROCESSED_DIR / "system_fundamentals_model_2023_2025.csv"
MODEL_OUTPUT_PARQUET = PROCESSED_DIR / "system_fundamentals_model_2023_2025.parquet"
MERGE_AUDIT_CSV = DIAGNOSTICS_DIR / "system_fundamentals_2023_2025_merge_audit.csv"
MISSINGNESS_CSV = DIAGNOSTICS_DIR / "system_fundamentals_2023_2025_missingness.csv"
INCONSISTENCIES_CSV = DIAGNOSTICS_DIR / "system_fundamentals_2023_2025_inconsistencies.csv"
MODEL_AUDIT_CSV = DIAGNOSTICS_DIR / "system_fundamentals_model_2023_2025_audit.csv"


def expected_spine() -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D").date
    return pd.MultiIndex.from_product(
        [[d.isoformat() for d in days], range(1, 49)],
        names=["settlementDate", "settlementPeriod"],
    ).to_frame(index=False)


def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=columns)


def audit_keys(df: pd.DataFrame, dataset: str) -> dict[str, object]:
    key_cols = ["settlementDate", "settlementPeriod"]
    duplicate_rows = int(df.duplicated(key_cols).sum())
    duplicate_keys = int(df.loc[df.duplicated(key_cols, keep=False), key_cols].drop_duplicates().shape[0])
    return {
        "dataset": dataset,
        "rows": int(len(df)),
        "uniqueKeys": int(df[key_cols].drop_duplicates().shape[0]) if not df.empty else 0,
        "duplicateRows": duplicate_rows,
        "duplicateKeys": duplicate_keys,
        "minDate": df["settlementDate"].min() if not df.empty else "",
        "maxDate": df["settlementDate"].max() if not df.empty else "",
    }


def add_issue(issues: list[dict[str, object]], severity: str, dataset: str, issue: str, detail: str) -> None:
    issues.append({"severity": severity, "dataset": dataset, "issue": issue, "detail": detail})


def reduce_day_ahead_demand_forecast(issues: list[dict[str, object]]) -> pd.DataFrame:
    raw_dir = RAW_FUNDAMENTALS_DIR / "day_ahead_demand_forecast_evolution"
    files = sorted(raw_dir.glob("*.csv"))
    if len(files) != (END - START).days + 1:
        add_issue(
            issues,
            "error",
            "day_ahead_national_demand_forecast",
            "raw_file_count",
            f"Expected 1096 daily raw files, found {len(files)}.",
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        day = pd.read_csv(
            path,
            usecols=["startTime", "settlementDate", "settlementPeriod", "publishTime", "nationalDemand"],
        )
        frames.append(day)

    raw = pd.concat(frames, ignore_index=True)
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["nationalDemand"] = pd.to_numeric(raw["nationalDemand"], errors="coerce")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["deliveryDate"] = pd.to_datetime(raw["settlementDate"], errors="coerce").dt.date
    raw["dayAheadDate"] = pd.to_datetime(raw["deliveryDate"].astype(str)) - pd.Timedelta(days=1)
    raw["publishDate"] = raw["publishTime"].dt.floor("D").dt.tz_localize(None)

    candidates = raw[(raw["publishDate"] == raw["dayAheadDate"]) & (raw["nationalDemand"].fillna(0) != 0)].copy()
    selected = (
        candidates.sort_values(["settlementDate", "settlementPeriod", "publishTime"])
        .drop_duplicates(["settlementDate", "settlementPeriod"], keep="last")
        .rename(columns={"nationalDemand": "dayAheadNationalDemandForecast"})
    )
    out = selected[
        [
            "settlementDate",
            "settlementPeriod",
            "dayAheadNationalDemandForecast",
            "publishTime",
            "startTime",
        ]
    ].rename(columns={"publishTime": "dayAheadNationalDemandForecast_publishTime"})

    out.to_csv(FUNDAMENTALS_DIR / "day_ahead_national_demand_forecast_2023_2025.csv", index=False)
    return out


def load_optional_processed(name: str, filename: str, issues: list[dict[str, object]]) -> pd.DataFrame | None:
    path = FUNDAMENTALS_DIR / filename
    if not path.exists():
        add_issue(issues, "warning", name, "missing_processed_file", f"{path} was not present.")
        return None
    return pd.read_csv(path)


def main() -> None:
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    issues: list[dict[str, object]] = []
    merge_audit: list[dict[str, object]] = []
    spine = expected_spine()

    system = read_csv(FUNDAMENTALS_DIR / "system_price_niv_2023_2025.csv")
    merge_audit.append(audit_keys(system, "system_price_niv"))
    if system.duplicated(["settlementDate", "settlementPeriod"]).any():
        add_issue(issues, "error", "system_price_niv", "duplicate_keys", "System price spine contains duplicate SP keys.")

    panel = spine.merge(system, on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")

    datasets: list[tuple[str, pd.DataFrame, list[str]]] = []
    actual_load = load_optional_processed("actual_total_load", "actual_total_load_2023_2025.csv", issues)
    if actual_load is not None:
        actual_load = actual_load.rename(columns={"ActualTotalLoad": "demandOutturn"})
        datasets.append(("actual_total_load", actual_load[["settlementDate", "settlementPeriod", "demandOutturn"]], ["demandOutturn"]))

    initial_demand = load_optional_processed("initial_demand_outturn", "initial_demand_outturn_2023_2025.csv", issues)
    if initial_demand is not None:
        initial_demand = initial_demand.rename(columns={"InitialDemandOutturn": "initialDemandOutturn"})
        datasets.append(
            (
                "initial_demand_outturn",
                initial_demand[["settlementDate", "settlementPeriod", "initialDemandOutturn"]],
                ["initialDemandOutturn"],
            )
        )

    demand_forecast = reduce_day_ahead_demand_forecast(issues)
    datasets.append(
        (
            "day_ahead_national_demand_forecast",
            demand_forecast[["settlementDate", "settlementPeriod", "dayAheadNationalDemandForecast"]],
            ["dayAheadNationalDemandForecast"],
        )
    )

    optional_specs = [
        (
            "actual_generation_wind_solar",
            "actual_generation_wind_solar_2023_2025.csv",
            {
                "windTotal": "windGeneration_actual",
                "windOnshore": "windOnshoreGeneration_actual",
                "windOffshore": "windOffshoreGeneration_actual",
                "solar": "solarGeneration_actual",
            },
        ),
        (
            "day_ahead_generation_wind_solar",
            "day_ahead_generation_wind_solar_2023_2025.csv",
            {
                "dayAheadWindTotalForecast": "windForecast",
                "dayAheadWindOnshoreForecast": "windOnshoreForecast",
                "dayAheadWindOffshoreForecast": "windOffshoreForecast",
                "dayAheadSolarForecast": "solarForecast",
            },
        ),
        ("interconnector_flows", "interconnector_flows_2023_2025.csv", {"netInterconnectorFlow": "netInterconnectorFlow"}),
        (
            "lolpdrm",
            "lolpdrm_2023_2025.csv",
            {"deratedMargin": "deratedMargin", "lossOfLoadProbability": "lossOfLoadProbability"},
        ),
        ("apx_mid", "apx_mid_2023_2025.csv", {"marketIndexPrice": "dayAheadPrice", "marketIndexVolume": "dayAheadVolume"}),
    ]
    for name, filename, rename_map in optional_specs:
        df = load_optional_processed(name, filename, issues)
        if df is None:
            continue
        df = df.rename(columns=rename_map)
        cols = ["settlementDate", "settlementPeriod", *rename_map.values()]
        datasets.append((name, df[cols], list(rename_map.values())))

    for name, df, value_cols in datasets:
        merge_audit.append(audit_keys(df, name))
        dup_keys = int(df.loc[df.duplicated(["settlementDate", "settlementPeriod"], keep=False), ["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0])
        if dup_keys:
            add_issue(issues, "error", name, "duplicate_keys", f"{dup_keys} duplicate SP keys before merge.")
        before = len(panel)
        panel = panel.merge(df, on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
        if len(panel) != before:
            add_issue(issues, "error", name, "row_count_changed", f"Rows changed from {before} to {len(panel)}.")
        for col in value_cols:
            missing = int(panel[col].isna().sum())
            if missing:
                add_issue(issues, "warning", name, f"{col}_missing", f"{missing} missing SP values after merge.")

    panel = panel.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    panel["systemPrice"] = np.where(
        np.isclose(panel["systemBuyPrice"], panel["systemSellPrice"], equal_nan=False),
        panel["systemBuyPrice"],
        np.nan,
    )
    derived_side = np.select(
        [panel["netImbalanceVolume"] > 0, panel["netImbalanceVolume"] < 0],
        ["short", "long"],
        default="balanced",
    )
    side_check_mask = panel["systemLongShort"].notna() & panel["netImbalanceVolume"].notna()
    mismatched_side = int(
        (panel.loc[side_check_mask, "systemLongShort"] != pd.Series(derived_side, index=panel.index).loc[side_check_mask]).sum()
    )
    if mismatched_side:
        add_issue(issues, "error", "system_price_niv", "systemLongShort_mismatch", f"{mismatched_side} rows mismatch NIV sign.")

    if {"dayAheadNationalDemandForecast", "demandOutturn"}.issubset(panel.columns):
        panel["demandForecastError"] = panel["dayAheadNationalDemandForecast"] - panel["demandOutturn"]
    if {"windForecast", "windGeneration_actual"}.issubset(panel.columns):
        panel["windForecastError"] = panel["windForecast"] - panel["windGeneration_actual"]
    if {"solarForecast", "solarGeneration_actual"}.issubset(panel.columns):
        panel["solarForecastError"] = panel["solarForecast"] - panel["solarGeneration_actual"]
    if "lossOfLoadProbability" in panel.columns:
        panel["lolp_event"] = (panel["lossOfLoadProbability"] > 0).astype("Int8")

    panel["settlementDate_dt"] = pd.to_datetime(panel["settlementDate"])
    panel["hour"] = ((panel["settlementPeriod"] - 1) // 2).astype("int8")
    panel["dayOfWeek"] = panel["settlementDate_dt"].dt.dayofweek.astype("int8")
    panel["isWeekend"] = panel["dayOfWeek"].isin([5, 6]).astype("int8")
    panel["month"] = panel["settlementDate_dt"].dt.month.astype("int8")
    panel["quarter"] = panel["settlementDate_dt"].dt.quarter.astype("int8")
    panel["year"] = panel["settlementDate_dt"].dt.year.astype("int16")
    radians = 2 * np.pi * ((panel["settlementPeriod"] - 1) / 48.0)
    panel["settlementPeriod_sin"] = np.sin(radians)
    panel["settlementPeriod_cos"] = np.cos(radians)
    panel = panel.drop(columns=["settlementDate_dt"])

    missingness = (
        panel.isna()
        .sum()
        .rename("missingValues")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missingness["missingShare"] = missingness["missingValues"] / len(panel)

    pd.DataFrame(merge_audit).to_csv(MERGE_AUDIT_CSV, index=False)
    missingness.to_csv(MISSINGNESS_CSV, index=False)
    pd.DataFrame(issues).to_csv(INCONSISTENCIES_CSV, index=False)
    panel.to_csv(OUTPUT_CSV, index=False)
    panel.to_parquet(OUTPUT_PARQUET, index=False)

    modelling_drop_cols = [
        "systemBuyPrice",
        "systemSellPrice",
        "demandOutturn",
        "demandForecastError",
        "dayAheadVolume",
    ]
    model_panel = panel.drop(columns=[c for c in modelling_drop_cols if c in panel.columns])
    model_panel_complete = model_panel.dropna().reset_index(drop=True)
    model_audit = pd.DataFrame(
        [
            {
                "inputRows": len(panel),
                "inputColumns": panel.shape[1],
                "modelRowsBeforeCompleteCase": len(model_panel),
                "modelColumns": model_panel.shape[1],
                "droppedRowsForMissingModelValues": len(model_panel) - len(model_panel_complete),
                "modelRows": len(model_panel_complete),
                "droppedColumns": ",".join([c for c in modelling_drop_cols if c in panel.columns]),
            }
        ]
    )
    model_panel_complete.to_csv(MODEL_OUTPUT_CSV, index=False)
    model_panel_complete.to_parquet(MODEL_OUTPUT_PARQUET, index=False)
    model_audit.to_csv(MODEL_AUDIT_CSV, index=False)

    print(f"wrote {OUTPUT_CSV} rows={len(panel)} cols={panel.shape[1]}")
    print(f"wrote {OUTPUT_PARQUET}")
    print(f"wrote {MODEL_OUTPUT_CSV} rows={len(model_panel_complete)} cols={model_panel_complete.shape[1]}")
    print(f"wrote {MODEL_OUTPUT_PARQUET}")
    print(f"issues={len(issues)} audit={MERGE_AUDIT_CSV}")


if __name__ == "__main__":
    main()
