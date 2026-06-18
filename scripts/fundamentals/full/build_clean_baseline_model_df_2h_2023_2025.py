from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "fundamentals"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "baseline_2h_model"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "baseline_2h_model"

START = "2023-01-01"
END = "2025-12-31"
FORECAST_HORIZON = 2
LAG_SETTLEMENT_PERIODS = 4
PSR_TYPES = ["Wind Onshore", "Wind Offshore", "Solar"]


def files_for_window(raw_name: str, allow_empty: bool = True) -> list[Path]:
    files: list[Path] = []
    day = pd.Timestamp(START)
    end = pd.Timestamp(END)
    raw_dir = RAW_ROOT / raw_name
    while day <= end:
        path = raw_dir / f"{day.strftime('%Y-%m-%d')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size > 0 or allow_empty:
            files.append(path)
        day += pd.Timedelta(days=1)
    return files


def read_daily_csv(raw_name: str, usecols: list[str], allow_empty: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    issues: list[dict[str, object]] = []
    for path in files_for_window(raw_name, allow_empty=allow_empty):
        if path.stat().st_size == 0:
            issues.append({"rawName": raw_name, "date": path.stem, "issue": "empty_file", "path": str(path)})
            continue
        try:
            frames.append(pd.read_csv(path, usecols=usecols))
        except ValueError as exc:
            issues.append({"rawName": raw_name, "date": path.stem, "issue": f"read_error: {exc}", "path": str(path)})
    if not frames:
        return pd.DataFrame(columns=usecols), pd.DataFrame(issues)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(issues)


def build_calendar_spine() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    day = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while day <= end:
        for sp in range(1, 49):
            records.append(
                {
                    "settlementDate": day.strftime("%Y-%m-%d"),
                    "settlementPeriod": sp,
                    "targetStartTime": pd.Timestamp(day, tz="UTC") + pd.Timedelta(minutes=30 * (sp - 1)),
                }
            )
        day += pd.Timedelta(days=1)
    spine = pd.DataFrame(records)
    spine["settlementPeriod"] = spine["settlementPeriod"].astype("int16")
    return spine


def build_lolp_drm_2h() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, issues = read_daily_csv(
        "lolpdrm",
        [
            "publishTime",
            "publishingPeriodCommencingTime",
            "startTime",
            "settlementDate",
            "settlementPeriod",
            "forecastHorizon",
            "lossOfLoadProbability",
            "deratedMargin",
        ],
    )
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="coerce").astype("Int16")
    raw["forecastHorizon"] = pd.to_numeric(raw["forecastHorizon"], errors="coerce").astype("Int16")
    raw = raw[raw["forecastHorizon"].eq(FORECAST_HORIZON)].copy()
    raw["lolpDrmPublishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["lossOfLoadProbability"] = pd.to_numeric(raw["lossOfLoadProbability"], errors="coerce")
    raw["deratedMargin"] = pd.to_numeric(raw["deratedMargin"], errors="coerce")
    keys = ["settlementDate", "settlementPeriod"]
    raw = raw.sort_values(keys + ["lolpDrmPublishTime"]).drop_duplicates(keys, keep="last")
    out = raw[keys + ["lossOfLoadProbability", "deratedMargin"]].copy()
    out["settlementPeriod"] = out["settlementPeriod"].astype("int16")
    return out.sort_values(keys).reset_index(drop=True), issues


def build_wind_solar() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, issues = read_daily_csv(
        "day_ahead_generation_wind_solar",
        ["publishTime", "processType", "businessType", "psrType", "startTime", "settlementDate", "settlementPeriod", "quantity"],
    )
    raw = raw[raw["psrType"].isin(PSR_TYPES)].copy()
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="coerce").astype("Int16")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")
    keys = ["settlementDate", "settlementPeriod"]
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
    out = values.merge(publish, on=keys, how="left", validate="one_to_one")
    out["settlementPeriod"] = out["settlementPeriod"].astype("int16")
    out["windSolarPublishTime"] = out["windOnshoreForecast_publishTime"]
    return out.sort_values(keys).reset_index(drop=True), issues


def build_demand_aligned(spine: pd.DataFrame, wind_solar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, issues = read_daily_csv(
        "day_ahead_demand_forecast_evolution",
        [
            "startTime",
            "settlementDate",
            "settlementPeriod",
            "boundary",
            "publishTime",
            "transmissionSystemDemand",
            "nationalDemand",
        ],
    )
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="coerce").astype("Int16")
    raw["demandForecast_publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["demandForecast"] = pd.to_numeric(raw["nationalDemand"], errors="coerce")
    raw["transmissionDemandForecast"] = pd.to_numeric(raw["transmissionSystemDemand"], errors="coerce")
    raw = raw[raw["boundary"].eq("N") & raw["demandForecast"].fillna(0).ne(0)].copy()

    keys = ["settlementDate", "settlementPeriod"]
    cutoffs = spine[keys].merge(
        wind_solar[keys + ["windSolarPublishTime"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    candidates = cutoffs.merge(
        raw[keys + ["demandForecast", "transmissionDemandForecast", "demandForecast_publishTime"]],
        on=keys,
        how="left",
        validate="one_to_many",
    )
    candidates = candidates[candidates["demandForecast_publishTime"] <= candidates["windSolarPublishTime"]]
    selected = (
        candidates.sort_values(keys + ["demandForecast_publishTime"])
        .drop_duplicates(keys, keep="last")
    )
    out = spine[keys].merge(
        selected[keys + ["demandForecast", "transmissionDemandForecast", "demandForecast_publishTime"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    out["settlementPeriod"] = out["settlementPeriod"].astype("int16")
    return out, issues


def build_system_price_and_niv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, issues = read_daily_csv(
        "system_price_niv",
        ["settlementDate", "settlementPeriod", "startTime", "systemSellPrice", "systemBuyPrice", "netImbalanceVolume"],
    )
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="coerce").astype("Int16")
    raw["systemSellPrice"] = pd.to_numeric(raw["systemSellPrice"], errors="coerce")
    raw["systemBuyPrice"] = pd.to_numeric(raw["systemBuyPrice"], errors="coerce")
    raw["systemPrice"] = raw["systemBuyPrice"].where(raw["systemBuyPrice"].eq(raw["systemSellPrice"]))
    raw["nivLag4"] = pd.to_numeric(raw["netImbalanceVolume"], errors="coerce")
    keys = ["settlementDate", "settlementPeriod"]
    system_price = raw[keys + ["systemPrice"]].drop_duplicates(keys, keep="last").copy()
    niv = raw[keys + ["nivLag4"]].rename(
        columns={"settlementDate": "lag4SourceSettlementDate", "settlementPeriod": "lag4SourceSP"}
    )
    niv = niv.drop_duplicates(["lag4SourceSettlementDate", "lag4SourceSP"], keep="last").copy()
    return system_price, niv, issues


def build_interconnector_flow() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, issues = read_daily_csv(
        "interconnector_flows",
        ["publishTime", "startTime", "settlementDate", "settlementPeriod", "interconnectorName", "generation"],
    )
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="coerce").astype("Int16")
    raw["generation"] = pd.to_numeric(raw["generation"], errors="coerce")
    out = (
        raw.groupby(["settlementDate", "settlementPeriod"], as_index=False)
        .agg(
            interconnectorFlowLag4=("generation", "sum"),
            interconnectorLag4ComponentCount=("interconnectorName", "nunique"),
        )
        .rename(columns={"settlementDate": "lag4SourceSettlementDate", "settlementPeriod": "lag4SourceSP"})
        .sort_values(["lag4SourceSettlementDate", "lag4SourceSP"])
    )
    out["lag4SourceSP"] = out["lag4SourceSP"].astype("int16")
    return out, issues


def build_quality(df: pd.DataFrame, raw_issues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "systemPrice",
        "solarForecast",
        "windForecast",
        "demandForecast",
        "lossOfLoadProbability",
        "deratedMargin",
        "nivLag4",
        "interconnectorFlowLag4",
    ]
    expected_calendar_rows = int(pd.date_range(START, END, freq="D").size * 48)
    checks = [
        ("expected_calendar_sp_rows_48_per_day", expected_calendar_rows),
        ("rows", len(df)),
        ("missing_spine_rows_vs_48_per_day", expected_calendar_rows - len(df)),
        ("initial_lag_warmup_rows", LAG_SETTLEMENT_PERIODS),
        ("complete_rows_all_model_columns", int(df[feature_cols].notna().all(axis=1).sum())),
        ("complete_share_all_model_columns", float(df[feature_cols].notna().all(axis=1).mean())),
        ("lag_settlement_periods", LAG_SETTLEMENT_PERIODS),
        ("lag_hours_from_target_start_time", LAG_SETTLEMENT_PERIODS * 0.5),
    ]
    checks.extend((f"missing_{col}_rows", int(df[col].isna().sum())) for col in feature_cols)
    quality = pd.DataFrame(checks, columns=["check", "value"])

    missing_by_date = (
        df.assign(anyMissing=df[feature_cols].isna().any(axis=1))
        .groupby("settlementDate", as_index=False)
        .agg(rows=("SP", "size"), missingRows=("anyMissing", "sum"))
    )
    missing_by_date = missing_by_date[missing_by_date["missingRows"].gt(0)]
    if not raw_issues.empty:
        raw_issues.to_csv(DIAG_DIR / "clean_baseline_model_df_2h_raw_read_issues_2023_2025.csv", index=False)
    return quality, missing_by_date


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print("Building complete settlement-period calendar spine ...")
    spine = build_calendar_spine()
    print(f"  spine rows: {len(spine):,}")

    print("Building LOLP/DRM 2h block ...")
    lolp_drm, issues_lolp = build_lolp_drm_2h()
    print(f"  LOLP/DRM rows: {len(lolp_drm):,}")

    print("Building wind/solar day-ahead block ...")
    wind_solar, issues_ws = build_wind_solar()
    print(f"  wind/solar rows: {len(wind_solar):,}")

    print("Aligning demand to wind/solar publish cutoff ...")
    demand, issues_demand = build_demand_aligned(spine, wind_solar)

    print("Building system price and NIV source ...")
    system_price, niv, issues_system = build_system_price_and_niv()

    print("Building interconnector lag source ...")
    interconnector, issues_interconnector = build_interconnector_flow()

    keys = ["settlementDate", "settlementPeriod"]
    df = spine.merge(lolp_drm, on=keys, how="left", validate="one_to_one")
    df = df.merge(
        wind_solar[keys + ["solarForecast", "windForecast", "windSolarPublishTime"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    df = df.merge(demand[keys + ["demandForecast"]], on=keys, how="left", validate="one_to_one")
    df = df.merge(system_price, on=keys, how="left", validate="one_to_one")
    df["lag4SourceSP"] = df["settlementPeriod"] - LAG_SETTLEMENT_PERIODS
    needs_previous_day = df["lag4SourceSP"].le(0)
    df["lag4SourceSettlementDate"] = df["settlementDate"]
    df.loc[needs_previous_day, "lag4SourceSettlementDate"] = (
        pd.to_datetime(df.loc[needs_previous_day, "settlementDate"]) - pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    df.loc[needs_previous_day, "lag4SourceSP"] = df.loc[needs_previous_day, "lag4SourceSP"] + 48
    df["lag4SourceSP"] = df["lag4SourceSP"].astype("int16")

    df = df.merge(niv, on=["lag4SourceSettlementDate", "lag4SourceSP"], how="left", validate="many_to_one")
    df = df.merge(
        interconnector,
        on=["lag4SourceSettlementDate", "lag4SourceSP"],
        how="left",
        validate="many_to_one",
    )
    df = df.rename(columns={"settlementPeriod": "SP"})

    clean_cols = [
        "settlementDate",
        "SP",
        "targetStartTime",
        "solarForecast",
        "windForecast",
        "demandForecast",
        "lossOfLoadProbability",
        "deratedMargin",
        "nivLag4",
        "interconnectorFlowLag4",
        "systemPrice",
    ]
    clean = df[clean_cols].sort_values(["settlementDate", "SP"]).reset_index(drop=True)
    audit_no_warmup = clean.iloc[LAG_SETTLEMENT_PERIODS:].reset_index(drop=True)
    complete = clean.dropna(
        subset=[
            "systemPrice",
            "solarForecast",
            "windForecast",
            "demandForecast",
            "lossOfLoadProbability",
            "deratedMargin",
            "nivLag4",
            "interconnectorFlowLag4",
        ]
    ).reset_index(drop=True)
    complete_no_warmup = audit_no_warmup.dropna(
        subset=[
            "systemPrice",
            "solarForecast",
            "windForecast",
            "demandForecast",
            "lossOfLoadProbability",
            "deratedMargin",
            "nivLag4",
            "interconnectorFlowLag4",
        ]
    ).reset_index(drop=True)

    raw_issues = pd.concat([issues_lolp, issues_ws, issues_demand, issues_system, issues_interconnector], ignore_index=True)
    quality, missing_by_date = build_quality(clean, raw_issues)
    no_warmup_quality, no_warmup_missing_by_date = build_quality(audit_no_warmup, raw_issues)

    out_path = OUT_DIR / "clean_baseline_model_df_2h_2023_2025"
    no_warmup_path = OUT_DIR / "clean_baseline_model_df_2h_no_lag_warmup_2023_2025"
    complete_path = OUT_DIR / "baseline_2h_model_complete_cases_2023_2025"
    complete_no_warmup_path = OUT_DIR / "baseline_2h_model_complete_cases_no_lag_warmup_2023_2025"
    clean.to_csv(out_path.with_suffix(".csv"), index=False)
    clean.to_parquet(out_path.with_suffix(".parquet"), index=False)
    audit_no_warmup.to_csv(no_warmup_path.with_suffix(".csv"), index=False)
    audit_no_warmup.to_parquet(no_warmup_path.with_suffix(".parquet"), index=False)
    complete.to_csv(complete_path.with_suffix(".csv"), index=False)
    complete.to_parquet(complete_path.with_suffix(".parquet"), index=False)
    complete_no_warmup.to_csv(complete_no_warmup_path.with_suffix(".csv"), index=False)
    complete_no_warmup.to_parquet(complete_no_warmup_path.with_suffix(".parquet"), index=False)
    quality.to_csv(DIAG_DIR / "clean_baseline_model_df_2h_quality_2023_2025.csv", index=False)
    missing_by_date.to_csv(DIAG_DIR / "clean_baseline_model_df_2h_missing_by_date_2023_2025.csv", index=False)
    no_warmup_quality.to_csv(DIAG_DIR / "clean_baseline_model_df_2h_no_lag_warmup_quality_2023_2025.csv", index=False)
    no_warmup_missing_by_date.to_csv(
        DIAG_DIR / "clean_baseline_model_df_2h_no_lag_warmup_missing_by_date_2023_2025.csv",
        index=False,
    )

    print(f"\nWrote audit spine DF to {out_path.with_suffix('.csv')}")
    print(f"Wrote no-warmup audit DF to {no_warmup_path.with_suffix('.csv')}")
    print(f"Wrote complete-case model DF to {complete_path.with_suffix('.csv')}")
    print(f"Wrote no-warmup complete-case model DF to {complete_no_warmup_path.with_suffix('.csv')}")
    print(quality.to_string(index=False))
    print(f"Dates with any missing rows: {len(missing_by_date):,}")


if __name__ == "__main__":
    main()
