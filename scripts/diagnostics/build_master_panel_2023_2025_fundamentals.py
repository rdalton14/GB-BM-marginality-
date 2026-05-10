from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUNDAMENTALS_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "fundamentals"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025"
DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "audits"

OUTPUT_CSV = PROCESSED_DIR / "master_panel_2023_2025_fundamentals.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "master_panel_2023_2025_fundamentals.parquet"
MISSINGNESS_CSV = DIAGNOSTICS_DIR / "master_panel_2023_2025_fundamentals_missingness.csv"
MERGE_LOG_CSV = DIAGNOSTICS_DIR / "master_panel_2023_2025_fundamentals_merge_log.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FUNDAMENTALS_DIR / name)


def merge_one(base: pd.DataFrame, incoming: pd.DataFrame, on: list[str], dataset: str, merge_log: list[dict]) -> pd.DataFrame:
    rows_before = len(base)
    merged = base.merge(incoming, on=on, how="left", validate="one_to_one")
    rows_after = len(merged)
    dup_after = int(merged.duplicated(on).sum())
    new_cols = [c for c in incoming.columns if c not in on]
    new_missing = int(merged[new_cols].isna().sum().sum())
    merge_log.append(
        {
            "dataset": dataset,
            "rowsBefore": rows_before,
            "rowsAfter": rows_after,
            "duplicatesAfter": dup_after,
            "newMissingValues": new_missing,
            "status": "ok" if rows_before == rows_after else "row_count_changed",
        }
    )
    if rows_before != rows_after:
        raise ValueError(f"Merge with {dataset} changed row count from {rows_before} to {rows_after}")
    return merged


def main() -> None:
    merge_log: list[dict] = []
    keys = ["settlementDate", "settlementPeriod"]

    spine = load_csv("system_price_niv_2023_2025.csv")[keys + ["systemBuyPrice", "systemSellPrice", "systemPrice", "netImbalanceVolume", "systemLongShort"]]
    spine["settlementPeriod"] = spine["settlementPeriod"].astype(int)

    initial_demand_outturn = load_csv("initial_demand_outturn_2023_2025.csv")[keys + ["InitialDemandOutturn"]].rename(columns={"InitialDemandOutturn": "demandOutturn"})
    demand_forecast = load_csv("day_ahead_national_demand_forecast_2023_2025.csv")[keys + ["dayAheadNationalDemandForecast"]]
    actual_wind_solar = load_csv("actual_generation_wind_solar_2023_2025.csv")[keys + ["windTotal", "windOnshore", "windOffshore", "solar"]].rename(
        columns={
            "windTotal": "windGeneration_actual",
            "windOnshore": "windOnshoreGeneration_actual",
            "windOffshore": "windOffshoreGeneration_actual",
            "solar": "solarGeneration_actual",
        }
    )
    forecast_wind_solar = load_csv("day_ahead_generation_wind_solar_2023_2025.csv")[keys + ["dayAheadWindTotalForecast", "dayAheadWindOnshoreForecast", "dayAheadWindOffshoreForecast", "dayAheadSolarForecast"]].rename(
        columns={
            "dayAheadWindTotalForecast": "windForecast",
            "dayAheadWindOnshoreForecast": "windOnshoreForecast",
            "dayAheadWindOffshoreForecast": "windOffshoreForecast",
            "dayAheadSolarForecast": "solarForecast",
        }
    )
    interconnector = load_csv("interconnector_flows_2023_2025.csv")[keys + ["netInterconnectorFlow"]]
    lolpdrm = load_csv("lolpdrm_2023_2025.csv")[keys + ["deratedMargin", "lossOfLoadProbability"]]
    apx = load_csv("apx_mid_2023_2025.csv")[keys + ["marketIndexPrice", "marketIndexVolume"]].rename(
        columns={"marketIndexPrice": "dayAheadPrice", "marketIndexVolume": "dayAheadVolume"}
    )

    panel = spine.copy()
    for name, df in [
        ("initial_demand_outturn", initial_demand_outturn),
        ("day_ahead_national_demand_forecast", demand_forecast),
        ("actual_generation_wind_solar", actual_wind_solar),
        ("day_ahead_generation_wind_solar", forecast_wind_solar),
        ("interconnector_flows", interconnector),
        ("lolpdrm", lolpdrm),
        ("apx_mid", apx),
    ]:
        panel = merge_one(panel, df, keys, name, merge_log)

    panel = panel.sort_values(keys).reset_index(drop=True)
    panel["settlementDate"] = pd.to_datetime(panel["settlementDate"])
    panel["lag_systemPrice"] = panel["systemPrice"].shift(1)
    panel["demandForecastError"] = panel["dayAheadNationalDemandForecast"] - panel["demandOutturn"]
    panel["windForecastError"] = panel["windForecast"] - panel["windGeneration_actual"]
    panel["solarForecastError"] = panel["solarForecast"] - panel["solarGeneration_actual"]
    panel["lolp_event"] = panel["lossOfLoadProbability"].gt(0).where(panel["lossOfLoadProbability"].notna(), pd.NA)
    panel["hour"] = ((panel["settlementPeriod"] - 1) // 2).astype(int)
    panel["dayOfWeek"] = panel["settlementDate"].dt.dayofweek.add(1).mod(7)
    panel["isWeekend"] = panel["dayOfWeek"].isin([0, 6]).astype(int)
    panel["month"] = panel["settlementDate"].dt.month
    panel["quarter"] = panel["settlementDate"].dt.quarter
    panel["year"] = panel["settlementDate"].dt.year
    radians = 2 * 3.141592653589793 * ((panel["settlementPeriod"] - 1) / 48.0)
    panel["settlementPeriod_sin"] = pd.Series(radians).map(__import__("math").sin)
    panel["settlementPeriod_cos"] = pd.Series(radians).map(__import__("math").cos)

    ordered_cols = [
        "settlementDate",
        "settlementPeriod",
        "systemBuyPrice",
        "systemSellPrice",
        "systemPrice",
        "netImbalanceVolume",
        "systemLongShort",
        "lag_systemPrice",
        "demandOutturn",
        "dayAheadNationalDemandForecast",
        "demandForecastError",
        "windGeneration_actual",
        "windOnshoreGeneration_actual",
        "windOffshoreGeneration_actual",
        "solarGeneration_actual",
        "windForecast",
        "windOnshoreForecast",
        "windOffshoreForecast",
        "solarForecast",
        "windForecastError",
        "solarForecastError",
        "netInterconnectorFlow",
        "deratedMargin",
        "lossOfLoadProbability",
        "lolp_event",
        "dayAheadPrice",
        "dayAheadVolume",
        "hour",
        "dayOfWeek",
        "isWeekend",
        "month",
        "quarter",
        "year",
        "settlementPeriod_sin",
        "settlementPeriod_cos",
    ]
    panel = panel[ordered_cols]
    panel["settlementDate"] = panel["settlementDate"].dt.strftime("%Y-%m-%d")

    panel.to_csv(OUTPUT_CSV, index=False)
    panel.to_parquet(OUTPUT_PARQUET, index=False)
    missingness = panel.isna().sum().rename("missingValues").reset_index()
    missingness.columns = ["column", "missingValues"]
    missingness.to_csv(MISSINGNESS_CSV, index=False)
    pd.DataFrame(merge_log).to_csv(MERGE_LOG_CSV, index=False)

    print("Built clean 2023-2025 fundamentals panel")
    print(f"shape = {panel.shape}")
    print(f"duplicate SP keys = {int(panel.duplicated(keys).sum())}")
    print(f"saved csv = {OUTPUT_CSV}")
    print(f"saved parquet = {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
