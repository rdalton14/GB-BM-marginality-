from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_fundamentals.parquet"
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_fundamentals.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_fundamentals_imputed.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_fundamentals_imputed.parquet"
IMPUTATION_LOG = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "master_panel_2023_2025_fundamentals_imputation_log.csv"


def load_panel() -> pd.DataFrame:
    if INPUT_PARQUET.exists():
        df = pd.read_parquet(INPUT_PARQUET)
    else:
        df = pd.read_csv(INPUT_CSV)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    return df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)


def impute_linear(df: pd.DataFrame, col: str, flag_col: str) -> None:
    df[flag_col] = df[col].isna().astype("int8")
    df[col] = df[col].interpolate(method="linear", limit_direction="both")


def impute_by_settlement_period(df: pd.DataFrame, col: str, flag_col: str) -> None:
    df[flag_col] = df[col].isna().astype("int8")
    df[col] = (
        df.groupby("settlementPeriod", group_keys=False)[col]
        .apply(lambda s: s.interpolate(method="linear", limit_direction="both"))
        .reset_index(level=0, drop=True)
    )


def main() -> None:
    df = load_panel()
    original_missing = df.isna().sum()

    impute_linear(df, "demandOutturn", "imputed_demandOutturn")
    for col in ["windForecast", "windOnshoreForecast", "windOffshoreForecast", "solarForecast"]:
        impute_by_settlement_period(df, col, f"imputed_{col}")
    for col in ["deratedMargin", "lossOfLoadProbability"]:
        impute_linear(df, col, f"imputed_{col}")

    df["lolp_event"] = (df["lossOfLoadProbability"] > 0).astype("int8")
    df["demandForecastError"] = df["dayAheadNationalDemandForecast"] - df["demandOutturn"]
    df["windForecastError"] = df["windForecast"] - df["windGeneration_actual"]
    df["solarForecastError"] = df["solarForecast"] - df["solarGeneration_actual"]
    df["lag_systemPrice"] = df["systemPrice"].shift(1)
    df["lag_systemPrice_boundary"] = df["lag_systemPrice"].isna().astype("int8")

    imputation_flag_cols = [col for col in df.columns if col.startswith("imputed_")]
    df["any_imputed"] = df[imputation_flag_cols].max(axis=1).astype("int8")
    final_missing = df.isna().sum()

    log = pd.DataFrame(
        [
            {
                "column": col,
                "originalMissing": int(original_missing.get(col, 0)),
                "postImputationMissing": int(final_missing.get(col, 0)),
                "imputedRows": int(df[f"imputed_{col}"].sum()) if f"imputed_{col}" in df.columns else 0,
            }
            for col in [
                "demandOutturn",
                "windForecast",
                "windOnshoreForecast",
                "windOffshoreForecast",
                "solarForecast",
                "deratedMargin",
                "lossOfLoadProbability",
                "demandForecastError",
                "windForecastError",
                "solarForecastError",
            ]
        ]
    )

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    log.to_csv(IMPUTATION_LOG, index=False)

    print("2023-2025 fundamentals imputation complete")
    print(f"shape = {df.shape}")
    print(f"rows with any imputed value = {int(df['any_imputed'].sum())}")
    print(f"rows with any remaining missing value = {int(df.isna().any(axis=1).sum())}")
    print(f"Saved CSV to {OUTPUT_CSV}")
    print(f"Saved Parquet to {OUTPUT_PARQUET}")
    print(f"Saved imputation log to {IMPUTATION_LOG}")


if __name__ == "__main__":
    main()
