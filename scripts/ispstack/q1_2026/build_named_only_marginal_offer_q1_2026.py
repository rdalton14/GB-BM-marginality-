from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_LONG_CSV = Path("data/processed/clean_offer_energy_q1_2026/marginal_offer_energy_q1_2026.csv")
SYSTEM_PRICE_CSV = Path("data/processed/q1_2026/fundamentals/system_price_niv_q1_2026.csv")
OUTPUT_DIR = Path("data/processed/clean_offer_energy_q1_2026")

NAMED_LONG_CSV = OUTPUT_DIR / "marginal_offer_energy_named_only_q1_2026.csv"
NAMED_LONG_PARQUET = OUTPUT_DIR / "marginal_offer_energy_named_only_q1_2026.parquet"
NAMED_MULTIHOT_CSV = OUTPUT_DIR / "marginal_offer_multihot_named_only_q1_2026.csv"
NAMED_MULTIHOT_PARQUET = OUTPUT_DIR / "marginal_offer_multihot_named_only_q1_2026.parquet"
NAMED_MULTIHOT_WITH_PRICE_CSV = OUTPUT_DIR / "marginal_offer_multihot_named_only_with_system_price_q1_2026.csv"
NAMED_MULTIHOT_WITH_PRICE_PARQUET = OUTPUT_DIR / "marginal_offer_multihot_named_only_with_system_price_q1_2026.parquet"
SUMMARY_CSV = OUTPUT_DIR / "marginal_offer_named_only_q1_2026_summary.csv"

NUMERIC_ONLY_RE = re.compile(r"^\d+(\.\d+)?$")


def is_numeric_only(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(NUMERIC_ONLY_RE.fullmatch(str(value).strip()))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(INPUT_LONG_CSV)
    price_df = pd.read_csv(SYSTEM_PRICE_CSV)

    long_df["settlementDate"] = pd.to_datetime(long_df["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    long_df["settlementPeriod"] = pd.to_numeric(long_df["settlementPeriod"], errors="coerce").astype("Int64")
    long_df["is_numeric_id"] = long_df["id"].map(is_numeric_only)

    price_df["settlementDate"] = pd.to_datetime(price_df["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    price_df["settlementPeriod"] = pd.to_numeric(price_df["settlementPeriod"], errors="coerce").astype("Int64")
    price_df["systemPrice"] = pd.to_numeric(price_df["systemPrice"], errors="coerce")

    named_long = long_df.loc[~long_df["is_numeric_id"]].copy().reset_index(drop=True)

    multihot_source = named_long[["settlementDate", "settlementPeriod", "id"]].dropna(subset=["id"]).drop_duplicates().copy()
    multihot_source["value"] = 1

    named_multihot = (
        multihot_source.pivot(index=["settlementDate", "settlementPeriod"], columns="id", values="value")
        .fillna(0)
        .astype("int8")
        .reset_index()
        .sort_values(["settlementDate", "settlementPeriod"])
        .reset_index(drop=True)
    )

    named_multihot_with_price = named_multihot.merge(
        price_df[["settlementDate", "settlementPeriod", "systemPrice"]],
        on=["settlementDate", "settlementPeriod"],
        how="left",
    )

    all_sp = long_df[["settlementDate", "settlementPeriod"]].drop_duplicates()
    named_sp = named_long[["settlementDate", "settlementPeriod"]].drop_duplicates()

    summary = pd.DataFrame(
        [
            {"metric": "total_joint_marginal_rows", "value": len(long_df)},
            {"metric": "named_only_joint_marginal_rows", "value": len(named_long)},
            {"metric": "removed_numeric_rows", "value": int(long_df["is_numeric_id"].sum())},
            {"metric": "removed_numeric_row_share", "value": float(long_df["is_numeric_id"].mean())},
            {"metric": "total_settlement_periods_in_original", "value": len(all_sp)},
            {"metric": "settlement_periods_with_any_named_joint_marginal", "value": len(named_sp)},
            {
                "metric": "settlement_periods_lost_after_removing_numeric_only_rows",
                "value": int(len(all_sp.merge(named_sp, on=["settlementDate", "settlementPeriod"], how="left", indicator=True).query('_merge == \"left_only\"'))),
            },
            {"metric": "named_multihot_rows", "value": len(named_multihot)},
            {"metric": "named_multihot_feature_count_excluding_keys", "value": named_multihot.shape[1] - 2},
            {"metric": "named_multihot_rows_missing_systemPrice", "value": int(named_multihot_with_price["systemPrice"].isna().sum())},
        ]
    )

    print(f"Original joint marginal rows: {len(long_df):,}")
    print(f"Named-only joint marginal rows: {len(named_long):,}")
    print(f"Removed numeric rows: {int(long_df['is_numeric_id'].sum()):,}")
    print(f"Original SPs: {len(all_sp):,}")
    print(f"SPs with any named joint marginal left: {len(named_sp):,}")
    print(f"Named-only multihot rows: {len(named_multihot):,}")
    print(f"Named-only multihot feature columns: {named_multihot.shape[1] - 2:,}")
    print(f"Rows missing systemPrice after join: {int(named_multihot_with_price['systemPrice'].isna().sum()):,}")

    named_long.to_csv(NAMED_LONG_CSV, index=False)
    named_long.to_parquet(NAMED_LONG_PARQUET, index=False)
    named_multihot.to_csv(NAMED_MULTIHOT_CSV, index=False)
    named_multihot.to_parquet(NAMED_MULTIHOT_PARQUET, index=False)
    named_multihot_with_price.to_csv(NAMED_MULTIHOT_WITH_PRICE_CSV, index=False)
    named_multihot_with_price.to_parquet(NAMED_MULTIHOT_WITH_PRICE_PARQUET, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"\nSaved: {NAMED_LONG_CSV}")
    print(f"Saved: {NAMED_LONG_PARQUET}")
    print(f"Saved: {NAMED_MULTIHOT_CSV}")
    print(f"Saved: {NAMED_MULTIHOT_PARQUET}")
    print(f"Saved: {NAMED_MULTIHOT_WITH_PRICE_CSV}")
    print(f"Saved: {NAMED_MULTIHOT_WITH_PRICE_PARQUET}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
