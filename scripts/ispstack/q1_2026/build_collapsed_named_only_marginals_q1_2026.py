from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_CSV = Path("data/processed/clean_offer_energy_q1_2026/marginal_offer_energy_named_only_q1_2026.csv")
SYSTEM_PRICE_CSV = Path("data/processed/q1_2026/fundamentals/system_price_niv_q1_2026.csv")
OUTPUT_DIR = Path("data/processed/clean_offer_energy_q1_2026")

BMU_COLLAPSED_CSV = OUTPUT_DIR / "marginal_offer_energy_named_only_bmu_collapsed_q1_2026.csv"
BMU_COLLAPSED_PARQUET = OUTPUT_DIR / "marginal_offer_energy_named_only_bmu_collapsed_q1_2026.parquet"
BMU_MULTIHOT_CSV = OUTPUT_DIR / "marginal_offer_multihot_named_only_bmu_collapsed_q1_2026.csv"
BMU_MULTIHOT_PARQUET = OUTPUT_DIR / "marginal_offer_multihot_named_only_bmu_collapsed_q1_2026.parquet"
BMU_MULTIHOT_PRICE_CSV = OUTPUT_DIR / "marginal_offer_multihot_named_only_bmu_collapsed_with_system_price_q1_2026.csv"
BMU_MULTIHOT_PRICE_PARQUET = OUTPUT_DIR / "marginal_offer_multihot_named_only_bmu_collapsed_with_system_price_q1_2026.parquet"

FAMILY_COLLAPSED_CSV = OUTPUT_DIR / "marginal_offer_energy_named_only_family_collapsed_q1_2026.csv"
FAMILY_COLLAPSED_PARQUET = OUTPUT_DIR / "marginal_offer_energy_named_only_family_collapsed_q1_2026.parquet"
FAMILY_MULTIHOT_CSV = OUTPUT_DIR / "marginal_offer_multihot_named_only_family_collapsed_q1_2026.csv"
FAMILY_MULTIHOT_PARQUET = OUTPUT_DIR / "marginal_offer_multihot_named_only_family_collapsed_q1_2026.parquet"
FAMILY_MULTIHOT_PRICE_CSV = OUTPUT_DIR / "marginal_offer_multihot_named_only_family_collapsed_with_system_price_q1_2026.csv"
FAMILY_MULTIHOT_PRICE_PARQUET = OUTPUT_DIR / "marginal_offer_multihot_named_only_family_collapsed_with_system_price_q1_2026.parquet"

SUMMARY_CSV = OUTPUT_DIR / "marginal_offer_named_only_collapse_summary_q1_2026.csv"

TRAILING_3_DIGIT_RE = re.compile(r"^(.*?)(\d{3})$")


def semantic_family(unit_id: object) -> str:
    if pd.isna(unit_id):
        return ""
    text = str(unit_id).strip()
    if "-" in text:
        return text.rsplit("-", 1)[0]
    match = TRAILING_3_DIGIT_RE.match(text)
    if "__" in text and match:
        return match.group(1)
    return text


def build_multihot(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    source = df[["settlementDate", "settlementPeriod", feature_col]].dropna(subset=[feature_col]).drop_duplicates().copy()
    source["value"] = 1
    return (
        source.pivot(index=["settlementDate", "settlementPeriod"], columns=feature_col, values="value")
        .fillna(0)
        .astype("int8")
        .reset_index()
        .sort_values(["settlementDate", "settlementPeriod"])
        .reset_index(drop=True)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    price_df = pd.read_csv(SYSTEM_PRICE_CSV)

    df["settlementDate"] = pd.to_datetime(df["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    df["sequenceNumber"] = pd.to_numeric(df["sequenceNumber"], errors="coerce")
    df["finalPrice"] = pd.to_numeric(df["finalPrice"], errors="coerce")
    df["semantic_family"] = df["id"].map(semantic_family)

    price_df["settlementDate"] = pd.to_datetime(price_df["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    price_df["settlementPeriod"] = pd.to_numeric(price_df["settlementPeriod"], errors="coerce").astype("Int64")
    price_df["systemPrice"] = pd.to_numeric(price_df["systemPrice"], errors="coerce")

    # Keep the latest sequence when the same BMU repeats within an SP.
    bmu_collapsed = (
        df.sort_values(["settlementDate", "settlementPeriod", "id", "sequenceNumber"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["settlementDate", "settlementPeriod", "id"], keep="first")
        .sort_values(["settlementDate", "settlementPeriod", "finalPrice", "sequenceNumber"], ascending=[True, True, False, False])
        .reset_index(drop=True)
    )

    # Collapse related units to a cautious semantic family root within each SP.
    family_collapsed = (
        df.sort_values(["settlementDate", "settlementPeriod", "semantic_family", "sequenceNumber"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["settlementDate", "settlementPeriod", "semantic_family"], keep="first")
        .sort_values(["settlementDate", "settlementPeriod", "finalPrice", "sequenceNumber"], ascending=[True, True, False, False])
        .reset_index(drop=True)
    )

    bmu_multihot = build_multihot(bmu_collapsed, "id")
    family_multihot = build_multihot(family_collapsed, "semantic_family")

    bmu_multihot_with_price = bmu_multihot.merge(
        price_df[["settlementDate", "settlementPeriod", "systemPrice"]],
        on=["settlementDate", "settlementPeriod"],
        how="left",
    )
    family_multihot_with_price = family_multihot.merge(
        price_df[["settlementDate", "settlementPeriod", "systemPrice"]],
        on=["settlementDate", "settlementPeriod"],
        how="left",
    )

    original_sp = df[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]

    summary = pd.DataFrame(
        [
            {"metric": "original_named_joint_rows", "value": len(df)},
            {"metric": "original_named_sps", "value": original_sp},
            {"metric": "bmu_collapsed_rows", "value": len(bmu_collapsed)},
            {"metric": "bmu_collapsed_sps", "value": bmu_collapsed[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]},
            {"metric": "family_collapsed_rows", "value": len(family_collapsed)},
            {"metric": "family_collapsed_sps", "value": family_collapsed[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]},
            {"metric": "bmu_multihot_feature_count", "value": bmu_multihot.shape[1] - 2},
            {"metric": "family_multihot_feature_count", "value": family_multihot.shape[1] - 2},
            {"metric": "bmu_multihot_missing_system_price_rows", "value": int(bmu_multihot_with_price["systemPrice"].isna().sum())},
            {"metric": "family_multihot_missing_system_price_rows", "value": int(family_multihot_with_price["systemPrice"].isna().sum())},
        ]
    )

    print(f"Original named joint-marginal rows: {len(df):,}")
    print(f"Original named SPs: {original_sp:,}")
    print(f"BMU-collapsed rows: {len(bmu_collapsed):,}")
    print(f"Family-collapsed rows: {len(family_collapsed):,}")
    print(f"BMU multihot features: {bmu_multihot.shape[1] - 2:,}")
    print(f"Family multihot features: {family_multihot.shape[1] - 2:,}")
    print(f"BMU multihot rows missing systemPrice: {int(bmu_multihot_with_price['systemPrice'].isna().sum()):,}")
    print(f"Family multihot rows missing systemPrice: {int(family_multihot_with_price['systemPrice'].isna().sum()):,}")

    bmu_collapsed.to_csv(BMU_COLLAPSED_CSV, index=False)
    bmu_collapsed.to_parquet(BMU_COLLAPSED_PARQUET, index=False)
    bmu_multihot.to_csv(BMU_MULTIHOT_CSV, index=False)
    bmu_multihot.to_parquet(BMU_MULTIHOT_PARQUET, index=False)
    bmu_multihot_with_price.to_csv(BMU_MULTIHOT_PRICE_CSV, index=False)
    bmu_multihot_with_price.to_parquet(BMU_MULTIHOT_PRICE_PARQUET, index=False)

    family_collapsed.to_csv(FAMILY_COLLAPSED_CSV, index=False)
    family_collapsed.to_parquet(FAMILY_COLLAPSED_PARQUET, index=False)
    family_multihot.to_csv(FAMILY_MULTIHOT_CSV, index=False)
    family_multihot.to_parquet(FAMILY_MULTIHOT_PARQUET, index=False)
    family_multihot_with_price.to_csv(FAMILY_MULTIHOT_PRICE_CSV, index=False)
    family_multihot_with_price.to_parquet(FAMILY_MULTIHOT_PRICE_PARQUET, index=False)

    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"\nSaved: {BMU_COLLAPSED_CSV}")
    print(f"Saved: {BMU_MULTIHOT_PRICE_CSV}")
    print(f"Saved: {FAMILY_COLLAPSED_CSV}")
    print(f"Saved: {FAMILY_MULTIHOT_PRICE_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
