from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_DIR = Path("data/raw/ispstack/full_2023_2025")
SYSTEM_PRICE_CSV = Path("data/processed/full_2023_2025/fundamentals/system_price_niv_2023_2025.csv")
OUTPUT_DIR = Path("data/processed/full_2023_2025/clean_offer_energy_2023_2025")

CLEAN_STACK_PARQUET = OUTPUT_DIR / "clean_offer_energy_stack_2023_2025.parquet"
MARGINAL_PARQUET = OUTPUT_DIR / "marginal_offer_energy_2023_2025.parquet"
MARGINAL_CSV = OUTPUT_DIR / "marginal_offer_energy_2023_2025.csv"
BMU_COLLAPSED_PARQUET = OUTPUT_DIR / "marginal_offer_energy_bmu_collapsed_2023_2025.parquet"
BMU_COLLAPSED_CSV = OUTPUT_DIR / "marginal_offer_energy_bmu_collapsed_2023_2025.csv"
MULTIHOT_PARQUET = OUTPUT_DIR / "marginal_offer_multihot_bmu_collapsed_2023_2025.parquet"
MULTIHOT_CSV = OUTPUT_DIR / "marginal_offer_multihot_bmu_collapsed_2023_2025.csv"
MULTIHOT_SP_PARQUET = OUTPUT_DIR / "marginal_offer_multihot_bmu_collapsed_with_system_price_2023_2025.parquet"
MULTIHOT_SP_CSV = OUTPUT_DIR / "marginal_offer_multihot_bmu_collapsed_with_system_price_2023_2025.csv"
SUMMARY_CSV = OUTPUT_DIR / "clean_offer_energy_2023_2025_summary.csv"

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2025-12-31")
NUMERIC_ONLY_RE = re.compile(r"^\d+(\.\d+)?$")

RAW_COLUMNS = [
    "settlementDate",
    "settlementPeriod",
    "startTime",
    "createdDateTime",
    "sequenceNumber",
    "id",
    "acceptanceId",
    "bidOfferPairId",
    "cadlFlag",
    "soFlag",
    "storProviderFlag",
    "repricedIndicator",
    "reserveScarcityPrice",
    "originalPrice",
    "volume",
    "dmatAdjustedVolume",
    "arbitrageAdjustedVolume",
    "nivAdjustedVolume",
    "parAdjustedVolume",
    "finalPrice",
    "transmissionLossMultiplier",
    "tlmAdjustedVolume",
    "tlmAdjustedCost",
    "direction",
]


def is_numeric_only(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(NUMERIC_ONLY_RE.fullmatch(str(value).strip()))


def load_full_files() -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    file_count = 0

    for path in sorted(INPUT_DIR.glob("*.parquet")):
        try:
            file_date = pd.Timestamp(path.stem)
        except ValueError:
            continue
        if not (START_DATE <= file_date <= END_DATE):
            continue
        frames.append(pd.read_parquet(path, columns=RAW_COLUMNS))
        file_count += 1

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")

    return pd.concat(frames, ignore_index=True), file_count


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, file_count = load_full_files()
    price_df = pd.read_csv(SYSTEM_PRICE_CSV)

    df["settlementDate"] = pd.to_datetime(df["settlementDate"], errors="coerce")
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    df["direction"] = df["direction"].astype(str).str.upper()
    df["finalPrice"] = pd.to_numeric(df["finalPrice"], errors="coerce")
    df["originalPrice"] = pd.to_numeric(df["originalPrice"], errors="coerce")
    df["parAdjustedVolume"] = pd.to_numeric(df["parAdjustedVolume"], errors="coerce")
    df["sequenceNumber"] = pd.to_numeric(df["sequenceNumber"], errors="coerce")

    total_raw_rows = len(df)
    raw_offer_rows = int((df["direction"] == "OFFER").sum())

    clean_offer = df.loc[
        (df["direction"] == "OFFER")
        & (df["finalPrice"].notna())
        & (df["soFlag"] != True)
        & (df["cadlFlag"] != True)
        & (df["storProviderFlag"] != True)
    ].copy()

    clean_offer["abs_par_volume"] = clean_offer["parAdjustedVolume"].abs()
    clean_offer["price_changed"] = clean_offer["originalPrice"] != clean_offer["finalPrice"]
    clean_offer["is_numeric_id"] = clean_offer["id"].map(is_numeric_only)
    clean_offer["is_missing_id"] = clean_offer["id"].isna()
    clean_offer["raw_cost_proxy"] = clean_offer["originalPrice"] * clean_offer["parAdjustedVolume"]
    clean_offer["final_cost_proxy"] = clean_offer["finalPrice"] * clean_offer["parAdjustedVolume"]
    clean_offer["abs_final_cost_proxy"] = clean_offer["final_cost_proxy"].abs()

    clean_offer = clean_offer.sort_values(
        ["settlementDate", "settlementPeriod", "finalPrice", "sequenceNumber"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    max_price_by_sp = clean_offer.groupby(["settlementDate", "settlementPeriod"], dropna=False)["finalPrice"].transform("max")
    marginal_offer = (
        clean_offer.loc[clean_offer["finalPrice"] == max_price_by_sp]
        .copy()
        .sort_values(["settlementDate", "settlementPeriod", "finalPrice", "sequenceNumber"], ascending=[True, True, False, False])
        .reset_index(drop=True)
    )

    # Collapse repeated appearances of the same raw unit ID within each SP.
    bmu_collapsed = (
        marginal_offer.sort_values(["settlementDate", "settlementPeriod", "id", "sequenceNumber"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["settlementDate", "settlementPeriod", "id"], keep="first")
        .sort_values(["settlementDate", "settlementPeriod", "finalPrice", "sequenceNumber"], ascending=[True, True, False, False])
        .reset_index(drop=True)
    )

    multihot_source = bmu_collapsed[["settlementDate", "settlementPeriod", "id"]].dropna(subset=["id"]).drop_duplicates().copy()
    multihot_source["value"] = 1
    multihot = (
        multihot_source.pivot(index=["settlementDate", "settlementPeriod"], columns="id", values="value")
        .fillna(0)
        .astype("int8")
        .reset_index()
        .sort_values(["settlementDate", "settlementPeriod"])
        .reset_index(drop=True)
    )

    price_df["settlementDate"] = pd.to_datetime(price_df["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    price_df["settlementPeriod"] = pd.to_numeric(price_df["settlementPeriod"], errors="coerce").astype("Int64")
    price_df["systemPrice"] = pd.to_numeric(price_df["systemPrice"], errors="coerce")

    multihot["settlementDate"] = pd.to_datetime(multihot["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    multihot_with_price = multihot.merge(
        price_df[["settlementDate", "settlementPeriod", "systemPrice"]],
        on=["settlementDate", "settlementPeriod"],
        how="left",
    )

    unique_dates = int(df["settlementDate"].dt.normalize().nunique())
    expected_sps = unique_dates * 48
    sp_covered_clean = int(clean_offer[["settlementDate", "settlementPeriod"]].dropna().drop_duplicates().shape[0])
    sp_covered_named = int(multihot[["settlementDate", "settlementPeriod"]].dropna().drop_duplicates().shape[0])

    summary = pd.DataFrame(
        [
            {"metric": "files_loaded", "value": file_count},
            {"metric": "total_raw_rows", "value": total_raw_rows},
            {"metric": "raw_offer_rows", "value": raw_offer_rows},
            {"metric": "clean_offer_rows", "value": len(clean_offer)},
            {"metric": "clean_offer_share_of_raw_offer_rows", "value": (len(clean_offer) / raw_offer_rows) if raw_offer_rows else 0.0},
            {"metric": "joint_marginal_rows", "value": len(marginal_offer)},
            {"metric": "bmu_collapsed_marginal_rows", "value": len(bmu_collapsed)},
            {"metric": "unique_settlement_dates", "value": unique_dates},
            {"metric": "expected_sps_48_per_day", "value": expected_sps},
            {"metric": "sps_covered_by_clean_offer_stack", "value": sp_covered_clean},
            {"metric": "sps_covered_by_bmu_collapsed_marginals", "value": sp_covered_named},
            {"metric": "numeric_id_share_clean_offer", "value": float(clean_offer["is_numeric_id"].mean()) if len(clean_offer) else 0.0},
            {"metric": "numeric_id_share_joint_marginal", "value": float(marginal_offer["is_numeric_id"].mean()) if len(marginal_offer) else 0.0},
            {"metric": "rows_missing_system_price_after_join", "value": int(multihot_with_price["systemPrice"].isna().sum())},
            {"metric": "multihot_feature_count", "value": multihot.shape[1] - 2},
        ]
    )

    print(f"Files loaded: {file_count:,}")
    print(f"Total raw rows: {total_raw_rows:,}")
    print(f"Raw offer rows: {raw_offer_rows:,}")
    print(f"Clean offer rows: {len(clean_offer):,}")
    print(f"Joint marginal rows: {len(marginal_offer):,}")
    print(f"BMU-collapsed marginal rows: {len(bmu_collapsed):,}")
    print(f"Unique settlement dates: {unique_dates:,}")
    print(f"Expected SPs: {expected_sps:,}")
    print(f"SPs covered by clean offer stack: {sp_covered_clean:,}")
    print(f"SPs covered by BMU-collapsed marginals: {sp_covered_named:,}")
    print(f"Numeric-ID share in clean offer: {float(clean_offer['is_numeric_id'].mean()) if len(clean_offer) else 0.0:.4f}")
    print(f"Numeric-ID share in joint marginal rows: {float(marginal_offer['is_numeric_id'].mean()) if len(marginal_offer) else 0.0:.4f}")
    print(f"Rows missing systemPrice after join: {int(multihot_with_price['systemPrice'].isna().sum()):,}")
    print(f"Multihot feature count: {multihot.shape[1] - 2:,}")

    clean_offer.to_parquet(CLEAN_STACK_PARQUET, index=False)
    marginal_offer.to_parquet(MARGINAL_PARQUET, index=False)
    marginal_offer.to_csv(MARGINAL_CSV, index=False)
    bmu_collapsed.to_parquet(BMU_COLLAPSED_PARQUET, index=False)
    bmu_collapsed.to_csv(BMU_COLLAPSED_CSV, index=False)
    multihot.to_parquet(MULTIHOT_PARQUET, index=False)
    multihot.to_csv(MULTIHOT_CSV, index=False)
    multihot_with_price.to_parquet(MULTIHOT_SP_PARQUET, index=False)
    multihot_with_price.to_csv(MULTIHOT_SP_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"\nSaved: {MARGINAL_CSV}")
    print(f"Saved: {BMU_COLLAPSED_CSV}")
    print(f"Saved: {MULTIHOT_SP_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
