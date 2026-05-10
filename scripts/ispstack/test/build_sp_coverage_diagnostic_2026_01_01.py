from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/ispstack/q1_2026/2026-01-01.parquet")
OUTPUT_PATH = Path("data/processed/diagnostics/2026-01-01_sp_coverage_diagnostic.parquet")
OUTPUT_CSV_PATH = Path("data/processed/diagnostics/2026-01-01_sp_coverage_diagnostic.csv")


def clean_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    direction_mask = df["direction"].astype(str).str.upper() == side.upper()
    par_adjusted = pd.to_numeric(df["parAdjustedVolume"], errors="coerce")
    return df.loc[
        direction_mask
        & (df["cadlFlag"] != True)
        & (df["soFlag"] != True)
        & (df["storProviderFlag"] != True)
        & (par_adjusted > 0)
        & (df["finalPrice"].notna())
    ].copy()


def classify_row(row: pd.Series) -> str:
    if row["clean_offer_rows"] > 0:
        return "clean_offer_available"
    if row["clean_offer_rows"] == 0 and row["clean_bid_rows"] > 0:
        return "clean_bid_available"
    if row["raw_total_rows"] > 0:
        return "only_flagged_or_filtered_actions"
    return "no_ispstack_rows"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)

    clean_offer = clean_side(df, "OFFER")
    clean_bid = clean_side(df, "BID")

    clean_offer_sps = set(pd.to_numeric(clean_offer["settlementPeriod"], errors="coerce").dropna().astype(int).tolist())
    all_sps = set(range(1, 49))
    missing_offer_sps = sorted(all_sps - clean_offer_sps)

    diagnostics = []
    for sp in range(1, 49):
        sp_df = df.loc[pd.to_numeric(df["settlementPeriod"], errors="coerce") == sp].copy()
        offer_df = sp_df.loc[sp_df["direction"].astype(str).str.upper() == "OFFER"]
        bid_df = sp_df.loc[sp_df["direction"].astype(str).str.upper() == "BID"]
        clean_offer_sp = clean_offer.loc[pd.to_numeric(clean_offer["settlementPeriod"], errors="coerce") == sp]
        clean_bid_sp = clean_bid.loc[pd.to_numeric(clean_bid["settlementPeriod"], errors="coerce") == sp]

        diagnostics.append(
            {
                "settlementDate": "2026-01-01",
                "settlementPeriod": sp,
                "raw_total_rows": int(len(sp_df)),
                "raw_offer_rows": int(len(offer_df)),
                "raw_bid_rows": int(len(bid_df)),
                "clean_offer_rows": int(len(clean_offer_sp)),
                "clean_bid_rows": int(len(clean_bid_sp)),
                "so_flagged_rows": int((sp_df["soFlag"] == True).sum()),
                "cadl_flagged_rows": int((sp_df["cadlFlag"] == True).sum()),
                "stor_provider_rows": int((sp_df["storProviderFlag"] == True).sum()),
                "par_positive_rows": int((pd.to_numeric(sp_df["parAdjustedVolume"], errors="coerce") > 0).sum()),
                "final_price_not_null_rows": int(sp_df["finalPrice"].notna().sum()),
            }
        )

    diag_df = pd.DataFrame(diagnostics)
    diag_df["classification"] = diag_df.apply(classify_row, axis=1)

    print(f"Raw row count: {len(df):,}")
    print(f"Clean offer row count: {len(clean_offer):,}")
    print(f"Missing settlement periods from clean offer stack: {missing_offer_sps}")
    print("\nClassification summary:")
    print(diag_df["classification"].value_counts().to_string())

    diag_df.to_parquet(OUTPUT_PATH, index=False)
    diag_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved diagnostic table to: {OUTPUT_PATH}")
    print(f"Saved diagnostic table to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
