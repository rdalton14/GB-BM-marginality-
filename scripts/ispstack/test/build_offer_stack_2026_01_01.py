from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/ispstack/q1_2026/2026-01-01.parquet")
OUTPUT_PATH = Path("data/processed/offer_stack/2026-01-01_offer_stack.parquet")
OUTPUT_CSV_PATH = Path("data/processed/offer_stack/2026-01-01_offer_stack.csv")

KEEP_COLS = [
    "settlementDate",
    "settlementPeriod",
    "startTime",
    "id",
    "acceptanceId",
    "bidOfferPairId",
    "sequenceNumber",
    "originalPrice",
    "finalPrice",
    "volume",
    "parAdjustedVolume",
    "tlmAdjustedVolume",
    "tlmAdjustedCost",
    "direction",
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    raw_row_count = len(df)

    offer_mask = df["direction"].astype(str).str.upper() == "OFFER"
    offer_df = df.loc[offer_mask].copy()
    offer_row_count_before_filtering = len(offer_df)

    par_adjusted = pd.to_numeric(offer_df["parAdjustedVolume"], errors="coerce")

    clean_offer_stack = offer_df.loc[
        (offer_df["cadlFlag"] != True)
        & (offer_df["soFlag"] != True)
        & (offer_df["storProviderFlag"] != True)
        & (par_adjusted > 0)
        & (offer_df["finalPrice"].notna())
    ].copy()

    clean_offer_stack = clean_offer_stack[KEEP_COLS].copy()
    clean_offer_stack = clean_offer_stack.sort_values(
        by=["settlementDate", "settlementPeriod", "finalPrice"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    clean_row_count = len(clean_offer_stack)
    n_settlement_periods = clean_offer_stack["settlementPeriod"].nunique()

    print(f"Raw row count: {raw_row_count:,}")
    print(f"Offer-side row count before filtering: {offer_row_count_before_filtering:,}")
    print(f"Clean offer stack row count after filtering: {clean_row_count:,}")
    print(f"Number of settlement periods covered: {n_settlement_periods:,}")

    clean_offer_stack.to_parquet(OUTPUT_PATH, index=False)
    clean_offer_stack.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved cleaned offer stack to: {OUTPUT_PATH}")
    print(f"Saved cleaned offer stack to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
