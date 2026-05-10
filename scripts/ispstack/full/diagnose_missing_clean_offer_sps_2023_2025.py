from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/ispstack/full_2023_2025")
SYSTEM_PRICE_CSV = Path("data/processed/full_2023_2025/fundamentals/system_price_niv_2023_2025.csv")
OUTPUT_DIR = Path("data/processed/full_2023_2025/clean_offer_energy_2023_2025")

OUTPUT_CSV = OUTPUT_DIR / "missing_clean_offer_sp_diagnostic_2023_2025.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "missing_clean_offer_sp_diagnostic_2023_2025_summary.csv"

RAW_COLUMNS = [
    "settlementDate",
    "settlementPeriod",
    "direction",
    "soFlag",
    "cadlFlag",
    "storProviderFlag",
    "finalPrice",
]


def load_raw_stack() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(RAW_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(path, columns=RAW_COLUMNS))
    if not frames:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_stack()
    spine = pd.read_csv(SYSTEM_PRICE_CSV, usecols=["settlementDate", "settlementPeriod"])

    raw["settlementDate"] = pd.to_datetime(raw["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="coerce").astype("Int64")
    raw["direction"] = raw["direction"].astype(str).str.upper()
    raw["finalPrice"] = pd.to_numeric(raw["finalPrice"], errors="coerce")

    spine["settlementDate"] = pd.to_datetime(spine["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    spine["settlementPeriod"] = pd.to_numeric(spine["settlementPeriod"], errors="coerce").astype("Int64")
    spine = spine.drop_duplicates().sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)

    rows = []
    for (settlement_date, settlement_period), g in raw.groupby(["settlementDate", "settlementPeriod"], dropna=False):
        offer = g.loc[g["direction"] == "OFFER"].copy()
        bid = g.loc[g["direction"] == "BID"].copy()

        clean_offer = offer.loc[
            offer["finalPrice"].notna()
            & (offer["soFlag"] != True)
            & (offer["cadlFlag"] != True)
            & (offer["storProviderFlag"] != True)
        ].copy()

        clean_bid = bid.loc[
            bid["finalPrice"].notna()
            & (bid["soFlag"] != True)
            & (bid["cadlFlag"] != True)
            & (bid["storProviderFlag"] != True)
        ].copy()

        if len(clean_offer) > 0:
            classification = "clean_offer_available"
        elif len(clean_bid) > 0:
            classification = "clean_bid_available_no_clean_offer"
        elif len(g) == 0:
            classification = "no_ispstack_rows"
        else:
            classification = "only_flagged_or_filtered_actions"

        rows.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": settlement_period,
                "raw_row_count": int(len(g)),
                "raw_offer_count": int(len(offer)),
                "raw_bid_count": int(len(bid)),
                "clean_offer_count": int(len(clean_offer)),
                "clean_bid_count": int(len(clean_bid)),
                "so_offer_count": int(((offer["soFlag"] == True)).sum()),
                "cadl_offer_count": int(((offer["cadlFlag"] == True)).sum()),
                "stor_offer_count": int(((offer["storProviderFlag"] == True)).sum()),
                "nonnull_finalprice_offer_count": int(offer["finalPrice"].notna().sum()),
                "classification": classification,
            }
        )

    diagnostic = spine.merge(pd.DataFrame(rows), on=["settlementDate", "settlementPeriod"], how="left")
    diagnostic["raw_row_count"] = diagnostic["raw_row_count"].fillna(0).astype(int)
    diagnostic["raw_offer_count"] = diagnostic["raw_offer_count"].fillna(0).astype(int)
    diagnostic["raw_bid_count"] = diagnostic["raw_bid_count"].fillna(0).astype(int)
    diagnostic["clean_offer_count"] = diagnostic["clean_offer_count"].fillna(0).astype(int)
    diagnostic["clean_bid_count"] = diagnostic["clean_bid_count"].fillna(0).astype(int)
    diagnostic["so_offer_count"] = diagnostic["so_offer_count"].fillna(0).astype(int)
    diagnostic["cadl_offer_count"] = diagnostic["cadl_offer_count"].fillna(0).astype(int)
    diagnostic["stor_offer_count"] = diagnostic["stor_offer_count"].fillna(0).astype(int)
    diagnostic["nonnull_finalprice_offer_count"] = diagnostic["nonnull_finalprice_offer_count"].fillna(0).astype(int)
    diagnostic["classification"] = diagnostic["classification"].fillna("no_ispstack_rows")

    summary = (
        diagnostic["classification"]
        .value_counts(dropna=False)
        .rename_axis("classification")
        .reset_index(name="sp_count")
    )
    summary["share_of_all_sps"] = summary["sp_count"] / len(diagnostic)

    missing = diagnostic.loc[diagnostic["clean_offer_count"] == 0].copy()
    print(f"Total SPs in spine: {len(diagnostic):,}")
    print(f"SPs with clean offer available: {(diagnostic['clean_offer_count'] > 0).sum():,}")
    print(f"SPs missing clean offer: {len(missing):,}")
    print("\nClassification summary")
    print(summary.to_string(index=False))

    print("\nTop missing-SP classifications")
    print(
        missing["classification"]
        .value_counts()
        .rename_axis("classification")
        .reset_index(name="sp_count")
        .to_string(index=False)
    )

    diagnostic.to_csv(OUTPUT_CSV, index=False)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
