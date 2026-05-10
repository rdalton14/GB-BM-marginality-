from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/ispstack/q1_2026/2026-01-01.parquet")
OUTPUT_DIR = Path("data/processed/diagnostics")
SO_CONTRIBUTORS_PATH = OUTPUT_DIR / "2026-01-01_so_price_contributors.parquet"
SO_BY_SP_PATH = OUTPUT_DIR / "2026-01-01_so_contribution_by_sp.parquet"
MARGINALS_PATH = OUTPUT_DIR / "2026-01-01_marginal_price_contributors.parquet"

REQUIRED_COLS = [
    "settlementDate",
    "settlementPeriod",
    "direction",
    "id",
    "acceptanceId",
    "bidOfferPairId",
    "sequenceNumber",
    "originalPrice",
    "finalPrice",
    "volume",
    "dmatAdjustedVolume",
    "arbitrageAdjustedVolume",
    "nivAdjustedVolume",
    "parAdjustedVolume",
    "tlmAdjustedVolume",
    "tlmAdjustedCost",
    "soFlag",
    "cadlFlag",
    "storProviderFlag",
    "repricedIndicator",
]


def build_marginals(df: pd.DataFrame) -> pd.DataFrame:
    contributors = df.loc[
        (df["parAdjustedVolume_num"] > 0) & df["finalPrice_num"].notna()
    ].copy()

    if contributors.empty:
        return contributors

    offer_rows = contributors.loc[contributors["direction_u"] == "OFFER"].copy()
    bid_rows = contributors.loc[contributors["direction_u"] == "BID"].copy()

    if not offer_rows.empty:
        offer_max = offer_rows.groupby(["settlementDate", "settlementPeriod"])["finalPrice_num"].transform("max")
        offer_marginals = offer_rows.loc[offer_rows["finalPrice_num"] == offer_max].copy()
    else:
        offer_marginals = offer_rows

    if not bid_rows.empty:
        bid_min = bid_rows.groupby(["settlementDate", "settlementPeriod"])["finalPrice_num"].transform("min")
        bid_marginals = bid_rows.loc[bid_rows["finalPrice_num"] == bid_min].copy()
    else:
        bid_marginals = bid_rows

    cols = [
        "settlementDate",
        "settlementPeriod",
        "direction",
        "id",
        "acceptanceId",
        "finalPrice",
        "originalPrice",
        "parAdjustedVolume",
        "tlmAdjustedCost",
        "soFlag",
        "cadlFlag",
        "storProviderFlag",
        "repricedIndicator",
    ]
    return pd.concat([offer_marginals[cols], bid_marginals[cols]], ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df["direction_u"] = df["direction"].astype(str).str.upper()
    for col in [
        "settlementPeriod",
        "sequenceNumber",
        "originalPrice",
        "finalPrice",
        "volume",
        "dmatAdjustedVolume",
        "arbitrageAdjustedVolume",
        "nivAdjustedVolume",
        "parAdjustedVolume",
        "tlmAdjustedVolume",
        "tlmAdjustedCost",
    ]:
        df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce")

    total_raw_rows = len(df)
    total_so_rows = int((df["soFlag"] == True).sum())
    total_so_par_positive = int(((df["soFlag"] == True) & (df["parAdjustedVolume_num"] > 0)).sum())

    so_price_contributors = df.loc[
        (df["soFlag"] == True)
        & (df["parAdjustedVolume_num"] > 0)
        & df["finalPrice_num"].notna()
    ].copy()

    total_so_par_positive_with_price = len(so_price_contributors)
    so_sps = so_price_contributors["settlementPeriod_num"].dropna().astype(int).nunique()
    so_sp_share = so_sps / 48 if 48 else 0.0

    by_sp = (
        so_price_contributors.groupby(["settlementDate", "settlementPeriod", "direction"], dropna=False)
        .agg(
            so_price_contributing_rows=("id", "size"),
            parAdjustedVolume_sum=("parAdjustedVolume_num", "sum"),
            tlmAdjustedVolume_sum=("tlmAdjustedVolume_num", "sum"),
            tlmAdjustedCost_sum=("tlmAdjustedCost_num", "sum"),
            finalPrice_min=("finalPrice_num", "min"),
            finalPrice_max=("finalPrice_num", "max"),
            finalPrice_mean=("finalPrice_num", "mean"),
        )
        .reset_index()
    )

    marginals = build_marginals(df)

    offer_marginals = marginals.loc[marginals["direction"].astype(str).str.upper() == "OFFER"]
    bid_marginals = marginals.loc[marginals["direction"].astype(str).str.upper() == "BID"]

    offer_so_count = int((offer_marginals["soFlag"] == True).sum()) if not offer_marginals.empty else 0
    bid_so_count = int((bid_marginals["soFlag"] == True).sum()) if not bid_marginals.empty else 0
    offer_share = offer_so_count / len(offer_marginals) if len(offer_marginals) else 0.0
    bid_share = bid_so_count / len(bid_marginals) if len(bid_marginals) else 0.0

    print(f"Total raw rows: {total_raw_rows:,}")
    print(f"Total SO-flagged rows: {total_so_rows:,}")
    print(f"Total SO-flagged rows with parAdjustedVolume > 0: {total_so_par_positive:,}")
    print(
        "Total SO-flagged rows with parAdjustedVolume > 0 and finalPrice not null: "
        f"{total_so_par_positive_with_price:,}"
    )
    print(f"Number of settlement periods where SO-flagged actions contributed: {so_sps:,}")
    print(f"Share of all 48 settlement periods where SO-flagged actions contributed: {so_sp_share:.4f}")
    print()
    print(f"OFFER marginal actions that were SO-flagged: {offer_so_count:,}")
    print(f"BID marginal actions that were SO-flagged: {bid_so_count:,}")
    print(f"Share of OFFER marginal actions that were SO-flagged: {offer_share:.4f}")
    print(f"Share of BID marginal actions that were SO-flagged: {bid_share:.4f}")

    so_price_contributors[REQUIRED_COLS].to_parquet(SO_CONTRIBUTORS_PATH, index=False)
    by_sp.to_parquet(SO_BY_SP_PATH, index=False)
    marginals.to_parquet(MARGINALS_PATH, index=False)

    print()
    print(f"Saved: {SO_CONTRIBUTORS_PATH}")
    print(f"Saved: {SO_BY_SP_PATH}")
    print(f"Saved: {MARGINALS_PATH}")


if __name__ == "__main__":
    main()
