from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/ispstack/q1_2026/2026-01-01.parquet")
DIAG_PATH = Path("data/processed/diagnostics/2026-01-01_sp_coverage_diagnostic.csv")
OUT_CSV = Path("data/processed/diagnostics/2026-01-01_missing_offer_sp_investigation.csv")
OUT_PARQUET = Path("data/processed/diagnostics/2026-01-01_missing_offer_sp_investigation.parquet")


def clean_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    mask = df["direction"].astype(str).str.upper() == side.upper()
    par = pd.to_numeric(df["parAdjustedVolume"], errors="coerce")
    return df.loc[
        mask
        & (df["cadlFlag"] != True)
        & (df["soFlag"] != True)
        & (df["storProviderFlag"] != True)
        & (par > 0)
        & (df["finalPrice"].notna())
    ].copy()


def classify_hypothesis(row: pd.Series) -> str:
    if row["offer_rows"] > row["bid_rows"] and row["offer_max_final"] >= row["bid_max_final"]:
        return "offer_side_active_but_no_par_positive_offer"
    if row["bid_rows"] > row["offer_rows"] and row["bid_max_final"] > row["offer_max_final"]:
        return "bid_side_more_active_but_no_par_positive_bid"
    if row["offer_so_share"] > 0.5 or row["bid_so_share"] > 0.5:
        return "mostly_so_flagged_activity"
    if row["offer_cadl_share"] > 0.1 or row["bid_cadl_share"] > 0.1:
        return "cadl_heavy_period"
    return "mixed_activity_no_par_positive_actions"


def main() -> None:
    raw = pd.read_parquet(RAW_PATH).copy()
    diag = pd.read_csv(DIAG_PATH)

    raw["finalPrice_num"] = pd.to_numeric(raw["finalPrice"], errors="coerce")
    raw["volume_num"] = pd.to_numeric(raw["volume"], errors="coerce")
    raw["parAdjustedVolume_num"] = pd.to_numeric(raw["parAdjustedVolume"], errors="coerce")

    missing_sps = diag.loc[
        diag["classification"] == "only_flagged_or_filtered_actions", "settlementPeriod"
    ].astype(int).tolist()

    clean_offer = clean_side(raw, "OFFER")
    clean_bid = clean_side(raw, "BID")

    records = []
    for sp in missing_sps:
        g = raw.loc[raw["settlementPeriod"] == sp].copy()
        offers = g.loc[g["direction"].astype(str).str.upper() == "OFFER"].copy()
        bids = g.loc[g["direction"].astype(str).str.upper() == "BID"].copy()
        offer_clean_sp = clean_offer.loc[clean_offer["settlementPeriod"] == sp]
        bid_clean_sp = clean_bid.loc[clean_bid["settlementPeriod"] == sp]

        record = {
            "settlementDate": "2026-01-01",
            "settlementPeriod": sp,
            "raw_rows": int(len(g)),
            "offer_rows": int(len(offers)),
            "bid_rows": int(len(bids)),
            "offer_share_rows": float(len(offers) / len(g)) if len(g) else np.nan,
            "bid_share_rows": float(len(bids) / len(g)) if len(g) else np.nan,
            "offer_abs_volume_total": float(offers["volume_num"].abs().sum()) if len(offers) else 0.0,
            "bid_abs_volume_total": float(bids["volume_num"].abs().sum()) if len(bids) else 0.0,
            "offer_max_final": float(offers["finalPrice_num"].max()) if len(offers) else np.nan,
            "offer_min_final": float(offers["finalPrice_num"].min()) if len(offers) else np.nan,
            "bid_max_final": float(bids["finalPrice_num"].max()) if len(bids) else np.nan,
            "bid_min_final": float(bids["finalPrice_num"].min()) if len(bids) else np.nan,
            "offer_par_positive_rows": int((offers["parAdjustedVolume_num"] > 0).sum()),
            "bid_par_positive_rows": int((bids["parAdjustedVolume_num"] > 0).sum()),
            "offer_so_rows": int((offers["soFlag"] == True).sum()),
            "bid_so_rows": int((bids["soFlag"] == True).sum()),
            "offer_cadl_rows": int((offers["cadlFlag"] == True).sum()),
            "bid_cadl_rows": int((bids["cadlFlag"] == True).sum()),
            "offer_final_price_not_null_rows": int(offers["finalPrice_num"].notna().sum()),
            "bid_final_price_not_null_rows": int(bids["finalPrice_num"].notna().sum()),
            "clean_offer_rows": int(len(offer_clean_sp)),
            "clean_bid_rows": int(len(bid_clean_sp)),
        }
        record["offer_so_share"] = record["offer_so_rows"] / record["offer_rows"] if record["offer_rows"] else np.nan
        record["bid_so_share"] = record["bid_so_rows"] / record["bid_rows"] if record["bid_rows"] else np.nan
        record["offer_cadl_share"] = record["offer_cadl_rows"] / record["offer_rows"] if record["offer_rows"] else np.nan
        record["bid_cadl_share"] = record["bid_cadl_rows"] / record["bid_rows"] if record["bid_rows"] else np.nan
        record["dominant_side_by_rows"] = (
            "offer" if record["offer_rows"] > record["bid_rows"] else "bid" if record["bid_rows"] > record["offer_rows"] else "tie"
        )
        record["dominant_side_by_abs_volume"] = (
            "offer"
            if record["offer_abs_volume_total"] > record["bid_abs_volume_total"]
            else "bid"
            if record["bid_abs_volume_total"] > record["offer_abs_volume_total"]
            else "tie"
        )
        record["highest_price_side"] = (
            "offer"
            if record["offer_max_final"] > record["bid_max_final"]
            else "bid"
            if record["bid_max_final"] > record["offer_max_final"]
            else "tie"
        )
        record["investigative_hypothesis"] = classify_hypothesis(pd.Series(record))
        records.append(record)

    out = pd.DataFrame(records).sort_values("settlementPeriod").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    out.to_parquet(OUT_PARQUET, index=False)

    print("Hypothesis summary:")
    print(out["investigative_hypothesis"].value_counts().to_string())
    print("\nDominant side by rows:")
    print(out["dominant_side_by_rows"].value_counts().to_string())
    print("\nHighest price side:")
    print(out["highest_price_side"].value_counts().to_string())
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {OUT_PARQUET}")


if __name__ == "__main__":
    main()
