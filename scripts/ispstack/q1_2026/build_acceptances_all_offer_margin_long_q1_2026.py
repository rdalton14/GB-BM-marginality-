from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
IN_PATH = PROJECT_ROOT / "data" / "processed" / "acceptances_all_q1_2026_long.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "q1_2026" / "acceptances_all_marginal_action_q1_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "acceptances_all_offer_margin_long_q1_2026.csv"
OUT_PARQUET = OUT_DIR / "acceptances_all_offer_margin_long_q1_2026.parquet"
SUMMARY_JSON = OUT_DIR / "acceptances_all_offer_margin_long_q1_2026_summary.json"

GROUP_KEYS = ["settlementDate", "settlementPeriod"]
SENTINEL_PRICES = {-99999.0, -9999.0, -999.0, 9999.0, 99999.0}


def load_df() -> pd.DataFrame:
    df = pd.read_parquet(IN_PATH)
    df["settlementDate"] = df["settlementDate"].astype("string")
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    df["offerPrice_num"] = pd.to_numeric(df["offerPrice"], errors="coerce")
    df["is_offer_sentinel"] = df["offerPrice_num"].isin(SENTINEL_PRICES)
    return df


def build_offer_margin_long(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["offerPrice_num"].notna() & ~df["is_offer_sentinel"]].copy()
    if valid.empty:
        return valid

    max_offer = valid.groupby(GROUP_KEYS)["offerPrice_num"].transform("max")
    tied = valid[valid["offerPrice_num"].eq(max_offer)].copy()

    tied["offer_tie_group_size"] = tied.groupby(GROUP_KEYS)["bmUnit"].transform("size").astype("Int64")
    tied["offer_has_tie"] = tied["offer_tie_group_size"].gt(1)
    tied["offer_margin_price"] = tied["offerPrice_num"]

    keep_cols = [
        "settlementDate",
        "settlementPeriod",
        "bmUnit",
        "nationalGridBmUnit",
        "acceptanceNumber",
        "acceptanceTime",
        "bidOfferPairId",
        "bidPrice",
        "offerPrice",
        "offerPrice_num",
        "offer_margin_price",
        "offer_tie_group_size",
        "offer_has_tie",
        "source_endpoint",
        "pull_timestamp",
    ]
    keep_cols = [c for c in keep_cols if c in tied.columns]
    out = tied[keep_cols].copy()
    out = out.sort_values(["settlementDate", "settlementPeriod", "bmUnit", "acceptanceNumber"], kind="stable")
    return out


def build_summary(df: pd.DataFrame) -> dict:
    group_sizes = df.groupby(GROUP_KEYS).size() if not df.empty else pd.Series(dtype="int64")
    return {
        "n_rows": int(len(df)),
        "n_settlement_periods": int(group_sizes.shape[0]),
        "n_unique_bm_units": int(df["bmUnit"].nunique(dropna=True)) if "bmUnit" in df.columns else 0,
        "n_tie_settlement_periods": int(group_sizes.gt(1).sum()),
        "max_tie_group_size": int(group_sizes.max()) if not group_sizes.empty else 0,
        "tie_group_size_distribution": {str(int(k)): int(v) for k, v in group_sizes.value_counts().sort_index().items()},
        "top_offer_margin_bmus": (
            df["bmUnit"].fillna("MISSING").value_counts().head(25).to_dict()
            if "bmUnit" in df.columns
            else {}
        ),
    }


def main() -> None:
    df = load_df()
    offer_margin_long = build_offer_margin_long(df)

    offer_margin_long.to_csv(OUT_CSV, index=False)
    offer_margin_long.to_parquet(OUT_PARQUET, index=False)

    summary = build_summary(offer_margin_long)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("ACCEPTANCES ALL OFFER-SIDE MARGIN LONG TABLE (Q1 2026)")
    print("=" * 72)
    print(f"Rows                         : {summary['n_rows']:,}")
    print(f"Settlement periods           : {summary['n_settlement_periods']:,}")
    print(f"Tie settlement periods       : {summary['n_tie_settlement_periods']:,}")
    print(f"Max tie group size           : {summary['max_tie_group_size']:,}")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
