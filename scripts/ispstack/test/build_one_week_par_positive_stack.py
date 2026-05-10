from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
IN_PATH = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / "ispstack_one_week_probe_2026_01_05_to_2026_01_11" / "ispstack_one_week_long.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed_test" / "ispstack_one_week_par_positive_stack"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG_CSV = OUT_DIR / "ispstack_one_week_par_positive_stack.csv"
OUT_LONG_PARQUET = OUT_DIR / "ispstack_one_week_par_positive_stack.parquet"
OUT_OFFER_CSV = OUT_DIR / "ispstack_one_week_par_positive_offer_stack.csv"
OUT_BID_CSV = OUT_DIR / "ispstack_one_week_par_positive_bid_stack.csv"
SUMMARY_JSON = OUT_DIR / "ispstack_one_week_par_positive_stack_summary.json"


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    df["parAdjustedVolume"] = pd.to_numeric(df["parAdjustedVolume"], errors="coerce")
    df["originalPrice"] = pd.to_numeric(df["originalPrice"], errors="coerce")
    df["finalPrice"] = pd.to_numeric(df["finalPrice"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    df["sequenceNumber"] = pd.to_numeric(df["sequenceNumber"], errors="coerce").astype("Int64")

    par = df[df["parAdjustedVolume"].gt(0)].copy()
    par["abs_volume"] = par["volume"].abs()

    sort_cols = ["settlementDate", "settlementPeriod", "direction", "sequenceNumber", "finalPrice", "id"]
    sort_cols = [c for c in sort_cols if c in par.columns]
    par = par.sort_values(sort_cols, kind="stable")

    par.to_csv(OUT_LONG_CSV, index=False)
    par.to_parquet(OUT_LONG_PARQUET, index=False)

    offers = par[par["direction"].astype(str).eq("offer")].copy()
    bids = par[par["direction"].astype(str).eq("bid")].copy()
    offers.to_csv(OUT_OFFER_CSV, index=False)
    bids.to_csv(OUT_BID_CSV, index=False)

    by_sp = (
        par.groupby(["settlementDate", "settlementPeriod", "direction"])
        .size()
        .reset_index(name="n_rows")
    )

    summary = {
        "source": str(IN_PATH),
        "filter_rule": "parAdjustedVolume > 0",
        "n_rows_total": int(len(df)),
        "n_rows_par_positive": int(len(par)),
        "share_rows_par_positive": float(len(par) / len(df)) if len(df) else None,
        "n_offer_rows_par_positive": int(len(offers)),
        "n_bid_rows_par_positive": int(len(bids)),
        "n_sps_with_any_par_positive": int(
            par[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
        ),
        "top_sp_direction_counts": by_sp.sort_values("n_rows", ascending=False).head(20).to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("ONE-WEEK PAR-POSITIVE STACK")
    print("=" * 72)
    print(f"Rows total              : {len(df):,}")
    print(f"Rows with PAR > 0       : {len(par):,}")
    print(f"Offer rows PAR > 0      : {len(offers):,}")
    print(f"Bid rows PAR > 0        : {len(bids):,}")
    print(f"SPs with any PAR > 0    : {summary['n_sps_with_any_par_positive']:,}")
    print(f"\nSaved: {OUT_LONG_CSV}")


if __name__ == "__main__":
    main()
