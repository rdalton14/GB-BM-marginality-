from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
IN_PATH = PROJECT_ROOT / "data" / "processed" / "acceptances_all_q1_2026_long.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "q1_2026" / "acceptances_all_marginal_action_q1_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "acceptances_all_marginal_bid_offer_sp_q1_2026.csv"
OUT_PARQUET = OUT_DIR / "acceptances_all_marginal_bid_offer_sp_q1_2026.parquet"
SUMMARY_JSON = OUT_DIR / "acceptances_all_marginal_bid_offer_sp_q1_2026_summary.json"

GROUP_KEYS = ["settlementDate", "settlementPeriod"]
SENTINEL_PRICES = {-99999.0, -9999.0, -999.0, 9999.0, 99999.0}


def _join_unique(values: pd.Series) -> str | None:
    cleaned = [str(v) for v in values.dropna().astype(str).tolist() if str(v) != ""]
    unique = list(dict.fromkeys(cleaned))
    if not unique:
        return None
    return "; ".join(unique)


def _scalar_if_single(values: pd.Series) -> Any:
    cleaned = [v for v in values.dropna().tolist()]
    unique = list(dict.fromkeys(cleaned))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        return None
    return None


def _top_side(grp: pd.DataFrame, price_col: str, side_name: str) -> dict[str, Any]:
    side = grp.copy()
    side["accepted_price"] = pd.to_numeric(side[price_col], errors="coerce")
    side["is_sentinel_price"] = side["accepted_price"].isin(SENTINEL_PRICES)

    record: dict[str, Any] = {
        f"{side_name}_n_actions_raw": int(side["accepted_price"].notna().sum()),
        f"{side_name}_n_sentinel_actions": int(side["is_sentinel_price"].sum()),
        f"{side_name}_price_col": price_col,
    }

    side = side[side["accepted_price"].notna() & ~side["is_sentinel_price"]].copy()

    record[f"{side_name}_n_actions"] = int(len(side))

    if side.empty:
        record.update(
            {
                f"marginal_{side_name}_price": None,
                f"marginal_{side_name}_identifier": None,
                f"marginal_{side_name}_national_grid_bmu": None,
                f"marginal_{side_name}_acceptance_number": None,
                f"marginal_{side_name}_acceptance_time": None,
                f"marginal_{side_name}_bid_offer_pair_id": None,
                f"marginal_{side_name}_n_tied_actions": 0,
                f"marginal_{side_name}_has_tie": False,
            }
        )
        return record

    max_price = float(side["accepted_price"].max())
    tied = side[side["accepted_price"].eq(max_price)].copy()

    record.update(
        {
            f"marginal_{side_name}_price": max_price,
            f"marginal_{side_name}_identifier": _join_unique(tied["bmUnit"]) if "bmUnit" in tied.columns else None,
            f"marginal_{side_name}_national_grid_bmu": _join_unique(tied["nationalGridBmUnit"]) if "nationalGridBmUnit" in tied.columns else None,
            f"marginal_{side_name}_acceptance_number": _join_unique(tied["acceptanceNumber"]) if "acceptanceNumber" in tied.columns else None,
            f"marginal_{side_name}_acceptance_time": _join_unique(tied["acceptanceTime"]) if "acceptanceTime" in tied.columns else None,
            f"marginal_{side_name}_bid_offer_pair_id": _join_unique(tied["bidOfferPairId"]) if "bidOfferPairId" in tied.columns else None,
            f"marginal_{side_name}_n_tied_actions": int(len(tied)),
            f"marginal_{side_name}_has_tie": bool(len(tied) > 1),
        }
    )
    return record


def build_sp_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for (settlement_date, settlement_period), grp in df.groupby(GROUP_KEYS, sort=True):
        record: dict[str, Any] = {
            "settlementDate": settlement_date,
            "settlementPeriod": int(settlement_period),
            "n_acceptance_rows": int(len(grp)),
        }
        record.update(_top_side(grp, "offerPrice", "offer"))
        record.update(_top_side(grp, "bidPrice", "bid"))
        records.append(record)

    return pd.DataFrame.from_records(records)


def build_summary(sp_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_settlement_periods": int(len(sp_df)),
        "offer_side_available_count": int(sp_df["marginal_offer_price"].notna().sum()),
        "bid_side_available_count": int(sp_df["marginal_bid_price"].notna().sum()),
        "offer_side_total_sentinel_actions": int(sp_df["offer_n_sentinel_actions"].sum()),
        "bid_side_total_sentinel_actions": int(sp_df["bid_n_sentinel_actions"].sum()),
        "offer_side_tie_count": int(sp_df["marginal_offer_has_tie"].fillna(False).sum()),
        "bid_side_tie_count": int(sp_df["marginal_bid_has_tie"].fillna(False).sum()),
        "top_offer_identifiers": (
            sp_df["marginal_offer_identifier"]
            .fillna("MISSING")
            .value_counts()
            .head(20)
            .to_dict()
        ),
        "top_bid_identifiers": (
            sp_df["marginal_bid_identifier"]
            .fillna("MISSING")
            .value_counts()
            .head(20)
            .to_dict()
        ),
    }


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    sp_df = build_sp_table(df)

    sp_df.to_csv(OUT_CSV, index=False)
    sp_df.to_parquet(OUT_PARQUET, index=False)

    summary = build_summary(sp_df)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("ACCEPTANCES ALL MARGINAL BID/OFFER SP TABLE (Q1 2026)")
    print("=" * 72)
    print(f"Settlement periods          : {len(sp_df):,}")
    print(f"Offer-side marginals found  : {summary['offer_side_available_count']:,}")
    print(f"Bid-side marginals found    : {summary['bid_side_available_count']:,}")
    print(f"Offer sentinel actions drop : {summary['offer_side_total_sentinel_actions']:,}")
    print(f"Bid sentinel actions drop   : {summary['bid_side_total_sentinel_actions']:,}")
    print(f"Offer-side tie SPs          : {summary['offer_side_tie_count']:,}")
    print(f"Bid-side tie SPs            : {summary['bid_side_tie_count']:,}")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
