from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/raw_test/ispstack")
ISPSTACK_DIR = ROOT / "ispstack_one_week_probe_2026_01_05_to_2026_01_11"
ACCEPTANCES_DIR = ROOT / "acceptances_all_probe_2026_01_05_to_2026_01_11"
OUT_DIR = Path("data/diagnostics/audits/acceptances_all_ispstack_join_audit_2026_01_05_to_2026_01_11")

ISPSTACK_PATH = ISPSTACK_DIR / "ispstack_one_week_long.parquet"
ACCEPTANCES_PATH = ACCEPTANCES_DIR / "acceptances_all_long_2026_01_05_2026_01_11.parquet"

SUMMARY_JSON = OUT_DIR / "acceptances_all_ispstack_join_summary.json"
DIRECT_MATCH_AUDIT_CSV = OUT_DIR / "acceptances_all_ispstack_direct_match_audit.csv"
NUMERIC_RESCUE_AUDIT_CSV = OUT_DIR / "acceptances_all_ispstack_numeric_rescue_audit.csv"
NUMERIC_UNIQUE_MATCHES_CSV = OUT_DIR / "acceptances_all_ispstack_numeric_unique_price_matches.csv"


def is_numeric_only(series: pd.Series) -> pd.Series:
    return series.astype(str).str.fullmatch(r"\d+").fillna(False)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    isp = pd.read_parquet(ISPSTACK_PATH)
    acc = pd.read_parquet(ACCEPTANCES_PATH)

    isp["acceptanceId_int"] = pd.to_numeric(isp["acceptanceId"], errors="coerce").astype("Int64")
    acc["acceptanceNumber_int"] = pd.to_numeric(acc["acceptanceNumber"], errors="coerce").astype("Int64")

    isp["is_numeric_id"] = is_numeric_only(isp["id"])
    isp["finalPrice_num"] = pd.to_numeric(isp["finalPrice"], errors="coerce")
    acc["offerPrice_num"] = pd.to_numeric(acc["offerPrice"], errors="coerce")
    acc["bidPrice_num"] = pd.to_numeric(acc["bidPrice"], errors="coerce")

    return isp, acc


def build_direct_match_audit(isp: pd.DataFrame, acc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = isp.merge(
        acc,
        left_on=["settlementDate", "settlementPeriod", "acceptanceId_int"],
        right_on=["settlementDate", "settlementPeriod", "acceptanceNumber_int"],
        how="left",
        indicator=True,
        suffixes=("_isp", "_acc"),
    )

    matched = joined[joined["_merge"] == "both"].copy()
    joined["offer_price_match"] = np.isclose(
        joined["finalPrice_num"],
        joined["offerPrice_num"],
        equal_nan=False,
    )
    joined["bid_price_match"] = np.isclose(
        joined["finalPrice_num"],
        joined["bidPrice_num"],
        equal_nan=False,
    )

    matched["offer_price_match"] = np.isclose(
        matched["finalPrice_num"],
        matched["offerPrice_num"],
        equal_nan=False,
    )
    matched["bid_price_match"] = np.isclose(
        matched["finalPrice_num"],
        matched["bidPrice_num"],
        equal_nan=False,
    )

    direct_summary = {
        "isp_rows": int(len(isp)),
        "acceptances_rows": int(len(acc)),
        "direct_match_rows": int((joined["_merge"] == "both").sum()),
        "direct_match_share_of_isp_rows": float((joined["_merge"] == "both").mean()),
        "isp_unique_acceptance_keys": int(
            isp[["settlementDate", "settlementPeriod", "acceptanceId_int"]]
            .dropna()
            .drop_duplicates()
            .shape[0]
        ),
        "acceptances_unique_acceptance_keys": int(
            acc[["settlementDate", "settlementPeriod", "acceptanceNumber_int"]]
            .dropna()
            .drop_duplicates()
            .shape[0]
        ),
        "acceptances_duplicate_acceptance_keys": int(
            acc.groupby(["settlementDate", "settlementPeriod", "acceptanceNumber_int"])
            .size()
            .gt(1)
            .sum()
        ),
        "direct_match_numeric_id_rows": int(
            ((joined["is_numeric_id"]) & (joined["_merge"] == "both")).sum()
        ),
        "direct_match_offer_price_eq_offer_price_share": float(
            matched.loc[matched["direction"] == "offer", "offer_price_match"].mean()
        ),
        "direct_match_offer_price_eq_bid_price_share": float(
            matched.loc[matched["direction"] == "offer", "bid_price_match"].mean()
        ),
        "direct_match_bid_price_eq_bid_price_share": float(
            matched.loc[matched["direction"] == "bid", "bid_price_match"].mean()
        ),
        "direct_match_bid_price_eq_offer_price_share": float(
            matched.loc[matched["direction"] == "bid", "offer_price_match"].mean()
        ),
        "direct_match_offer_pair_positive_share": float(
            (pd.to_numeric(matched.loc[matched["direction"] == "offer", "bidOfferPairId_acc"], errors="coerce") > 0).mean()
        ),
        "direct_match_bid_pair_negative_share": float(
            (pd.to_numeric(matched.loc[matched["direction"] == "bid", "bidOfferPairId_acc"], errors="coerce") < 0).mean()
        ),
    }

    out_cols = [
        "settlementDate",
        "settlementPeriod",
        "direction",
        "id",
        "is_numeric_id",
        "acceptanceId_int",
        "bmUnit",
        "acceptanceNumber_int",
        "bidOfferPairId_isp",
        "bidOfferPairId_acc",
        "finalPrice",
        "offerPrice",
        "bidPrice",
        "offer_price_match",
        "bid_price_match",
        "_merge",
    ]
    return joined[out_cols].copy(), direct_summary


def build_numeric_rescue_audit(isp: pd.DataFrame, acc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    numeric = isp[isp["is_numeric_id"]].copy()
    same_sp = numeric.merge(
        acc,
        on=["settlementDate", "settlementPeriod"],
        how="left",
        suffixes=("_isp", "_acc"),
    )

    same_sp["candidate_price"] = np.where(
        same_sp["direction"].eq("offer"),
        same_sp["offerPrice_num"],
        same_sp["bidPrice_num"],
    )
    same_sp["exact_price_match"] = np.isclose(
        same_sp["finalPrice_num"],
        same_sp["candidate_price"],
        equal_nan=False,
    )

    group_cols = ["settlementDate", "settlementPeriod", "direction", "id", "finalPrice_num"]
    grouped = (
        same_sp.groupby(group_cols)
        .agg(
            exact_price_match_count=("exact_price_match", "sum"),
            candidate_bmu_count=("bmUnit", "nunique"),
            example_candidate_bmus=("bmUnit", lambda s: "; ".join(sorted(pd.Series(s.dropna().astype(str).unique()).tolist())[:10])),
        )
        .reset_index()
        .rename(columns={"finalPrice_num": "finalPrice"})
    )
    grouped["uniquely_rescuable_by_exact_price"] = grouped["exact_price_match_count"].eq(1)
    grouped["multi_match_ambiguous"] = grouped["exact_price_match_count"].gt(1)
    grouped["no_exact_match"] = grouped["exact_price_match_count"].eq(0)

    unique_matches = same_sp[same_sp["exact_price_match"]].copy()
    unique_group_cols = ["settlementDate", "settlementPeriod", "direction", "id", "finalPrice"]
    unique_keys = grouped[grouped["uniquely_rescuable_by_exact_price"]][unique_group_cols].copy()
    unique_matches = unique_matches.drop(columns=["finalPrice"], errors="ignore").rename(
        columns={"finalPrice_num": "finalPrice"}
    )
    unique_matches = unique_matches.merge(unique_keys, on=unique_group_cols, how="inner")
    unique_matches = unique_matches[
        [
            "settlementDate",
            "settlementPeriod",
            "direction",
            "id",
            "finalPrice",
            "bmUnit",
            "acceptanceNumber_int",
            "acceptanceTime",
            "bidOfferPairId_acc",
            "offerPrice",
            "bidPrice",
        ]
    ].rename(columns={"bidOfferPairId_acc": "bidOfferPairId"})

    numeric_summary = {
        "numeric_isp_rows": int(len(numeric)),
        "numeric_isp_row_share": float(len(numeric) / len(isp)),
        "numeric_groups_tested": int(len(grouped)),
        "numeric_groups_with_any_exact_price_match": int(grouped["exact_price_match_count"].gt(0).sum()),
        "numeric_groups_uniquely_rescuable_by_exact_price": int(grouped["uniquely_rescuable_by_exact_price"].sum()),
        "numeric_groups_ambiguous_exact_price_match": int(grouped["multi_match_ambiguous"].sum()),
        "numeric_groups_with_no_exact_price_match": int(grouped["no_exact_match"].sum()),
        "numeric_unique_exact_price_rescue_share": float(grouped["uniquely_rescuable_by_exact_price"].mean()),
    }

    return grouped, unique_matches, numeric_summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    isp, acc = load_inputs()

    direct_audit, direct_summary = build_direct_match_audit(isp, acc)
    numeric_audit, unique_numeric_matches, numeric_summary = build_numeric_rescue_audit(isp, acc)

    direct_audit.to_csv(DIRECT_MATCH_AUDIT_CSV, index=False)
    numeric_audit.to_csv(NUMERIC_RESCUE_AUDIT_CSV, index=False)
    unique_numeric_matches.to_csv(NUMERIC_UNIQUE_MATCHES_CSV, index=False)

    summary = {
        "one_week_window": {"start_date": "2026-01-05", "end_date": "2026-01-11"},
        "direct_linkage": direct_summary,
        "numeric_rescue": numeric_summary,
        "conclusion": {
            "direct_join_on_acceptance_number_is_strong": bool(
                direct_summary["direct_match_share_of_isp_rows"] > 0.9
            ),
            "numeric_ids_rescued_by_direct_join": bool(
                direct_summary["direct_match_numeric_id_rows"] > 0
            ),
            "numeric_ids_reliably_rescued_by_exact_price_only": bool(
                numeric_summary["numeric_unique_exact_price_rescue_share"] > 0.5
            ),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print("ACCEPTANCES ALL x ISPSTACK JOIN AUDIT")
    print("=" * 72)
    print(f"Direct acceptanceId/acceptanceNumber match share : {direct_summary['direct_match_share_of_isp_rows']:.4%}")
    print(f"Direct matched numeric-id ISPSTACK rows          : {direct_summary['direct_match_numeric_id_rows']:,}")
    print(f"Offer rows with finalPrice == offerPrice         : {direct_summary['direct_match_offer_price_eq_offer_price_share']:.2%}")
    print(f"Bid rows with finalPrice == bidPrice             : {direct_summary['direct_match_bid_price_eq_bid_price_share']:.2%}")
    print(f"Numeric groups uniquely rescuable by price only  : {numeric_summary['numeric_groups_uniquely_rescuable_by_exact_price']:,} / {numeric_summary['numeric_groups_tested']:,}")
    print(f"Numeric groups ambiguous on exact price          : {numeric_summary['numeric_groups_ambiguous_exact_price_match']:,}")
    print(f"Numeric groups with no exact price match         : {numeric_summary['numeric_groups_with_no_exact_price_match']:,}")
    print(f"\nSummary JSON: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
