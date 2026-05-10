from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "q1_2026" / "ispstack_marginal_action_q1_2026"
RAW_PATH = BASE_DIR / "accepted_actions_long_raw_q1_2026.parquet"
RANKED_PATH = BASE_DIR / "accepted_actions_ranked_q1_2026.parquet"
ACTION_RESOLVED_PATH = BASE_DIR / "marginal_action_sp_q1_2026_resolved.parquet"
REVIEW_QUEUE_PATH = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_id_resolution_q1_2026_review_queue.csv"

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "q1_numeric_id_review_pack"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "numeric_id_review_summary.csv"
WINNER_CASES_PATH = OUT_DIR / "numeric_id_winner_cases.csv"
STACK_CONTEXT_PATH = OUT_DIR / "numeric_id_stack_context.csv"
PAIR_CONTEXT_PATH = OUT_DIR / "numeric_id_bid_offer_pair_context.csv"
TOP_IDS_JSON = OUT_DIR / "numeric_id_review_pack_summary.json"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(RAW_PATH)
    ranked = pd.read_parquet(RANKED_PATH)
    action = pd.read_parquet(ACTION_RESOLVED_PATH)
    review = pd.read_csv(REVIEW_QUEUE_PATH)
    return raw, ranked, action, review


def main() -> None:
    raw, ranked, action, review = load_inputs()

    numeric_ids = review.loc[review["id_type"] == "numeric_only", "raw_id"].astype(str).tolist()
    top_numeric_ids = (
        review.loc[review["id_type"] == "numeric_only"]
        .sort_values(["winner_sp_count", "raw_row_count"], ascending=False)
        .head(25)["raw_id"]
        .astype(str)
        .tolist()
    )

    raw["id"] = raw["id"].astype(str)
    ranked["id"] = ranked["id"].astype(str)

    summary = (
        review.loc[review["id_type"] == "numeric_only"]
        .sort_values(["winner_sp_count", "raw_row_count"], ascending=False)
        .copy()
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    winner_cases = action[
        action["marginal_id_winner"].astype(str).isin(top_numeric_ids)
    ][
        [
            "settlementDate",
            "settlementPeriod",
            "marginal_side_winner",
            "marginal_id_winner",
            "marginal_price_winner",
            "marginal_volume_winner",
            "winner_side_n_actions_energy",
            "winner_side_price_spread_energy",
            "marginal_offer_id",
            "marginal_offer_price",
            "marginal_bid_id",
            "marginal_bid_price",
        ]
    ].sort_values(["marginal_id_winner", "settlementDate", "settlementPeriod"])
    winner_cases.to_csv(WINNER_CASES_PATH, index=False)

    # Pull nearby stack rows for the winning SPs so manual review can see surrounding named units.
    contexts: list[pd.DataFrame] = []
    for _, row in winner_cases.iterrows():
        settlement_date = row["settlementDate"]
        settlement_period = int(row["settlementPeriod"])
        raw_id = str(row["marginal_id_winner"])
        side = row["marginal_side_winner"]

        subset = ranked[
            (ranked["settlementDate"] == settlement_date)
            & (ranked["settlementPeriod"] == settlement_period)
            & (ranked["direction"] == side)
        ].copy()

        if subset.empty:
            continue

        target_rows = subset[subset["id"] == raw_id]
        if target_rows.empty:
            continue

        min_seq = int(target_rows["sequenceNumber"].min())
        max_seq = int(target_rows["sequenceNumber"].max())

        window = subset[
            subset["sequenceNumber"].between(max(1, min_seq - 5), max_seq + 5)
        ][
            [
                "settlementDate",
                "settlementPeriod",
                "direction",
                "sequenceNumber",
                "id",
                "acceptanceId",
                "bidOfferPairId",
                "stack_price",
                "volume",
                "abs_volume",
                "repricedIndicator",
                "soFlag",
            ]
        ].copy()
        window.insert(0, "focus_numeric_id", raw_id)
        contexts.append(window)

    stack_context = pd.concat(contexts, ignore_index=True) if contexts else pd.DataFrame()
    stack_context.to_csv(STACK_CONTEXT_PATH, index=False)

    # Show whether numeric ids share pair ids with named rows anywhere in Q1.
    numeric_rows = raw[raw["id"].isin(top_numeric_ids)].copy()
    pair_ids = numeric_rows["bidOfferPairId"].dropna().unique().tolist()
    pair_context = raw[
        raw["bidOfferPairId"].isin(pair_ids)
    ][
        [
            "settlementDate",
            "settlementPeriod",
            "direction",
            "id",
            "acceptanceId",
            "bidOfferPairId",
            "originalPrice",
            "finalPrice",
            "volume",
            "soFlag",
        ]
    ].sort_values(["bidOfferPairId", "settlementDate", "settlementPeriod", "direction", "id"])
    pair_context.to_csv(PAIR_CONTEXT_PATH, index=False)

    summary_json = {
        "n_numeric_ids_total": len(numeric_ids),
        "n_top_numeric_ids_in_pack": len(top_numeric_ids),
        "winner_cases_rows": int(len(winner_cases)),
        "stack_context_rows": int(len(stack_context)),
        "pair_context_rows": int(len(pair_context)),
        "top_numeric_ids": top_numeric_ids,
    }
    with TOP_IDS_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("=" * 72)
    print("Q1 Numeric ID Manual Review Pack Built")
    print("=" * 72)
    print(f"Summary      : {SUMMARY_PATH}")
    print(f"Winner cases : {WINNER_CASES_PATH}")
    print(f"Stack ctx    : {STACK_CONTEXT_PATH}")
    print(f"Pair ctx     : {PAIR_CONTEXT_PATH}")
    print(f"Meta json    : {TOP_IDS_JSON}")


if __name__ == "__main__":
    main()
