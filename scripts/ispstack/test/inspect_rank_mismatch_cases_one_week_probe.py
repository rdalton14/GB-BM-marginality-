from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_test"
    / "ispstack_one_week_probe_accepted_actions"
    / "accepted_actions_ranked.parquet"
)
OUT_DIR = INPUT_PATH.parent
SUMMARY_OUT = OUT_DIR / "rank_mismatch_case_summary.json"


def print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    key_cols = ["settlementDate", "settlementPeriod", "direction"]

    grouped = (
        df.assign(rank_mismatch=df["rank_in_stack_seq"] != df["rank_in_stack_price"])
        .groupby(key_cols)
        .agg(
            n_rows=("sequenceNumber", "size"),
            n_mismatches=("rank_mismatch", "sum"),
            mismatch_share=("rank_mismatch", "mean"),
            n_sentinel=("has_sentinel_price", "sum"),
            min_price=("stack_price", "min"),
            max_price=("stack_price", "max"),
            min_volume=("volume", "min"),
            max_volume=("volume", "max"),
        )
        .reset_index()
    )

    mismatches = grouped[grouped["n_mismatches"] > 0].sort_values(
        ["mismatch_share", "n_mismatches", "n_rows"], ascending=[False, False, False]
    )
    sentinels = grouped[grouped["n_sentinel"] > 0].sort_values(
        ["n_sentinel", "n_rows"], ascending=[False, False]
    )
    bid_heavy = grouped[grouped["direction"] == "bid"].sort_values(
        ["n_rows", "mismatch_share"], ascending=[False, False]
    )

    print_section("Top Sequence-vs-Price Mismatch Groups")
    print(mismatches.head(15).to_string(index=False))

    print_section("Sentinel Price Groups")
    if sentinels.empty:
        print("No sentinel-price groups found.")
    else:
        print(sentinels.to_string(index=False))

    print_section("Largest Bid Groups")
    print(bid_heavy.head(15).to_string(index=False))

    case_groups = []
    seen = set()

    for source_df, label in [
        (mismatches.head(4), "mismatch"),
        (sentinels.head(4), "sentinel"),
        (bid_heavy.head(4), "bid_heavy"),
    ]:
        for _, row in source_df.iterrows():
            key = (row["settlementDate"], int(row["settlementPeriod"]), row["direction"])
            if key in seen:
                continue
            seen.add(key)
            case_groups.append((label, key))

    for label, (settlement_date, settlement_period, direction) in case_groups:
        subset = df[
            (df["settlementDate"] == settlement_date)
            & (df["settlementPeriod"] == settlement_period)
            & (df["direction"] == direction)
        ].sort_values("sequenceNumber")

        print_section(f"Case: {label} | {settlement_date} SP{settlement_period} {direction}")
        print(
            subset[
                [
                    "sequenceNumber",
                    "id",
                    "acceptanceId",
                    "soFlag",
                    "repricedIndicator",
                    "originalPrice",
                    "finalPrice",
                    "volume",
                    "abs_volume",
                    "rank_in_stack_seq",
                    "rank_in_stack_price",
                    "cum_abs_volume_seq",
                    "price_gap_prev_seq",
                    "price_gap_next_seq",
                    "has_sentinel_price",
                ]
            ]
            .head(40)
            .to_string(index=False)
        )

    summary = {
        "n_groups_total": int(len(grouped)),
        "n_groups_with_rank_mismatch": int((grouped["n_mismatches"] > 0).sum()),
        "n_groups_with_sentinel_price": int((grouped["n_sentinel"] > 0).sum()),
        "top_mismatch_groups": mismatches.head(15).to_dict(orient="records"),
        "sentinel_groups": sentinels.to_dict(orient="records"),
        "top_bid_groups": bid_heavy.head(15).to_dict(orient="records"),
    }

    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved summary -> {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
