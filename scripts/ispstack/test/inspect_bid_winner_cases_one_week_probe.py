from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
BASE_DIR = PROJECT_ROOT / "data" / "processed_test" / "ispstack_one_week_probe_accepted_actions"
ACTION_PATH = BASE_DIR / "marginal_action_sp.parquet"
RANKED_PATH = BASE_DIR / "accepted_actions_ranked.parquet"


def print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    action_sp = pd.read_parquet(ACTION_PATH)
    ranked = pd.read_parquet(RANKED_PATH)

    bid_winners = action_sp[action_sp["marginal_side_winner"] == "bid"].copy()

    print_section("Bid Winner Summary")
    print(f"Bid-winner settlement periods: {len(bid_winners)}")
    print(
        bid_winners[
            [
                "settlementDate",
                "settlementPeriod",
                "marginal_offer_price",
                "marginal_bid_price",
                "marginal_price_winner",
                "marginal_plant_winner",
                "winner_side_volume_total_energy",
                "winner_side_n_actions_energy",
                "winner_side_price_spread_energy",
                "offer_volume_total_energy",
                "bid_volume_total_energy",
            ]
        ].to_string(index=False)
    )

    for _, row in bid_winners.iterrows():
        settlement_date = row["settlementDate"]
        settlement_period = int(row["settlementPeriod"])

        print_section(f"Bid Winner Case: {settlement_date} SP{settlement_period}")
        print(
            row[
                [
                    "marginal_offer_id",
                    "marginal_offer_plant",
                    "marginal_offer_price",
                    "marginal_offer_volume",
                    "marginal_offer_sequenceNumber",
                    "marginal_bid_id",
                    "marginal_bid_plant",
                    "marginal_bid_price",
                    "marginal_bid_volume",
                    "marginal_bid_sequenceNumber",
                    "marginal_side_winner",
                    "marginal_id_winner",
                    "marginal_plant_winner",
                    "marginal_price_winner",
                    "winner_side_volume_total_energy",
                    "winner_side_n_actions_energy",
                    "winner_side_n_plants_energy",
                    "winner_side_price_spread_energy",
                ]
            ].to_string()
        )

        subset = ranked[
            (ranked["settlementDate"] == settlement_date)
            & (ranked["settlementPeriod"] == settlement_period)
        ].copy()

        offers = subset[subset["direction"] == "offer"].sort_values("stack_price", ascending=False).head(15)
        bids = subset[subset["direction"] == "bid"].sort_values("stack_price", ascending=False).head(15)

        print("\nTop offers by price:")
        print(
            offers[
                [
                    "direction",
                    "id",
                    "plant_id_candidate",
                    "stack_price",
                    "volume",
                    "abs_volume",
                    "sequenceNumber",
                    "rank_in_stack_seq",
                    "rank_in_stack_price",
                    "repricedIndicator",
                    "has_sentinel_price",
                ]
            ].to_string(index=False)
        )

        print("\nTop bids by price:")
        print(
            bids[
                [
                    "direction",
                    "id",
                    "plant_id_candidate",
                    "stack_price",
                    "volume",
                    "abs_volume",
                    "sequenceNumber",
                    "rank_in_stack_seq",
                    "rank_in_stack_price",
                    "repricedIndicator",
                    "has_sentinel_price",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
