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

CANDIDATES_OUT = OUT_DIR / "marginal_candidates_sp.parquet"
ACTION_OUT = OUT_DIR / "marginal_action_sp.parquet"
SUMMARY_OUT = OUT_DIR / "marginal_candidates_summary.json"


GROUP_KEYS = ["settlementDate", "settlementPeriod"]


def load_ranked() -> pd.DataFrame:
    return pd.read_parquet(INPUT_PATH)


def build_stack_summary(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []

    for (settlement_date, settlement_period), grp in df.groupby(GROUP_KEYS, sort=True):
        offers = grp[grp["direction"] == "offer"]
        bids = grp[grp["direction"] == "bid"]

        records.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": int(settlement_period),
                "offer_volume_total_energy": float(offers["abs_volume"].sum()),
                "bid_volume_total_energy": float(bids["abs_volume"].sum()),
                "gross_accepted_volume_energy": float(grp["abs_volume"].sum()),
                "offer_n_actions_energy": int(len(offers)),
                "bid_n_actions_energy": int(len(bids)),
                "offer_n_plants_energy": int(offers["plant_id_candidate"].nunique(dropna=True)),
                "bid_n_plants_energy": int(bids["plant_id_candidate"].nunique(dropna=True)),
                "offer_min_price_energy": float(offers["stack_price"].min()) if not offers.empty else None,
                "offer_max_price_energy": float(offers["stack_price"].max()) if not offers.empty else None,
                "offer_price_spread_energy": (
                    float(offers["stack_price"].max() - offers["stack_price"].min()) if not offers.empty else None
                ),
                "bid_min_price_energy": float(bids["stack_price"].min()) if not bids.empty else None,
                "bid_max_price_energy": float(bids["stack_price"].max()) if not bids.empty else None,
                "bid_price_spread_energy": (
                    float(bids["stack_price"].max() - bids["stack_price"].min()) if not bids.empty else None
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def select_candidate(side_df: pd.DataFrame, side_name: str) -> pd.Series | None:
    if side_df.empty:
        return None

    valid = side_df[
        side_df["stack_price_valid"]
        & side_df["stack_price"].notna()
        & side_df["volume"].notna()
    ].copy()

    if valid.empty:
        return None

    valid = valid.sort_values(
        by=["stack_price", "abs_volume", "sequenceNumber"],
        ascending=[False, False, False],
        kind="stable",
    )
    top = valid.iloc[0]

    return pd.Series(
        {
            f"marginal_{side_name}_id": top["id"],
            f"marginal_{side_name}_plant": top["plant_id_candidate"],
            f"marginal_{side_name}_price": float(top["stack_price"]),
            f"marginal_{side_name}_volume": float(top["volume"]),
            f"marginal_{side_name}_abs_volume": float(top["abs_volume"]),
            f"marginal_{side_name}_sequenceNumber": int(top["sequenceNumber"]),
            f"marginal_{side_name}_rank_in_stack_seq": int(top["rank_in_stack_seq"]),
            f"marginal_{side_name}_rank_in_stack_price": int(top["rank_in_stack_price"]),
            f"marginal_{side_name}_price_gap_prev": (
                float(top["price_gap_prev_seq"]) if pd.notna(top["price_gap_prev_seq"]) else None
            ),
            f"marginal_{side_name}_price_gap_next": (
                float(top["price_gap_next_seq"]) if pd.notna(top["price_gap_next_seq"]) else None
            ),
            f"marginal_{side_name}_repricedIndicator": bool(top["repricedIndicator"]),
            f"marginal_{side_name}_soFlag": bool(top["soFlag"]),
        }
    )


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    stack_summary = build_stack_summary(df)
    records: list[dict] = []

    for (settlement_date, settlement_period), grp in df.groupby(GROUP_KEYS, sort=True):
        offers = grp[grp["direction"] == "offer"]
        bids = grp[grp["direction"] == "bid"]

        record: dict = {
            "settlementDate": settlement_date,
            "settlementPeriod": int(settlement_period),
        }

        offer_candidate = select_candidate(offers, "offer")
        bid_candidate = select_candidate(bids, "bid")

        if offer_candidate is not None:
            record.update(offer_candidate.to_dict())
        if bid_candidate is not None:
            record.update(bid_candidate.to_dict())

        has_offer = offer_candidate is not None
        has_bid = bid_candidate is not None

        if has_offer and not has_bid:
            marginal_side = "offer"
        elif has_bid and not has_offer:
            marginal_side = "bid"
        elif has_offer and has_bid:
            marginal_side = "mixed_or_ambiguous"
        else:
            marginal_side = "missing"

        record["marginal_side"] = marginal_side

        offer_price = record.get("marginal_offer_price")
        bid_price = record.get("marginal_bid_price")

        if pd.notna(offer_price) and pd.isna(bid_price):
            marginal_side_winner = "offer"
        elif pd.notna(bid_price) and pd.isna(offer_price):
            marginal_side_winner = "bid"
        elif pd.notna(offer_price) and pd.notna(bid_price):
            if float(offer_price) > float(bid_price):
                marginal_side_winner = "offer"
            elif float(bid_price) > float(offer_price):
                marginal_side_winner = "bid"
            else:
                marginal_side_winner = "mixed_or_ambiguous"
        else:
            marginal_side_winner = "missing"

        record["marginal_side_winner"] = marginal_side_winner
        records.append(record)

    candidates = pd.DataFrame.from_records(records)
    return candidates.merge(stack_summary, on=GROUP_KEYS, how="left", validate="one_to_one")


def build_marginal_action_sp(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()

    selected_fields = {
        "marginal_id": [],
        "marginal_plant": [],
        "marginal_price": [],
        "marginal_volume": [],
        "marginal_abs_volume": [],
        "marginal_sequenceNumber": [],
        "marginal_rank_in_stack_seq": [],
        "marginal_rank_in_stack_price": [],
        "marginal_price_gap_prev": [],
        "marginal_price_gap_next": [],
    }

    for _, row in out.iterrows():
        side = row["marginal_side"]

        if side == "offer":
            prefix = "marginal_offer_"
        elif side == "bid":
            prefix = "marginal_bid_"
        else:
            prefix = None

        for field in selected_fields:
            if prefix is None:
                selected_fields[field].append(None)
            else:
                source_col = prefix + field.replace("marginal_", "")
                selected_fields[field].append(row.get(source_col))

    for field, values in selected_fields.items():
        out[field] = values

    winner_fields = {
        "marginal_id_winner": [],
        "marginal_plant_winner": [],
        "marginal_price_winner": [],
        "marginal_volume_winner": [],
        "marginal_abs_volume_winner": [],
        "marginal_sequenceNumber_winner": [],
        "marginal_rank_in_stack_seq_winner": [],
        "marginal_rank_in_stack_price_winner": [],
        "marginal_price_gap_prev_winner": [],
        "marginal_price_gap_next_winner": [],
    }

    for _, row in out.iterrows():
        side = row["marginal_side_winner"]

        if side == "offer":
            prefix = "marginal_offer_"
        elif side == "bid":
            prefix = "marginal_bid_"
        else:
            prefix = None

        mapping = {
            "marginal_id_winner": "id",
            "marginal_plant_winner": "plant",
            "marginal_price_winner": "price",
            "marginal_volume_winner": "volume",
            "marginal_abs_volume_winner": "abs_volume",
            "marginal_sequenceNumber_winner": "sequenceNumber",
            "marginal_rank_in_stack_seq_winner": "rank_in_stack_seq",
            "marginal_rank_in_stack_price_winner": "rank_in_stack_price",
            "marginal_price_gap_prev_winner": "price_gap_prev",
            "marginal_price_gap_next_winner": "price_gap_next",
        }

        for field, suffix in mapping.items():
            if prefix is None:
                winner_fields[field].append(None)
            else:
                winner_fields[field].append(row.get(prefix + suffix))

    for field, values in winner_fields.items():
        out[field] = values

    same_side_context = {
        "winner_side_volume_total_energy": [],
        "winner_side_n_actions_energy": [],
        "winner_side_n_plants_energy": [],
        "winner_side_min_price_energy": [],
        "winner_side_max_price_energy": [],
        "winner_side_price_spread_energy": [],
    }

    for _, row in out.iterrows():
        side = row["marginal_side_winner"]
        if side == "offer":
            prefix = "offer"
        elif side == "bid":
            prefix = "bid"
        else:
            prefix = None

        mapping = {
            "winner_side_volume_total_energy": "volume_total_energy",
            "winner_side_n_actions_energy": "n_actions_energy",
            "winner_side_n_plants_energy": "n_plants_energy",
            "winner_side_min_price_energy": "min_price_energy",
            "winner_side_max_price_energy": "max_price_energy",
            "winner_side_price_spread_energy": "price_spread_energy",
        }

        for field, suffix in mapping.items():
            if prefix is None:
                same_side_context[field].append(None)
            else:
                same_side_context[field].append(row.get(f"{prefix}_{suffix}"))

    for field, values in same_side_context.items():
        out[field] = values

    out["winner_side_density_energy"] = out["winner_side_volume_total_energy"] / out["winner_side_n_actions_energy"]
    out["winner_side_concentration_energy"] = out["winner_side_volume_total_energy"] / out["winner_side_n_plants_energy"]

    return out


def build_summary(candidates: pd.DataFrame) -> dict:
    side_counts = candidates["marginal_side"].value_counts(dropna=False).to_dict()
    side_counts_winner = candidates["marginal_side_winner"].value_counts(dropna=False).to_dict()
    total = len(candidates)

    return {
        "n_settlement_periods": int(total),
        "marginal_side_counts": {str(k): int(v) for k, v in side_counts.items()},
        "marginal_side_shares": {str(k): float(v / total) for k, v in side_counts.items()},
        "marginal_side_winner_counts": {str(k): int(v) for k, v in side_counts_winner.items()},
        "marginal_side_winner_shares": {str(k): float(v / total) for k, v in side_counts_winner.items()},
        "offer_candidate_missing_count": int(candidates["marginal_offer_id"].isna().sum()) if "marginal_offer_id" in candidates.columns else total,
        "bid_candidate_missing_count": int(candidates["marginal_bid_id"].isna().sum()) if "marginal_bid_id" in candidates.columns else total,
        "top_offer_plants": (
            candidates["marginal_offer_plant"].value_counts(dropna=True).head(20).to_dict()
            if "marginal_offer_plant" in candidates.columns
            else {}
        ),
        "top_bid_plants": (
            candidates["marginal_bid_plant"].value_counts(dropna=True).head(20).to_dict()
            if "marginal_bid_plant" in candidates.columns
            else {}
        ),
    }


def save_outputs(candidates: pd.DataFrame, action_sp: pd.DataFrame, summary: dict) -> None:
    candidates.to_parquet(CANDIDATES_OUT, index=False)
    action_sp.to_parquet(ACTION_OUT, index=False)

    candidates.to_csv(CANDIDATES_OUT.with_suffix(".csv"), index=False)
    action_sp.to_csv(ACTION_OUT.with_suffix(".csv"), index=False)

    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ranked = load_ranked()
    candidates = build_candidates(ranked)
    action_sp = build_marginal_action_sp(candidates)
    summary = build_summary(candidates)
    save_outputs(candidates, action_sp, summary)

    print("=" * 72)
    print("Marginal Candidates One-Week Probe Built")
    print("=" * 72)
    print(f"Settlement periods : {len(candidates):,}")
    print(f"Candidates out     : {CANDIDATES_OUT}")
    print(f"Action SP out      : {ACTION_OUT}")
    print(f"Summary out        : {SUMMARY_OUT}")
    print("Marginal side counts (conservative):")
    for key, value in summary["marginal_side_counts"].items():
        print(f"  {key}: {value}")
    print("Marginal side counts (winner):")
    for key, value in summary["marginal_side_winner_counts"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
