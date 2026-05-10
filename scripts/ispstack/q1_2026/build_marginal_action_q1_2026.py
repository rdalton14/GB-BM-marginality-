from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ispstack" / "q1_2026"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "q1_2026" / "ispstack_marginal_action_q1_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUT_DIR / "accepted_actions_long_raw_q1_2026.parquet"
ENERGY_OUT = OUT_DIR / "accepted_actions_long_energy_q1_2026.parquet"
RANKED_OUT = OUT_DIR / "accepted_actions_ranked_q1_2026.parquet"
CANDIDATES_OUT = OUT_DIR / "marginal_candidates_sp_q1_2026.parquet"
ACTION_OUT = OUT_DIR / "marginal_action_sp_q1_2026.parquet"
SUMMARY_OUT = OUT_DIR / "marginal_action_q1_2026_summary.json"

GROUP_KEYS = ["settlementDate", "settlementPeriod"]


def load_raw_q1() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {RAW_DIR}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def add_core_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abs_volume"] = out["volume"].abs()
    out["is_offer"] = out["direction"].eq("offer")
    out["is_bid"] = out["direction"].eq("bid")
    out["is_energy_action"] = ~out["soFlag"].fillna(False)
    out["is_system_action"] = out["soFlag"].fillna(False)
    out["has_sentinel_price"] = out["finalPrice"].isin([-99999, 9999]) | out["originalPrice"].isin([-99999, 9999])
    out["stack_price"] = out["finalPrice"]
    out["stack_price_valid"] = ~out["has_sentinel_price"] & out["stack_price"].notna()
    out["stack_unit_id"] = out["id"].astype(str)
    out["plant_id_candidate"] = out["stack_unit_id"].str.replace(r"([_-])\d+$", "", regex=True)
    return out


def build_energy_actions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_energy_action"]].copy()


def _cum_abs_volume(group: pd.DataFrame, order_col: str) -> pd.Series:
    ordered = group.sort_values(order_col, kind="stable")
    csum = ordered["abs_volume"].cumsum()
    return csum.reindex(group.index)


def _price_gap_prev(group: pd.DataFrame, order_col: str, price_col: str) -> pd.Series:
    ordered = group.sort_values(order_col, kind="stable")
    gaps = ordered[price_col] - ordered[price_col].shift(1)
    return gaps.reindex(group.index)


def _price_gap_next(group: pd.DataFrame, order_col: str, price_col: str) -> pd.Series:
    ordered = group.sort_values(order_col, kind="stable")
    gaps = ordered[price_col].shift(-1) - ordered[price_col]
    return gaps.reindex(group.index)


def build_ranked_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keys = ["settlementDate", "settlementPeriod", "direction"]

    out["rank_in_stack_seq"] = out.groupby(keys)["sequenceNumber"].rank(method="first", ascending=True).astype("Int64")
    out["rank_in_stack_price"] = out.groupby(keys)["stack_price"].rank(method="first", ascending=True).astype("Int64")

    out["cum_abs_volume_seq"] = out.groupby(keys, group_keys=False).apply(lambda g: _cum_abs_volume(g, "sequenceNumber")).astype(float)
    out["cum_abs_volume_price"] = out.groupby(keys, group_keys=False).apply(lambda g: _cum_abs_volume(g, "stack_price")).astype(float)
    out["price_gap_prev_seq"] = out.groupby(keys, group_keys=False).apply(lambda g: _price_gap_prev(g, "sequenceNumber", "stack_price")).astype(float)
    out["price_gap_next_seq"] = out.groupby(keys, group_keys=False).apply(lambda g: _price_gap_next(g, "sequenceNumber", "stack_price")).astype(float)
    out["price_gap_prev_price"] = out.groupby(keys, group_keys=False).apply(lambda g: _price_gap_prev(g, "stack_price", "stack_price")).astype(float)
    out["price_gap_next_price"] = out.groupby(keys, group_keys=False).apply(lambda g: _price_gap_next(g, "stack_price", "stack_price")).astype(float)
    return out


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
                "offer_price_spread_energy": float(offers["stack_price"].max() - offers["stack_price"].min()) if not offers.empty else None,
                "bid_min_price_energy": float(bids["stack_price"].min()) if not bids.empty else None,
                "bid_max_price_energy": float(bids["stack_price"].max()) if not bids.empty else None,
                "bid_price_spread_energy": float(bids["stack_price"].max() - bids["stack_price"].min()) if not bids.empty else None,
            }
        )
    return pd.DataFrame.from_records(records)


def select_candidate(side_df: pd.DataFrame, side_name: str) -> pd.Series | None:
    if side_df.empty:
        return None
    valid = side_df[side_df["stack_price_valid"] & side_df["stack_price"].notna() & side_df["volume"].notna()].copy()
    if valid.empty:
        return None
    valid = valid.sort_values(by=["stack_price", "abs_volume", "sequenceNumber"], ascending=[False, False, False], kind="stable")
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
            f"marginal_{side_name}_price_gap_prev": float(top["price_gap_prev_seq"]) if pd.notna(top["price_gap_prev_seq"]) else None,
            f"marginal_{side_name}_price_gap_next": float(top["price_gap_next_seq"]) if pd.notna(top["price_gap_next_seq"]) else None,
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
        record: dict = {"settlementDate": settlement_date, "settlementPeriod": int(settlement_period)}
        offer_candidate = select_candidate(offers, "offer")
        bid_candidate = select_candidate(bids, "bid")
        if offer_candidate is not None:
            record.update(offer_candidate.to_dict())
        if bid_candidate is not None:
            record.update(bid_candidate.to_dict())

        has_offer = offer_candidate is not None
        has_bid = bid_candidate is not None
        if has_offer and not has_bid:
            record["marginal_side"] = "offer"
        elif has_bid and not has_offer:
            record["marginal_side"] = "bid"
        elif has_offer and has_bid:
            record["marginal_side"] = "mixed_or_ambiguous"
        else:
            record["marginal_side"] = "missing"

        offer_price = record.get("marginal_offer_price")
        bid_price = record.get("marginal_bid_price")
        if pd.notna(offer_price) and pd.isna(bid_price):
            record["marginal_side_winner"] = "offer"
        elif pd.notna(bid_price) and pd.isna(offer_price):
            record["marginal_side_winner"] = "bid"
        elif pd.notna(offer_price) and pd.notna(bid_price):
            if float(offer_price) > float(bid_price):
                record["marginal_side_winner"] = "offer"
            elif float(bid_price) > float(offer_price):
                record["marginal_side_winner"] = "bid"
            else:
                record["marginal_side_winner"] = "mixed_or_ambiguous"
        else:
            record["marginal_side_winner"] = "missing"
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
        prefix = "marginal_offer_" if side == "offer" else "marginal_bid_" if side == "bid" else None
        for field in selected_fields:
            selected_fields[field].append(None if prefix is None else row.get(prefix + field.replace("marginal_", "")))
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
    for _, row in out.iterrows():
        side = row["marginal_side_winner"]
        prefix = "marginal_offer_" if side == "offer" else "marginal_bid_" if side == "bid" else None
        for field, suffix in mapping.items():
            winner_fields[field].append(None if prefix is None else row.get(prefix + suffix))
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
    ctx_map = {
        "winner_side_volume_total_energy": "volume_total_energy",
        "winner_side_n_actions_energy": "n_actions_energy",
        "winner_side_n_plants_energy": "n_plants_energy",
        "winner_side_min_price_energy": "min_price_energy",
        "winner_side_max_price_energy": "max_price_energy",
        "winner_side_price_spread_energy": "price_spread_energy",
    }
    for _, row in out.iterrows():
        side = row["marginal_side_winner"]
        prefix = "offer" if side == "offer" else "bid" if side == "bid" else None
        for field, suffix in ctx_map.items():
            same_side_context[field].append(None if prefix is None else row.get(f"{prefix}_{suffix}"))
    for field, values in same_side_context.items():
        out[field] = values

    out["winner_side_density_energy"] = out["winner_side_volume_total_energy"] / out["winner_side_n_actions_energy"]
    out["winner_side_concentration_energy"] = out["winner_side_volume_total_energy"] / out["winner_side_n_plants_energy"]
    return out


def build_summary(raw_df: pd.DataFrame, energy_df: pd.DataFrame, ranked_df: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    valid_rank = ranked_df["rank_in_stack_seq"].notna() & ranked_df["rank_in_stack_price"].notna()
    rank_match_share = float((ranked_df.loc[valid_rank, "rank_in_stack_seq"] == ranked_df.loc[valid_rank, "rank_in_stack_price"]).mean()) if valid_rank.any() else None
    side_counts = candidates["marginal_side"].value_counts(dropna=False).to_dict()
    side_counts_winner = candidates["marginal_side_winner"].value_counts(dropna=False).to_dict()
    total_sps = len(candidates)

    return {
        "raw_rows": int(len(raw_df)),
        "energy_rows": int(len(energy_df)),
        "ranked_rows": int(len(ranked_df)),
        "settlement_period_rows": int(total_sps),
        "raw_offer_rows": int(raw_df["is_offer"].sum()),
        "raw_bid_rows": int(raw_df["is_bid"].sum()),
        "energy_offer_rows": int(energy_df["is_offer"].sum()),
        "energy_bid_rows": int(energy_df["is_bid"].sum()),
        "raw_system_action_share": float(raw_df["is_system_action"].mean()),
        "raw_energy_action_share": float(raw_df["is_energy_action"].mean()),
        "raw_sentinel_price_rows": int(raw_df["has_sentinel_price"].sum()),
        "energy_sentinel_price_rows": int(energy_df["has_sentinel_price"].sum()),
        "sequence_rank_matches_price_rank_share": rank_match_share,
        "marginal_side_counts": {str(k): int(v) for k, v in side_counts.items()},
        "marginal_side_shares": {str(k): float(v / total_sps) for k, v in side_counts.items()},
        "marginal_side_winner_counts": {str(k): int(v) for k, v in side_counts_winner.items()},
        "marginal_side_winner_shares": {str(k): float(v / total_sps) for k, v in side_counts_winner.items()},
    }


def save_outputs(raw_df: pd.DataFrame, energy_df: pd.DataFrame, ranked_df: pd.DataFrame, candidates: pd.DataFrame, action_sp: pd.DataFrame, summary: dict) -> None:
    for df, path in [
        (raw_df, RAW_OUT),
        (energy_df, ENERGY_OUT),
        (ranked_df, RANKED_OUT),
        (candidates, CANDIDATES_OUT),
        (action_sp, ACTION_OUT),
    ]:
        df.to_parquet(path, index=False)
        df.to_csv(path.with_suffix(".csv"), index=False)
    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    raw = add_core_flags(load_raw_q1())
    energy = build_energy_actions(raw)
    ranked = build_ranked_actions(energy)
    candidates = build_candidates(ranked)
    action_sp = build_marginal_action_sp(candidates)
    summary = build_summary(raw, energy, ranked, candidates)
    save_outputs(raw, energy, ranked, candidates, action_sp, summary)

    print("=" * 72)
    print("Q1 2026 ISPSTACK Marginal Action Build Complete")
    print("=" * 72)
    print(f"Raw rows             : {len(raw):,}")
    print(f"Energy rows          : {len(energy):,}")
    print(f"Settlement periods   : {len(candidates):,}")
    print(f"Winner side counts   : {summary['marginal_side_winner_counts']}")
    print(f"Output dir           : {OUT_DIR}")


if __name__ == "__main__":
    main()
