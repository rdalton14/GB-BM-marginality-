from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

RAW_INPUT = (
    PROJECT_ROOT
    / "data"
    / "raw_test"
    / "ispstack"
    / "ispstack_one_week_probe_2026_01_05_to_2026_01_11"
    / "ispstack_one_week_long.parquet"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_test"
    / "ispstack_one_week_probe_accepted_actions"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUT_DIR / "accepted_actions_long_raw.parquet"
ENERGY_OUT = OUT_DIR / "accepted_actions_long_energy.parquet"
RANKED_OUT = OUT_DIR / "accepted_actions_ranked.parquet"
SUMMARY_OUT = OUT_DIR / "accepted_actions_one_week_summary.json"


def load_raw() -> pd.DataFrame:
    return pd.read_parquet(RAW_INPUT)


def add_core_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["abs_volume"] = out["volume"].abs()
    out["is_offer"] = out["direction"].eq("offer")
    out["is_bid"] = out["direction"].eq("bid")
    out["is_energy_action"] = ~out["soFlag"].fillna(False)
    out["is_system_action"] = out["soFlag"].fillna(False)
    out["has_sentinel_price"] = (
        out["finalPrice"].isin([-99999, 9999]) | out["originalPrice"].isin([-99999, 9999])
    )
    out["stack_price"] = out["finalPrice"]
    out["stack_price_valid"] = ~out["has_sentinel_price"] & out["stack_price"].notna()

    # Keep a raw identifier column and a cleaner candidate plant field for later mapping work.
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
    group_keys = ["settlementDate", "settlementPeriod", "direction"]

    # Canonical stack order from sequenceNumber
    out["rank_in_stack_seq"] = (
        out.groupby(group_keys)["sequenceNumber"]
        .rank(method="first", ascending=True)
        .astype("Int64")
    )

    # Diagnostic price-based order within side
    out["rank_in_stack_price"] = (
        out.groupby(group_keys)["stack_price"]
        .rank(method="first", ascending=True)
        .astype("Int64")
    )

    out["cum_abs_volume_seq"] = (
        out.groupby(group_keys, group_keys=False)
        .apply(lambda g: _cum_abs_volume(g, "sequenceNumber"))
        .astype(float)
    )
    out["cum_abs_volume_price"] = (
        out.groupby(group_keys, group_keys=False)
        .apply(lambda g: _cum_abs_volume(g, "stack_price"))
        .astype(float)
    )

    out["price_gap_prev_seq"] = (
        out.groupby(group_keys, group_keys=False)
        .apply(lambda g: _price_gap_prev(g, "sequenceNumber", "stack_price"))
        .astype(float)
    )
    out["price_gap_next_seq"] = (
        out.groupby(group_keys, group_keys=False)
        .apply(lambda g: _price_gap_next(g, "sequenceNumber", "stack_price"))
        .astype(float)
    )

    out["price_gap_prev_price"] = (
        out.groupby(group_keys, group_keys=False)
        .apply(lambda g: _price_gap_prev(g, "stack_price", "stack_price"))
        .astype(float)
    )
    out["price_gap_next_price"] = (
        out.groupby(group_keys, group_keys=False)
        .apply(lambda g: _price_gap_next(g, "stack_price", "stack_price"))
        .astype(float)
    )

    return out


def build_summary(raw_df: pd.DataFrame, energy_df: pd.DataFrame, ranked_df: pd.DataFrame) -> dict:
    def share(mask: pd.Series, denom: int) -> float | None:
        return float(mask.mean()) if denom else None

    # Difference between sequence and price ordering as a diagnostic only
    valid_rank = ranked_df["rank_in_stack_seq"].notna() & ranked_df["rank_in_stack_price"].notna()
    rank_match_share = None
    if valid_rank.any():
        rank_match_share = float(
            (ranked_df.loc[valid_rank, "rank_in_stack_seq"] == ranked_df.loc[valid_rank, "rank_in_stack_price"]).mean()
        )

    summary = {
        "raw_rows": int(len(raw_df)),
        "energy_rows": int(len(energy_df)),
        "ranked_rows": int(len(ranked_df)),
        "raw_offer_rows": int(raw_df["is_offer"].sum()),
        "raw_bid_rows": int(raw_df["is_bid"].sum()),
        "energy_offer_rows": int(energy_df["is_offer"].sum()),
        "energy_bid_rows": int(energy_df["is_bid"].sum()),
        "raw_system_action_share": share(raw_df["is_system_action"], len(raw_df)),
        "raw_energy_action_share": share(raw_df["is_energy_action"], len(raw_df)),
        "raw_sentinel_price_rows": int(raw_df["has_sentinel_price"].sum()),
        "energy_sentinel_price_rows": int(energy_df["has_sentinel_price"].sum()),
        "sequence_rank_matches_price_rank_share": rank_match_share,
        "raw_columns": sorted(raw_df.columns.tolist()),
        "ranked_columns": sorted(ranked_df.columns.tolist()),
        "volume_sign_by_direction": (
            raw_df.assign(
                volume_sign=np.select(
                    [raw_df["volume"] > 0, raw_df["volume"] < 0, raw_df["volume"] == 0],
                    ["positive", "negative", "zero"],
                    default="missing",
                )
            )
            .groupby(["direction", "volume_sign"])
            .size()
            .reset_index(name="row_count")
            .to_dict(orient="records")
        ),
    }
    return summary


def save_outputs(raw_df: pd.DataFrame, energy_df: pd.DataFrame, ranked_df: pd.DataFrame, summary: dict) -> None:
    raw_df.to_parquet(RAW_OUT, index=False)
    energy_df.to_parquet(ENERGY_OUT, index=False)
    ranked_df.to_parquet(RANKED_OUT, index=False)

    raw_df.to_csv(RAW_OUT.with_suffix(".csv"), index=False)
    energy_df.to_csv(ENERGY_OUT.with_suffix(".csv"), index=False)
    ranked_df.to_csv(RANKED_OUT.with_suffix(".csv"), index=False)

    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    raw = add_core_flags(load_raw())
    energy = build_energy_actions(raw)
    ranked = build_ranked_actions(energy)
    summary = build_summary(raw, energy, ranked)
    save_outputs(raw, energy, ranked, summary)

    print("=" * 72)
    print("Accepted Actions One-Week Probe Built")
    print("=" * 72)
    print(f"Raw rows      : {len(raw):,}")
    print(f"Energy rows   : {len(energy):,}")
    print(f"Ranked rows   : {len(ranked):,}")
    print(f"Raw out       : {RAW_OUT}")
    print(f"Energy out    : {ENERGY_OUT}")
    print(f"Ranked out    : {RANKED_OUT}")
    print(f"Summary out   : {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
