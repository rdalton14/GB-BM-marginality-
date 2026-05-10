from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/diagnostics/2026-01-01_so_constraint_actions.parquet")
OUTPUT_DIR = Path("data/processed/diagnostics")
BMU_DIRECTION_OUT = OUTPUT_DIR / "2026-01-01_so_bmu_direction_summary.parquet"
SP_DIRECTION_OUT = OUTPUT_DIR / "2026-01-01_so_sp_direction_summary.parquet"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    so_actions = pd.read_parquet(INPUT_PATH).copy()

    so_actions["direction"] = so_actions["direction"].astype(str).str.upper()
    so_actions["settlementPeriod"] = pd.to_numeric(so_actions["settlementPeriod"], errors="coerce")
    so_actions["volume"] = pd.to_numeric(so_actions["volume"], errors="coerce")
    so_actions["originalPrice"] = pd.to_numeric(so_actions["originalPrice"], errors="coerce")

    so_actions["abs_volume"] = so_actions["volume"].abs()
    so_actions["raw_cost_proxy"] = so_actions["originalPrice"] * so_actions["volume"]
    so_actions["abs_raw_cost_proxy"] = so_actions["raw_cost_proxy"].abs()
    so_actions["weighted_price_volume"] = so_actions["originalPrice"] * so_actions["abs_volume"]

    bmu_direction_summary = (
        so_actions.groupby(["id", "direction"], dropna=False)
        .agg(
            action_count=("id", "size"),
            abs_volume=("abs_volume", "sum"),
            raw_cost_proxy=("raw_cost_proxy", "sum"),
            abs_raw_cost_proxy=("abs_raw_cost_proxy", "sum"),
            weighted_price_volume=("weighted_price_volume", "sum"),
            mean_original_price=("originalPrice", "mean"),
            min_original_price=("originalPrice", "min"),
            max_original_price=("originalPrice", "max"),
            number_of_settlement_periods_present=("settlementPeriod", "nunique"),
        )
        .reset_index()
    )
    bmu_direction_summary["vwap_original_price"] = (
        bmu_direction_summary["weighted_price_volume"] / bmu_direction_summary["abs_volume"]
    )
    bmu_direction_summary = bmu_direction_summary.drop(columns=["weighted_price_volume"])

    sp_direction = (
        so_actions.groupby(["settlementPeriod", "direction"], dropna=False)
        .agg(
            so_abs_volume=("abs_volume", "sum"),
            so_abs_raw_cost_proxy=("abs_raw_cost_proxy", "sum"),
        )
        .reset_index()
    )

    bid_sp = (
        sp_direction.loc[sp_direction["direction"] == "BID", ["settlementPeriod", "so_abs_volume", "so_abs_raw_cost_proxy"]]
        .rename(
            columns={
                "so_abs_volume": "so_bid_abs_volume",
                "so_abs_raw_cost_proxy": "so_bid_abs_raw_cost_proxy",
            }
        )
    )
    offer_sp = (
        sp_direction.loc[sp_direction["direction"] == "OFFER", ["settlementPeriod", "so_abs_volume", "so_abs_raw_cost_proxy"]]
        .rename(
            columns={
                "so_abs_volume": "so_offer_abs_volume",
                "so_abs_raw_cost_proxy": "so_offer_abs_raw_cost_proxy",
            }
        )
    )

    sp_direction_summary = (
        pd.DataFrame({"settlementPeriod": range(1, 49)})
        .merge(bid_sp, on="settlementPeriod", how="left")
        .merge(offer_sp, on="settlementPeriod", how="left")
        .fillna(0.0)
    )
    total_so_volume = sp_direction_summary["so_bid_abs_volume"] + sp_direction_summary["so_offer_abs_volume"]
    sp_direction_summary["bid_share_of_so_volume"] = (
        sp_direction_summary["so_bid_abs_volume"] / total_so_volume.replace(0, pd.NA)
    )
    sp_direction_summary["offer_share_of_so_volume"] = (
        sp_direction_summary["so_offer_abs_volume"] / total_so_volume.replace(0, pd.NA)
    )

    print("\nTop 20 BMUs by SO BID abs_volume")
    print(
        bmu_direction_summary.loc[bmu_direction_summary["direction"] == "BID"]
        .sort_values("abs_volume", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\nTop 20 BMUs by SO OFFER abs_volume")
    print(
        bmu_direction_summary.loc[bmu_direction_summary["direction"] == "OFFER"]
        .sort_values("abs_volume", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\nTop 20 BMUs by SO BID abs_raw_cost_proxy")
    print(
        bmu_direction_summary.loc[bmu_direction_summary["direction"] == "BID"]
        .sort_values("abs_raw_cost_proxy", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\nTop 20 BMUs by SO OFFER abs_raw_cost_proxy")
    print(
        bmu_direction_summary.loc[bmu_direction_summary["direction"] == "OFFER"]
        .sort_values("abs_raw_cost_proxy", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    bmu_direction_summary.to_parquet(BMU_DIRECTION_OUT, index=False)
    sp_direction_summary.to_parquet(SP_DIRECTION_OUT, index=False)

    print(f"\nSaved: {BMU_DIRECTION_OUT}")
    print(f"Saved: {SP_DIRECTION_OUT}")


if __name__ == "__main__":
    main()
