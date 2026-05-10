from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

STACK_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "full_2023_2025" / "bid_offer_stack_2023_2025"
    / "bid_offer_stack_energy_2023_2025_long.parquet"
)
NIV_PATH = (
    PROJECT_ROOT
    / "archive" / "final_non_raw_data_archive_2026_04_27"
    / "data" / "processed" / "full_2023_2025" / "fundamentals"
    / "system_price_niv_2023_2025.csv"
)
OUT_CSV = (
    PROJECT_ROOT
    / "reports" / "full_2023_2025" / "diagnostics"
    / "bid_marginal_selection_logic_check_2023_2025.csv"
)
SP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "full_2023_2025" / "bid_offer_stack_2023_2025"
    / "bid_offer_stack_energy_2023_2025_niv_marginal_sp_summary.parquet"
)

SP_KEYS = ["settlementDate", "settlementPeriod"]


def main() -> None:
    stack = pd.read_parquet(STACK_PATH)
    niv = pd.read_csv(
        NIV_PATH,
        usecols=["settlementDate", "settlementPeriod", "netImbalanceVolume", "systemPrice"],
    )

    stack["settlementDate"] = pd.to_datetime(stack["settlementDate"])
    niv["settlementDate"] = pd.to_datetime(niv["settlementDate"])
    niv["niv_active_side"] = np.where(
        niv["netImbalanceVolume"].gt(0),
        "offer",
        np.where(niv["netImbalanceVolume"].lt(0), "bid", "neutral"),
    )

    df = stack.merge(
        niv[SP_KEYS + ["niv_active_side", "systemPrice", "netImbalanceVolume"]],
        on=SP_KEYS,
        how="inner",
    )
    bid = df.loc[df["niv_active_side"].eq("bid") & df["side"].eq("bid")].copy()

    stats = (
        bid.groupby(SP_KEYS)
        .agg(
            current_code_final_price=("finalPrice", "max"),
            intended_mirror_final_price=("finalPrice", "min"),
            n_bid_rows=("finalPrice", "size"),
            niv_volume=("netImbalanceVolume", "first"),
            system_price=("systemPrice", "first"),
        )
        .reset_index()
    )
    stats["differs"] = stats["current_code_final_price"].ne(stats["intended_mirror_final_price"])
    stats["current_code_cost_to_so"] = -stats["current_code_final_price"]
    stats["intended_mirror_cost_to_so"] = -stats["intended_mirror_final_price"]
    stats["cost_to_so_delta"] = (
        stats["intended_mirror_cost_to_so"] - stats["current_code_cost_to_so"]
    )

    summary = pd.read_parquet(SP_SUMMARY_PATH)
    summary["settlementDate"] = pd.to_datetime(summary["settlementDate"])
    summary_bid = summary.loc[summary["niv_active_side"].eq("bid")].copy()
    summary_prices = (
        summary_bid.groupby(SP_KEYS)
        .agg(
            summary_min_selected_price=("marginal_final_price", "min"),
            summary_max_selected_price=("marginal_final_price", "max"),
            summary_n_candidates=("marginal_final_price", "size"),
        )
        .reset_index()
    )
    stats = stats.merge(summary_prices, on=SP_KEYS, how="left")
    stats["summary_matches_intended_min"] = stats["summary_min_selected_price"].eq(
        stats["intended_mirror_final_price"]
    )
    stats["summary_still_matches_old_max"] = stats["summary_max_selected_price"].eq(
        stats["current_code_final_price"]
    )

    examples = stats.loc[stats["differs"]].sort_values(SP_KEYS).head(10)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_CSV, index=False)

    print("Bid-side marginal selection logic check")
    print(f"  Bid-active SPs with bid energy actions : {len(stats):,}")
    print(f"  Current max(finalPrice) = intended min(finalPrice): {(~stats['differs']).sum():,}")
    print(
        f"  Differing SPs                           : {stats['differs'].sum():,} "
        f"({stats['differs'].mean() * 100:.2f}%)"
    )
    print()
    print("Selected bid finalPrice:")
    print(f"  Current code median max(finalPrice)     : {stats['current_code_final_price'].median():.4f}")
    print(f"  Intended mirror median min(finalPrice)  : {stats['intended_mirror_final_price'].median():.4f}")
    print(f"  Current code mean max(finalPrice)       : {stats['current_code_final_price'].mean():.4f}")
    print(f"  Intended mirror mean min(finalPrice)    : {stats['intended_mirror_final_price'].mean():.4f}")
    print()
    print("Selected bid cost-to-SO = -finalPrice:")
    print(f"  Current code median cost-to-SO          : {stats['current_code_cost_to_so'].median():.4f}")
    print(f"  Intended mirror median cost-to-SO       : {stats['intended_mirror_cost_to_so'].median():.4f}")
    print(f"  Median intended-current cost delta      : {stats['cost_to_so_delta'].median():.4f}")
    print()
    print("Current rebuilt SP summary validation:")
    print(
        f"  Summary matches intended min(finalPrice): "
        f"{stats['summary_matches_intended_min'].sum():,} / {stats['summary_matches_intended_min'].notna().sum():,}"
    )
    print(
        f"  Summary still matches old max(finalPrice): "
        f"{stats['summary_still_matches_old_max'].sum():,} / {stats['summary_still_matches_old_max'].notna().sum():,}"
    )
    print()
    print("First differing examples:")
    print(examples.to_string(index=False))
    print()
    print(f"Saved full check -> {OUT_CSV}")


if __name__ == "__main__":
    main()
