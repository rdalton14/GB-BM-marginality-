from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/ispstack/q1_2026/2026-01-01.parquet")
OUTPUT_DIR = Path("data/processed/diagnostics")
SP_SUMMARY_PATH = OUTPUT_DIR / "2026-01-01_so_constraint_sp_summary.parquet"
SO_ACTIONS_PATH = OUTPUT_DIR / "2026-01-01_so_constraint_actions.parquet"
BMU_SUMMARY_PATH = OUTPUT_DIR / "2026-01-01_so_constraint_bmu_summary.parquet"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH).copy()

    df["direction"] = df["direction"].astype(str).str.upper()
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["originalPrice"] = pd.to_numeric(df["originalPrice"], errors="coerce")

    df["abs_volume"] = df["volume"].abs()
    df["raw_cost_proxy"] = df["originalPrice"] * df["volume"]
    df["abs_raw_cost_proxy"] = (df["originalPrice"] * df["volume"]).abs()

    so_actions = df.loc[df["soFlag"] == True].copy()

    sp_rows = []
    for (settlement_date, settlement_period), g in df.groupby(["settlementDate", "settlementPeriod"], dropna=False):
        so_g = g.loc[g["soFlag"] == True].copy()
        so_offer = so_g.loc[so_g["direction"] == "OFFER"].copy()
        so_bid = so_g.loc[so_g["direction"] == "BID"].copy()

        total_abs_volume = g["abs_volume"].sum()
        so_abs_volume = so_g["abs_volume"].sum()

        sp_rows.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": settlement_period,
                "total_action_count": int(len(g)),
                "so_action_count": int(len(so_g)),
                "so_action_share": float(len(so_g) / len(g)) if len(g) else 0.0,
                "so_offer_count": int(len(so_offer)),
                "so_bid_count": int(len(so_bid)),
                "total_abs_volume": float(total_abs_volume),
                "so_abs_volume": float(so_abs_volume),
                "so_volume_share": float(so_abs_volume / total_abs_volume) if total_abs_volume else 0.0,
                "so_offer_abs_volume": float(so_offer["abs_volume"].sum()),
                "so_bid_abs_volume": float(so_bid["abs_volume"].sum()),
                "so_raw_cost_proxy": float(so_g["raw_cost_proxy"].sum()),
                "so_abs_raw_cost_proxy": float(so_g["abs_raw_cost_proxy"].sum()),
                "average_so_original_price": float(so_g["originalPrice"].mean()) if len(so_g) else pd.NA,
                "so_vwap_original_price": (
                    float((so_g["originalPrice"] * so_g["abs_volume"]).sum() / so_abs_volume)
                    if so_abs_volume > 0
                    else pd.NA
                ),
            }
        )

    sp_summary = pd.DataFrame(sp_rows).sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)

    so_bmu_summary = (
        so_actions.groupby("id", dropna=False)
        .agg(
            so_action_count=("id", "size"),
            so_abs_volume=("abs_volume", "sum"),
            so_abs_raw_cost_proxy=("abs_raw_cost_proxy", "sum"),
        )
        .reset_index()
        .sort_values(["so_action_count", "so_abs_volume", "so_abs_raw_cost_proxy"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    total_raw_actions = len(df)
    total_so_actions = len(so_actions)
    so_action_share = total_so_actions / total_raw_actions if total_raw_actions else 0.0
    sps_with_any_so = int((sp_summary["so_action_count"] > 0).sum())
    total_so_abs_volume = float(so_actions["abs_volume"].sum())
    total_so_abs_raw_cost_proxy = float(so_actions["abs_raw_cost_proxy"].sum())
    so_vwap_original_price = (
        float((so_actions["originalPrice"] * so_actions["abs_volume"]).sum() / total_so_abs_volume)
        if total_so_abs_volume > 0
        else pd.NA
    )
    so_offer_abs_volume = float(so_actions.loc[so_actions["direction"] == "OFFER", "abs_volume"].sum())
    so_bid_abs_volume = float(so_actions.loc[so_actions["direction"] == "BID", "abs_volume"].sum())
    total_so_side_volume = so_offer_abs_volume + so_bid_abs_volume
    so_offer_volume_share = so_offer_abs_volume / total_so_side_volume if total_so_side_volume else 0.0
    so_bid_volume_share = so_bid_abs_volume / total_so_side_volume if total_so_side_volume else 0.0

    print(f"Total raw actions: {total_raw_actions:,}")
    print(f"Total SO-flagged actions: {total_so_actions:,}")
    print(f"SO action share: {so_action_share:.4f}")
    print(f"SPs with any SO-flagged action: {sps_with_any_so:,}")
    print(f"Total SO abs accepted volume: {total_so_abs_volume:,.4f}")
    print(f"Total SO abs raw cost proxy: {total_so_abs_raw_cost_proxy:,.4f}")
    print(f"SO VWAP original price: {so_vwap_original_price}")
    print(
        "SO volume split (abs volume): "
        f"OFFER={so_offer_abs_volume:,.4f} ({so_offer_volume_share:.4f}), "
        f"BID={so_bid_abs_volume:,.4f} ({so_bid_volume_share:.4f})"
    )

    print("\nTop 10 SPs by so_action_share")
    print(
        sp_summary.sort_values("so_action_share", ascending=False)
        .head(10)[["settlementDate", "settlementPeriod", "so_action_share", "so_action_count", "total_action_count"]]
        .to_string(index=False)
    )

    print("\nTop 10 SPs by so_abs_volume")
    print(
        sp_summary.sort_values("so_abs_volume", ascending=False)
        .head(10)[["settlementDate", "settlementPeriod", "so_abs_volume", "so_offer_abs_volume", "so_bid_abs_volume"]]
        .to_string(index=False)
    )

    print("\nTop 10 SPs by so_abs_raw_cost_proxy")
    print(
        sp_summary.sort_values("so_abs_raw_cost_proxy", ascending=False)
        .head(10)[["settlementDate", "settlementPeriod", "so_abs_raw_cost_proxy", "so_raw_cost_proxy"]]
        .to_string(index=False)
    )

    print("\nTop 20 BMU IDs by SO action count / abs volume / abs raw cost proxy")
    print(so_bmu_summary.head(20).to_string(index=False))

    sp_summary.to_parquet(SP_SUMMARY_PATH, index=False)
    so_actions.to_parquet(SO_ACTIONS_PATH, index=False)
    so_bmu_summary.to_parquet(BMU_SUMMARY_PATH, index=False)

    print(f"\nSaved: {SP_SUMMARY_PATH}")
    print(f"Saved: {SO_ACTIONS_PATH}")
    print(f"Saved: {BMU_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
