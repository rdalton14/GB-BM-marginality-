from __future__ import annotations

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Input: set either a single parquet file or a folder of daily parquets
# -------------------------------------------------------------------
INPUT_PATH = Path("data/raw/ispstack/q1_2026/2026-01-01.parquet")

OUTPUT_DIR = Path("data/processed/diagnostics")
SP_SUMMARY_PATH = OUTPUT_DIR / "2026-01-01_so_sp_summary.parquet"
SO_ACTIONS_PATH = OUTPUT_DIR / "2026-01-01_so_actions.parquet"
SO_PRICE_CONTRIB_PATH = OUTPUT_DIR / "2026-01-01_so_price_contributing_actions.parquet"
SO_BMU_SUMMARY_PATH = OUTPUT_DIR / "2026-01-01_so_bmu_summary.parquet"

REQUIRED_COLS = [
    "settlementDate",
    "settlementPeriod",
    "startTime",
    "direction",
    "id",
    "acceptanceId",
    "originalPrice",
    "finalPrice",
    "volume",
    "parAdjustedVolume",
    "tlmAdjustedVolume",
    "tlmAdjustedCost",
    "soFlag",
    "cadlFlag",
    "storProviderFlag",
    "repricedIndicator",
]


def load_ispstack(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in folder: {path}")
        return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    raise FileNotFoundError(f"Input path not found: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ispstack(INPUT_PATH)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df["direction"] = df["direction"].astype(str).str.upper()

    numeric_cols = ["settlementPeriod", "originalPrice", "finalPrice", "volume", "parAdjustedVolume", "tlmAdjustedVolume", "tlmAdjustedCost"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["abs_parAdjustedVolume"] = df["parAdjustedVolume"].abs()
    df["abs_tlmAdjustedCost"] = df["tlmAdjustedCost"].abs()

    so_actions = df.loc[df["soFlag"] == True].copy()
    so_price_contributing_actions = so_actions.loc[
        so_actions["parAdjustedVolume"].notna()
        & (so_actions["parAdjustedVolume"] != 0)
        & so_actions["finalPrice"].notna()
    ].copy()

    df["cost_proxy"] = df["finalPrice"] * df["parAdjustedVolume"]
    so_actions["cost_proxy"] = so_actions["finalPrice"] * so_actions["parAdjustedVolume"]
    so_price_contributing_actions["cost_proxy"] = (
        so_price_contributing_actions["finalPrice"] * so_price_contributing_actions["parAdjustedVolume"]
    )
    so_price_contributing_actions["abs_cost_proxy"] = so_price_contributing_actions["cost_proxy"].abs()

    sp_records = []
    for (settlement_date, settlement_period), g in df.groupby(["settlementDate", "settlementPeriod"], dropna=False):
        so_g = g.loc[g["soFlag"] == True].copy()
        so_offer = so_g.loc[so_g["direction"] == "OFFER"].copy()
        so_bid = so_g.loc[so_g["direction"] == "BID"].copy()
        so_pc = so_g.loc[
            so_g["parAdjustedVolume"].notna()
            & (so_g["parAdjustedVolume"] != 0)
            & so_g["finalPrice"].notna()
        ].copy()
        so_pc["cost_proxy"] = so_pc["finalPrice"] * so_pc["parAdjustedVolume"]
        so_pc["abs_cost_proxy"] = so_pc["cost_proxy"].abs()

        total_abs_par_volume = g["abs_parAdjustedVolume"].sum()
        so_abs_par_volume = so_g["abs_parAdjustedVolume"].sum()
        total_abs_tlm_cost = g["abs_tlmAdjustedCost"].sum()
        so_abs_tlm_cost = so_g["abs_tlmAdjustedCost"].sum()
        so_pc_abs_par = so_pc["abs_parAdjustedVolume"].sum()

        sp_records.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": settlement_period,
                "raw_action_count": int(len(g)),
                "so_action_count": int(len(so_g)),
                "so_offer_count": int(len(so_offer)),
                "so_bid_count": int(len(so_bid)),
                "so_price_contributing_count": int(len(so_pc)),
                "total_abs_par_volume": float(total_abs_par_volume),
                "so_abs_par_volume": float(so_abs_par_volume),
                "so_offer_abs_par_volume": float(so_offer["abs_parAdjustedVolume"].sum()),
                "so_bid_abs_par_volume": float(so_bid["abs_parAdjustedVolume"].sum()),
                "total_abs_tlm_cost": float(total_abs_tlm_cost),
                "so_abs_tlm_cost": float(so_abs_tlm_cost),
                "so_cost_proxy": float(so_pc["cost_proxy"].sum()),
                "so_abs_cost_proxy": float(so_pc["abs_cost_proxy"].sum()),
                "so_vwap_final_price": (
                    float((so_pc["finalPrice"] * so_pc["abs_parAdjustedVolume"]).sum() / so_pc_abs_par)
                    if so_pc_abs_par > 0
                    else pd.NA
                ),
            }
        )

    sp_summary = pd.DataFrame(sp_records).sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    sp_summary["so_share_actions"] = sp_summary["so_action_count"] / sp_summary["raw_action_count"]
    sp_summary["so_share_par_volume"] = sp_summary["so_abs_par_volume"] / sp_summary["total_abs_par_volume"].replace(0, pd.NA)
    sp_summary["so_share_tlm_cost"] = sp_summary["so_abs_tlm_cost"] / sp_summary["total_abs_tlm_cost"].replace(0, pd.NA)

    so_bmu_summary = (
        so_actions.groupby("id", dropna=False)
        .agg(
            so_action_count=("id", "size"),
            so_abs_par_volume=("abs_parAdjustedVolume", "sum"),
            so_abs_tlmAdjustedCost=("abs_tlmAdjustedCost", "sum"),
        )
        .reset_index()
        .sort_values(["so_action_count", "so_abs_par_volume", "so_abs_tlmAdjustedCost"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    total_raw_actions = len(df)
    total_so_actions = len(so_actions)
    so_action_share = total_so_actions / total_raw_actions if total_raw_actions else 0.0
    sps_with_any_so = int((sp_summary["so_action_count"] > 0).sum())
    sps_with_so_price_contrib = int((sp_summary["so_price_contributing_count"] > 0).sum())
    total_so_abs_par_volume = float(so_actions["abs_parAdjustedVolume"].sum())
    total_so_abs_tlm_cost = float(so_actions["abs_tlmAdjustedCost"].sum())
    avg_so_vwap_final_price = (
        float(sp_summary["so_vwap_final_price"].dropna().mean())
        if sp_summary["so_vwap_final_price"].notna().any()
        else pd.NA
    )

    print(f"Total raw actions: {total_raw_actions:,}")
    print(f"Total SO-flagged actions: {total_so_actions:,}")
    print(f"SO-flagged action share: {so_action_share:.4f}")
    print(f"Number of SPs with any SO-flagged action: {sps_with_any_so:,}")
    print(f"Number of SPs with SO-flagged price-contributing actions: {sps_with_so_price_contrib:,}")
    print(f"Total SO abs par volume: {total_so_abs_par_volume:,.4f}")
    print(f"Total SO abs tlm cost: {total_so_abs_tlm_cost:,.4f}")
    print(f"Average SO VWAP final price: {avg_so_vwap_final_price}")

    print("\nTop 10 SPs by so_abs_par_volume")
    print(
        sp_summary.sort_values("so_abs_par_volume", ascending=False)
        .head(10)[["settlementDate", "settlementPeriod", "so_abs_par_volume", "so_action_count", "so_price_contributing_count"]]
        .to_string(index=False)
    )

    print("\nTop 10 SPs by so_abs_tlm_cost")
    print(
        sp_summary.sort_values("so_abs_tlm_cost", ascending=False)
        .head(10)[["settlementDate", "settlementPeriod", "so_abs_tlm_cost", "so_action_count", "so_price_contributing_count"]]
        .to_string(index=False)
    )

    print("\nTop 10 SPs by so_vwap_final_price")
    print(
        sp_summary.sort_values("so_vwap_final_price", ascending=False)
        .head(10)[["settlementDate", "settlementPeriod", "so_vwap_final_price", "so_action_count", "so_price_contributing_count"]]
        .to_string(index=False)
    )

    print("\nTop 20 SO-flagged BMU IDs by action count / abs parAdjustedVolume / abs tlmAdjustedCost")
    print(so_bmu_summary.head(20).to_string(index=False))

    sp_summary.to_parquet(SP_SUMMARY_PATH, index=False)
    so_actions.to_parquet(SO_ACTIONS_PATH, index=False)
    so_price_contributing_actions.to_parquet(SO_PRICE_CONTRIB_PATH, index=False)
    so_bmu_summary.to_parquet(SO_BMU_SUMMARY_PATH, index=False)

    print(f"\nSaved: {SP_SUMMARY_PATH}")
    print(f"Saved: {SO_ACTIONS_PATH}")
    print(f"Saved: {SO_PRICE_CONTRIB_PATH}")
    print(f"Saved: {SO_BMU_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
