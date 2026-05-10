from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
IN_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "ispstack_marginal_action_2023_2025"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "soflag_margin_impact_2023_2025"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_ACTIONS = IN_DIR / "accepted_actions_long_raw_2023_2025.parquet"

SUMMARY_JSON = OUT_DIR / "soflag_margin_impact_summary_2023_2025.json"
SP_COMPARISON_CSV = OUT_DIR / "soflag_margin_impact_sp_comparison_2023_2025.csv"

GROUP_KEYS = ["settlementDate", "settlementPeriod"]


def load_actions() -> pd.DataFrame:
    df = pd.read_parquet(RAW_ACTIONS)
    df["settlementPeriod"] = df["settlementPeriod"].astype("int16")
    df["sequenceNumber"] = df["sequenceNumber"].astype("int32")
    df["id"] = df["id"].astype("string")
    df["direction"] = df["direction"].astype("string")
    for col in ["soFlag", "repricedIndicator"]:
        df[col] = df[col].fillna(False).astype(bool)
    df["abs_volume"] = pd.to_numeric(df["volume"], errors="coerce").abs()
    df["stack_price"] = pd.to_numeric(df["finalPrice"], errors="coerce")
    df["has_sentinel_price"] = df["stack_price"].isin([-99999, 9999]) | pd.to_numeric(df["originalPrice"], errors="coerce").isin([-99999, 9999])
    df["stack_price_valid"] = ~df["has_sentinel_price"] & df["stack_price"].notna()
    df["plant_id_candidate"] = df["id"].astype(str).str.replace(r"([_-])\d+$", "", regex=True)
    return df


def select_candidate(side_df: pd.DataFrame, include_so: bool) -> pd.Series | None:
    if side_df.empty:
        return None

    valid = side_df[side_df["stack_price_valid"] & side_df["stack_price"].notna() & side_df["volume"].notna()].copy()
    if not include_so:
        valid = valid[~valid["soFlag"]].copy()
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
            "id": top["id"],
            "plant": top["plant_id_candidate"],
            "price": float(top["stack_price"]),
            "volume": float(top["volume"]),
            "sequenceNumber": int(top["sequenceNumber"]),
            "soFlag": bool(top["soFlag"]),
        }
    )


def build_candidates(df: pd.DataFrame, include_so: bool) -> pd.DataFrame:
    records: list[dict] = []
    for (settlement_date, settlement_period), grp in df.groupby(GROUP_KEYS, sort=True):
        offers = grp[grp["direction"] == "offer"]
        bids = grp[grp["direction"] == "bid"]

        offer = select_candidate(offers, include_so=include_so)
        bid = select_candidate(bids, include_so=include_so)

        rec: dict[str, object] = {
            "settlementDate": settlement_date,
            "settlementPeriod": int(settlement_period),
        }
        if offer is not None:
            for k, v in offer.items():
                rec[f"offer_{k}"] = v
        if bid is not None:
            for k, v in bid.items():
                rec[f"bid_{k}"] = v

        offer_price = rec.get("offer_price")
        bid_price = rec.get("bid_price")
        if pd.notna(offer_price) and pd.isna(bid_price):
            winner_side = "offer"
        elif pd.notna(bid_price) and pd.isna(offer_price):
            winner_side = "bid"
        elif pd.notna(offer_price) and pd.notna(bid_price):
            if float(offer_price) > float(bid_price):
                winner_side = "offer"
            elif float(bid_price) > float(offer_price):
                winner_side = "bid"
            else:
                winner_side = "mixed_or_ambiguous"
        else:
            winner_side = "missing"

        rec["winner_side"] = winner_side
        if winner_side == "offer":
            rec["winner_id"] = rec.get("offer_id")
            rec["winner_plant"] = rec.get("offer_plant")
            rec["winner_price"] = rec.get("offer_price")
            rec["winner_soFlag"] = rec.get("offer_soFlag")
        elif winner_side == "bid":
            rec["winner_id"] = rec.get("bid_id")
            rec["winner_plant"] = rec.get("bid_plant")
            rec["winner_price"] = rec.get("bid_price")
            rec["winner_soFlag"] = rec.get("bid_soFlag")
        else:
            rec["winner_id"] = pd.NA
            rec["winner_plant"] = pd.NA
            rec["winner_price"] = pd.NA
            rec["winner_soFlag"] = pd.NA

        records.append(rec)

    return pd.DataFrame.from_records(records)


def main() -> None:
    df = load_actions()
    all_actions = build_candidates(df, include_so=True)
    energy_only = build_candidates(df, include_so=False)

    compare = all_actions.merge(
        energy_only,
        on=GROUP_KEYS,
        how="outer",
        suffixes=("_all", "_energy"),
        validate="one_to_one",
    )

    compare["offer_margin_is_so_all"] = compare["offer_soFlag_all"].fillna(False).astype(bool)
    compare["bid_margin_is_so_all"] = compare["bid_soFlag_all"].fillna(False).astype(bool)
    compare["winner_margin_is_so_all"] = compare["winner_soFlag_all"].fillna(False).astype(bool)

    compare["offer_margin_changes_with_so_reincluded"] = (
        compare["offer_id_all"].astype("string") != compare["offer_id_energy"].astype("string")
    ) & compare["offer_id_all"].notna() & compare["offer_id_energy"].notna()
    compare["bid_margin_changes_with_so_reincluded"] = (
        compare["bid_id_all"].astype("string") != compare["bid_id_energy"].astype("string")
    ) & compare["bid_id_all"].notna() & compare["bid_id_energy"].notna()
    compare["winner_changes_with_so_reincluded"] = (
        compare["winner_id_all"].astype("string") != compare["winner_id_energy"].astype("string")
    ) & compare["winner_id_all"].notna() & compare["winner_id_energy"].notna()

    compare.to_csv(SP_COMPARISON_CSV, index=False)

    total_sps = len(compare)
    offer_candidates_all = int(compare["offer_id_all"].notna().sum())
    bid_candidates_all = int(compare["bid_id_all"].notna().sum())
    winner_candidates_all = int(compare["winner_id_all"].notna().sum())

    summary = {
        "total_settlement_periods": int(total_sps),
        "offer_margin_with_so": {
            "candidate_sps": offer_candidates_all,
            "soflag_true_count": int(compare["offer_margin_is_so_all"].sum()),
            "soflag_true_share_of_offer_candidate_sps": float(compare["offer_margin_is_so_all"].sum() / offer_candidates_all) if offer_candidates_all else None,
            "changes_vs_energy_only_count": int(compare["offer_margin_changes_with_so_reincluded"].sum()),
            "changes_vs_energy_only_share": float(compare["offer_margin_changes_with_so_reincluded"].mean()),
        },
        "bid_margin_with_so": {
            "candidate_sps": bid_candidates_all,
            "soflag_true_count": int(compare["bid_margin_is_so_all"].sum()),
            "soflag_true_share_of_bid_candidate_sps": float(compare["bid_margin_is_so_all"].sum() / bid_candidates_all) if bid_candidates_all else None,
            "changes_vs_energy_only_count": int(compare["bid_margin_changes_with_so_reincluded"].sum()),
            "changes_vs_energy_only_share": float(compare["bid_margin_changes_with_so_reincluded"].mean()),
        },
        "winner_margin_with_so": {
            "candidate_sps": winner_candidates_all,
            "soflag_true_count": int(compare["winner_margin_is_so_all"].sum()),
            "soflag_true_share_of_winner_candidate_sps": float(compare["winner_margin_is_so_all"].sum() / winner_candidates_all) if winner_candidates_all else None,
            "changes_vs_energy_only_count": int(compare["winner_changes_with_so_reincluded"].sum()),
            "changes_vs_energy_only_share": float(compare["winner_changes_with_so_reincluded"].mean()),
            "winner_side_counts_all": compare["winner_side_all"].fillna("missing").value_counts(dropna=False).to_dict(),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print("SO-FLAG MARGIN IMPACT (2023-2025)")
    print("=" * 72)
    print(f"Total settlement periods                 : {total_sps:,}")
    print(
        "Offer-side marginal is SO-flagged       : "
        f"{summary['offer_margin_with_so']['soflag_true_count']:,} / {offer_candidates_all:,} "
        f"({summary['offer_margin_with_so']['soflag_true_share_of_offer_candidate_sps']:.2%})"
    )
    print(
        "Bid-side marginal is SO-flagged         : "
        f"{summary['bid_margin_with_so']['soflag_true_count']:,} / {bid_candidates_all:,} "
        f"({summary['bid_margin_with_so']['soflag_true_share_of_bid_candidate_sps']:.2%})"
    )
    print(
        "Winner marginal is SO-flagged           : "
        f"{summary['winner_margin_with_so']['soflag_true_count']:,} / {winner_candidates_all:,} "
        f"({summary['winner_margin_with_so']['soflag_true_share_of_winner_candidate_sps']:.2%})"
    )
    print(
        "Winner changes when SO is re-included   : "
        f"{summary['winner_margin_with_so']['changes_vs_energy_only_count']:,} / {total_sps:,} "
        f"({summary['winner_margin_with_so']['changes_vs_energy_only_share']:.2%})"
    )
    print(f"\nSaved summary -> {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
