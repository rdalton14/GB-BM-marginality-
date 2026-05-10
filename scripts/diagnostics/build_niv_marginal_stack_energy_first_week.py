from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

STACK_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "q1_2026" / "bid_offer_stack_first_week_energy"
    / "bid_offer_stack_energy_first_week_q1_2026_long.parquet"
)
NIV_PATH = (
    PROJECT_ROOT
    / "archive" / "final_non_raw_data_archive_2026_04_27"
    / "data" / "processed" / "q1_2026" / "fundamentals"
    / "system_price_niv_q1_2026.csv"
)
OUT_DIR = STACK_PATH.parent
BMU_MAP_PATH = OUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_bmu_map.csv"
REGISTER_PATH = PROJECT_ROOT / "data" / "raw" / "reference" / "bmu_register_first_week_q1_2026.csv"
OUT_LONG = OUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_niv_marginal.parquet"
OUT_SP = OUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_niv_marginal_sp_summary.parquet"

SP_KEYS = ["settlementDate", "settlementPeriod"]

BMU_REF_COLS = ["fuel_type_resolved", "leadPartyName", "bmUnitType", "generationCapacity", "gspGroupName"]
BMU_REF_RENAME = {
    "fuel_type_resolved": "fuel_type",
    "leadPartyName": "lead_party_name",
    "bmUnitType": "bmu_type",
    "generationCapacity": "gen_capacity_mw",
    "gspGroupName": "gsp_group",
}


def load_niv() -> pd.DataFrame:
    niv = pd.read_csv(NIV_PATH)
    niv["settlementDate"] = pd.to_datetime(niv["settlementDate"])
    niv["settlementPeriod"] = niv["settlementPeriod"].astype(int)
    niv = niv[
        SP_KEYS
        + ["netImbalanceVolume", "systemLongShort", "systemBuyPrice", "systemSellPrice", "systemPrice"]
    ].copy()
    niv["niv_active_side"] = niv["netImbalanceVolume"].apply(
        lambda v: "offer" if v > 0 else ("bid" if v < 0 else "neutral")
    )
    return niv.rename(columns={"netImbalanceVolume": "niv_volume", "systemLongShort": "niv_long_short"})


def identify_marginal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_niv_marginal"] = False
    df["n_tied_marginal_candidates"] = pd.array([pd.NA] * len(df), dtype="Int64")

    def _flag_all_at_top(subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        sp_max = subset.groupby(SP_KEYS)["finalPrice"].transform("max")
        df.loc[subset.index[subset["finalPrice"] == sp_max], "is_niv_marginal"] = True

    _flag_all_at_top(df[(df["niv_active_side"] == "offer") & (df["side"] == "offer")].copy())
    _flag_all_at_top(
        df[(df["niv_active_side"] == "bid") & (df["side"] == "bid") & df["bid_rank_price_desc"].notna()].copy()
    )

    # De-duplicate: same BMU can have multiple rows at the same price in one SP;
    # keep the highest stack_rank row per (SP, BMU) so each unit appears once.
    marginals = df[df["is_niv_marginal"]]
    keep_idx = marginals.groupby(SP_KEYS + ["resolved_bmu_id"])["stack_rank"].idxmax().values
    df["is_niv_marginal"] = False
    df.loc[keep_idx, "is_niv_marginal"] = True

    # n_tied_marginal_candidates = unique BMUs at the top price per SP
    counts = df[df["is_niv_marginal"]].groupby(SP_KEYS)["resolved_bmu_id"].transform("count")
    df.loc[df["is_niv_marginal"], "n_tied_marginal_candidates"] = counts.astype("Int64").values

    return df


def build_sp_summary(df: pd.DataFrame) -> pd.DataFrame:
    marginal = df[df["is_niv_marginal"]].copy()
    marginal = marginal.rename(
        columns={
            "id":                         "marginal_bmu_raw",
            "resolved_bmu_id":            "marginal_bmu_id",
            "is_numeric_id":              "marginal_is_numeric_id",
            "identity_source":            "marginal_identity_source",
            "finalPrice":                 "marginal_final_price",
            "originalPrice":              "marginal_original_price",
            "stack_rank":                 "marginal_stack_rank",
            "bid_rank_price_desc":        "marginal_bid_rank",
            "volume":                     "marginal_volume",
            "n_tied_marginal_candidates": "n_tied_marginal_candidates",
            "fuel_type":                  "marginal_fuel_type",
            "family_id":                  "marginal_family_id",
            "lead_party_name":            "marginal_lead_party",
            "bmu_type":                   "marginal_bmu_type",
            "gen_capacity_mw":            "marginal_gen_capacity_mw",
            "gsp_group":                  "marginal_gsp_group",
        }
    )
    marginal["marginal_price_cost_to_so"] = marginal.apply(
        lambda r: -r["marginal_final_price"] if r["niv_active_side"] == "bid" else r["marginal_final_price"],
        axis=1,
    )
    keep = SP_KEYS + [
        "niv_volume", "niv_long_short", "niv_active_side",
        "systemBuyPrice", "systemSellPrice", "systemPrice",
        "marginal_bmu_raw", "marginal_bmu_id", "marginal_is_numeric_id",
        "marginal_identity_source", "marginal_fuel_type", "marginal_family_id",
        "marginal_lead_party", "marginal_bmu_type", "marginal_gen_capacity_mw", "marginal_gsp_group",
        "marginal_final_price", "marginal_original_price",
        "marginal_price_cost_to_so", "marginal_stack_rank", "marginal_bid_rank", "marginal_volume",
        "n_tied_marginal_candidates",
    ]
    return marginal[keep].sort_values(SP_KEYS).reset_index(drop=True)


def main() -> None:
    print("Loading bid-offer stack ...")
    stack = pd.read_parquet(STACK_PATH)
    n_sps = stack[SP_KEYS].drop_duplicates().shape[0]
    print(f"  rows: {len(stack):,}   SPs: {n_sps:,}")

    print("Loading NIV ...")
    niv = load_niv()
    stack_dates = stack["settlementDate"].dt.normalize().unique()
    niv = niv[niv["settlementDate"].isin(stack_dates)]
    print(f"  NIV rows matched to stack dates: {len(niv):,}")

    print("Joining NIV to stack ...")
    df = stack.merge(niv, on=SP_KEYS, how="left", validate="many_to_one")
    n_null_niv = df["niv_volume"].isna().sum()
    assert n_null_niv == 0, f"{n_null_niv} rows missing NIV after join"

    active_side_sp = df.groupby(SP_KEYS)["niv_active_side"].first().value_counts()
    print(f"  NIV active side (SP-level): {active_side_sp.to_dict()}")

    print("Identifying marginal rows ...")
    df = identify_marginal(df)
    marginal_df = df[df["is_niv_marginal"]]
    n_marginal_candidates = int(df["is_niv_marginal"].sum())
    n_marginal_sps = marginal_df[SP_KEYS].drop_duplicates().shape[0]
    n_numeric_marginals = int(marginal_df["is_numeric_id"].sum())
    sps_no_marginal = n_sps - n_marginal_sps
    print(f"  SPs with a marginal      : {n_marginal_sps:,} / {n_sps:,}")
    print(f"  Total marginal candidates: {n_marginal_candidates:,}  (> SPs where prices tied)")
    print(f"  Numeric-ID marginals     : {n_numeric_marginals:,}")
    if sps_no_marginal:
        print(f"  Dropping {sps_no_marginal:,} SPs with no marginal (neutral NIV or empty active-side stack)")
    df = df.merge(marginal_df[SP_KEYS].drop_duplicates(), on=SP_KEYS, how="inner")

    print("Joining BMU reference (fuel type, party, capacity) ...")
    bmu_map = pd.read_csv(BMU_MAP_PATH)[["resolved_bmu_id"] + BMU_REF_COLS].rename(columns=BMU_REF_RENAME)
    df = df.merge(bmu_map, on="resolved_bmu_id", how="left")
    n_fuel_null = df["fuel_type"].isna().sum()
    print(f"  fuel_type coverage: {len(df) - n_fuel_null:,} / {len(df):,} rows ({n_fuel_null:,} null — expected for numeric IDs)")

    print("Joining family_id from register ...")
    assert REGISTER_PATH.exists(), f"Register not found: {REGISTER_PATH} — run build_bmu_register_first_week.py first"
    reg = pd.read_csv(REGISTER_PATH)[["elexon_bmu_id", "family_id"]]
    df = df.merge(reg, left_on="resolved_bmu_id", right_on="elexon_bmu_id", how="left").drop(columns="elexon_bmu_id")
    n_fam_null = df["family_id"].isna().sum()
    print(f"  family_id coverage: {len(df) - n_fam_null:,} / {len(df):,} rows")

    print("Building SP summary ...")
    sp_summary = build_sp_summary(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_LONG, index=False)
    sp_summary.to_parquet(OUT_SP, index=False)
    sp_summary.to_csv(OUT_SP.with_suffix(".csv"), index=False)

    print(f"Saved -> {OUT_LONG}")
    print(f"Saved -> {OUT_SP}")
    print()
    print("SP summary preview (first 10):")
    preview_cols = [
        "settlementDate", "settlementPeriod", "niv_volume", "niv_active_side",
        "marginal_bmu_id", "marginal_is_numeric_id", "marginal_final_price", "marginal_stack_rank",
    ]
    print(sp_summary[preview_cols].head(10).to_string(index=False))
    print()
    print("Marginal tech breakdown (by bmu prefix):")
    sp_summary["bmu_prefix"] = sp_summary["marginal_bmu_id"].astype(str).str[:2]
    print(sp_summary.groupby(["niv_active_side", "marginal_is_numeric_id"]).size().rename("n_sps").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
