from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_ISP_DIR = DATA_DIR / "raw" / "ispstack" / "full_2023_2025"
REFERENCE_DIR = DATA_DIR / "raw" / "reference"
FUNDAMENTALS_DIR = DATA_DIR / "processed" / "full_2023_2025" / "fundamentals"
PROCESSED_DIR = DATA_DIR / "processed" / "full_2023_2025"
INTERIM_DIR = DATA_DIR / "interim" / "ispstack"
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics" / "audits"

OUTPUT_CSV = PROCESSED_DIR / "master_panel_2023_2025_energy_offer_stack.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "master_panel_2023_2025_energy_offer_stack.parquet"
MISSINGNESS_CSV = DIAGNOSTICS_DIR / "master_panel_2023_2025_energy_offer_stack_missingness.csv"
BUILD_LOG_CSV = DIAGNOSTICS_DIR / "master_panel_2023_2025_energy_offer_stack_build_log.csv"
BMU_AUDIT_CSV = DIAGNOSTICS_DIR / "master_panel_2023_2025_energy_offer_stack_bmu_match_audit.csv"
MARGINAL_AUDIT_CSV = DIAGNOSTICS_DIR / "master_panel_2023_2025_energy_offer_stack_marginal_audit.csv"
REFERENCE_OUTPUT_CSV = INTERIM_DIR / "bmu_reference_canonical_2023_2025.csv"

SP_KEY = ["settlementDate", "settlementPeriod"]
STACK_COLUMNS = [
    "settlementDate",
    "settlementPeriod",
    "id",
    "direction",
    "soFlag",
    "repricedIndicator",
    "bidOfferPairId",
    "originalPrice",
    "volume",
]

TECH_BUCKETS = [
    "BATTERY",
    "BIOMASS",
    "CCGT",
    "COAL",
    "DIESEL",
    "GAS",
    "LOAD RESPONSE",
    "NPSHYD",
    "OCGT",
    "OTHER",
    "PS",
    "SOLAR",
    "UNCLASSIFIED",
    "WIND",
]

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_tech(value: object) -> str:
    if pd.isna(value):
        return "UNCLASSIFIED"
    tech = str(value).strip().upper()
    tech = tech.replace("_", " ")
    aliases = {
        "BATTERY STORAGE": "BATTERY",
        "BESS": "BATTERY",
        "CCGT GAS": "CCGT",
        "GAS RECIP": "GAS",
        "LOAD_RESPONSE": "LOAD RESPONSE",
        "LOADRESPONSE": "LOAD RESPONSE",
        "PUMPED STORAGE": "PS",
        "PUMP STORAGE": "PS",
        "HYDRO": "NPSHYD",
        "NON-PS HYDRO": "NPSHYD",
        "NON PS HYDRO": "NPSHYD",
        "UNKNOWN": "UNCLASSIFIED",
    }
    tech = aliases.get(tech, tech)
    return tech if tech in TECH_BUCKETS else "OTHER"


def load_spine() -> pd.DataFrame:
    spine = pd.read_csv(FUNDAMENTALS_DIR / "system_price_niv_2023_2025.csv", usecols=["settlementDate", "settlementPeriod", "netImbalanceVolume"])
    spine["settlementPeriod"] = spine["settlementPeriod"].astype(int)
    spine = spine.rename(columns={"netImbalanceVolume": "niv"})
    return spine.sort_values(SP_KEY).reset_index(drop=True)


def build_bmu_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    register_q1 = pd.read_csv(REFERENCE_DIR / "bmu_register_q1_2026.csv", usecols=["elexon_bmu_id", "reg_fuel_type", "match_source"])
    register_q1 = register_q1.drop_duplicates(subset=["elexon_bmu_id"]).rename(
        columns={"elexon_bmu_id": "id", "reg_fuel_type": "tech_raw", "match_source": "mapping_detail"}
    )
    register_q1["mapping_source"] = "bmu_register_q1_2026"
    register_q1["mapping_priority"] = 1

    supplement = pd.read_csv(REFERENCE_DIR / "bmu_register_supplement_2023_2025.csv", usecols=["elexon_bmu_id", "reg_fuel_type", "source"])
    supplement = supplement.drop_duplicates(subset=["elexon_bmu_id"]).rename(
        columns={"elexon_bmu_id": "id", "reg_fuel_type": "tech_raw", "source": "mapping_detail"}
    )
    supplement["mapping_source"] = "bmu_register_supplement_2023_2025"
    supplement["mapping_priority"] = 2

    reference = pd.concat([register_q1, supplement], ignore_index=True)
    reference = reference.sort_values(["id", "mapping_priority"]).drop_duplicates(subset=["id"], keep="first")
    reference["tech_final"] = reference["tech_raw"].map(normalize_tech)
    reference["mapping_confidence"] = reference["mapping_source"].map(
        {
            "bmu_register_q1_2026": "high",
            "bmu_register_supplement_2023_2025": "medium",
        }
    )
    reference["is_unmatched"] = False
    reference = reference[["id", "tech_raw", "tech_final", "mapping_source", "mapping_detail", "mapping_confidence", "is_unmatched"]]

    stack_ids = (
        pd.read_parquet(RAW_ISP_DIR, columns=["id"])["id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
        .to_frame(name="id")
    )
    audit = stack_ids.merge(reference, on="id", how="left")
    audit["is_unmatched"] = audit["tech_final"].isna()
    audit["tech_final"] = audit["tech_final"].fillna("UNCLASSIFIED")
    audit["mapping_source"] = audit["mapping_source"].fillna("unmatched")
    audit["mapping_detail"] = audit["mapping_detail"].fillna("")
    audit["mapping_confidence"] = audit["mapping_confidence"].fillna("unmatched")
    audit["tech_raw"] = audit["tech_raw"].fillna("")

    reference.to_csv(REFERENCE_OUTPUT_CSV, index=False)
    audit.to_csv(BMU_AUDIT_CSV, index=False)
    return reference, audit


def load_stack() -> pd.DataFrame:
    stack = pd.read_parquet(RAW_ISP_DIR, columns=STACK_COLUMNS)
    stack["settlementDate"] = stack["settlementDate"].astype(str).str[:10]
    stack["settlementPeriod"] = stack["settlementPeriod"].astype(int)
    stack["id"] = stack["id"].astype(str)
    stack["soFlag"] = stack["soFlag"].astype(bool)
    stack["repricedIndicator"] = stack["repricedIndicator"].astype(bool)
    stack["originalPrice"] = pd.to_numeric(stack["originalPrice"], errors="coerce")
    stack["volume"] = pd.to_numeric(stack["volume"], errors="coerce")
    stack["direction"] = stack["direction"].astype(str).str.lower()
    return stack


def build_panel() -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    build_log: list[dict] = []
    spine = load_spine()
    reference, audit = build_bmu_reference()
    stack = load_stack().merge(reference[["id", "tech_final"]], on="id", how="left")
    stack["tech_final"] = stack["tech_final"].fillna("UNCLASSIFIED")

    offers = stack[stack["direction"] == "offer"].copy()
    bids = stack[stack["direction"] == "bid"].copy()
    offer_energy = offers[offers["soFlag"] == False].copy()
    offer_system = offers[offers["soFlag"] == True].copy()
    bid_energy = bids[bids["soFlag"] == False].copy()
    bid_system = bids[bids["soFlag"] == True].copy()

    build_log.extend(
        [
            {"step": "spine_rows", "value": len(spine)},
            {"step": "stack_rows", "value": len(stack)},
            {"step": "offers_rows", "value": len(offers)},
            {"step": "bids_rows", "value": len(bids)},
            {"step": "offer_energy_rows", "value": len(offer_energy)},
            {"step": "offer_system_rows", "value": len(offer_system)},
            {"step": "bid_energy_rows", "value": len(bid_energy)},
            {"step": "bid_system_rows", "value": len(bid_system)},
            {"step": "unmatched_bmu_ids", "value": int(audit["is_unmatched"].sum())},
        ]
    )

    panel = spine.copy()

    def merge_part(base: pd.DataFrame, part: pd.DataFrame, name: str) -> pd.DataFrame:
        before = len(base)
        merged = base.merge(part, on=SP_KEY, how="left", validate="one_to_one")
        build_log.append(
            {
                "step": f"merge_{name}",
                "value": len(merged),
                "rows_before": before,
                "rows_after": len(merged),
                "duplicate_keys_after": int(merged.duplicated(SP_KEY).sum()),
            }
        )
        return merged

    offer_totals = (
        offers.groupby(SP_KEY)
        .agg(
            offer_volume_total=("volume", "sum"),
            offer_n_bmus=("id", "nunique"),
            offer_n_energy_actions=("soFlag", lambda s: int((~s).sum())),
            offer_n_system_actions=("soFlag", lambda s: int(s.sum())),
        )
        .reset_index()
    )
    offer_energy_totals = offer_energy.groupby(SP_KEY)["volume"].sum().reset_index(name="offer_volume_energy")
    offer_system_totals = offer_system.groupby(SP_KEY)["volume"].sum().reset_index(name="offer_volume_system")

    bid_totals = (
        bids.groupby(SP_KEY)
        .agg(
            bid_volume_total=("volume", "sum"),
            bid_n_bmus=("id", "nunique"),
            bid_n_energy_actions=("soFlag", lambda s: int((~s).sum())),
            bid_n_system_actions=("soFlag", lambda s: int(s.sum())),
        )
        .reset_index()
    )
    bid_energy_totals = bid_energy.groupby(SP_KEY)["volume"].sum().reset_index(name="bid_volume_energy")
    bid_system_totals = bid_system.groupby(SP_KEY)["volume"].sum().reset_index(name="bid_volume_system")

    tech_volumes = (
        offer_energy.groupby(SP_KEY + ["tech_final"])["volume"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    tech_volumes.columns.name = None
    for tech in TECH_BUCKETS:
        if tech not in tech_volumes.columns:
            tech_volumes[tech] = 0.0
    tech_volumes = tech_volumes[SP_KEY + TECH_BUCKETS]
    tech_volumes = tech_volumes.rename(columns={tech: f"vol_{tech.lower().replace(' ', '_')}" for tech in TECH_BUCKETS})

    energy_offer_price_mask = (~offer_energy["repricedIndicator"]) & offer_energy["originalPrice"].notna() & (offer_energy["originalPrice"] < 9999)
    offer_max_price_energy = (
        offer_energy.loc[energy_offer_price_mask]
        .groupby(SP_KEY)["originalPrice"]
        .max()
        .reset_index(name="offer_max_price_energy")
    )
    bid_min_price_energy = (
        bid_energy.loc[bid_energy["originalPrice"].notna()]
        .groupby(SP_KEY)["originalPrice"]
        .min()
        .reset_index(name="bid_min_price_energy")
    )

    marginal_candidates = offer_energy.loc[energy_offer_price_mask].copy()
    marginal_idx = marginal_candidates.groupby(SP_KEY)["originalPrice"].idxmax()
    marginal = (
        marginal_candidates.loc[marginal_idx, SP_KEY + ["originalPrice", "id", "tech_final"]]
        .rename(
            columns={
                "originalPrice": "marginal_price_energy_offer",
                "id": "marginal_bmu_energy_offer",
                "tech_final": "marginal_tech_energy_offer",
            }
        )
        .reset_index(drop=True)
    )
    marginal["marginal_defined_energy_offer"] = 1

    for name, part in [
        ("offer_totals", offer_totals),
        ("offer_energy_totals", offer_energy_totals),
        ("offer_system_totals", offer_system_totals),
        ("offer_max_price_energy", offer_max_price_energy),
        ("bid_totals", bid_totals),
        ("bid_energy_totals", bid_energy_totals),
        ("bid_system_totals", bid_system_totals),
        ("bid_min_price_energy", bid_min_price_energy),
        ("tech_volumes", tech_volumes),
        ("marginal", marginal),
    ]:
        panel = merge_part(panel, part, name)

    panel["bess_offer_volume"] = panel["vol_battery"].fillna(0.0)
    bess_bid = (
        bid_energy.loc[bid_energy["tech_final"] == "BATTERY"]
        .groupby(SP_KEY)["volume"]
        .sum()
        .reset_index(name="bess_bid_volume")
    )
    panel = merge_part(panel, bess_bid, "bess_bid_volume")

    zero_fill_cols = [
        "offer_volume_total",
        "offer_volume_energy",
        "offer_volume_system",
        "offer_n_bmus",
        "offer_n_energy_actions",
        "offer_n_system_actions",
        "bid_volume_total",
        "bid_volume_energy",
        "bid_volume_system",
        "bid_n_bmus",
        "bid_n_energy_actions",
        "bid_n_system_actions",
        "bess_offer_volume",
        "bess_bid_volume",
    ] + [f"vol_{tech.lower().replace(' ', '_')}" for tech in TECH_BUCKETS]
    for col in zero_fill_cols:
        panel[col] = panel[col].fillna(0)

    int_cols = [
        "offer_n_bmus",
        "offer_n_energy_actions",
        "offer_n_system_actions",
        "bid_n_bmus",
        "bid_n_energy_actions",
        "bid_n_system_actions",
    ]
    for col in int_cols:
        panel[col] = panel[col].astype(int)

    panel["marginal_defined_energy_offer"] = panel["marginal_defined_energy_offer"].fillna(0).astype(int)
    panel.loc[
        panel["marginal_defined_energy_offer"].eq(1) & panel["marginal_bmu_energy_offer"].isna(),
        "marginal_bmu_energy_offer",
    ] = "UNKNOWN_BMU"
    panel["has_any_offer"] = (panel["offer_volume_total"] > 0).astype(int)
    panel["has_any_bid"] = (panel["bid_volume_total"] > 0).astype(int)
    panel["has_energy_offer"] = (panel["offer_volume_energy"] > 0).astype(int)
    panel["has_energy_bid"] = (panel["bid_volume_energy"] > 0).astype(int)
    panel["has_system_offer"] = (panel["offer_volume_system"] > 0).astype(int)
    panel["has_system_bid"] = (panel["bid_volume_system"] > 0).astype(int)
    panel["offer_system_share"] = panel["offer_volume_system"].div(panel["offer_volume_total"].replace(0, pd.NA))
    panel["gross_bm_volume_energy"] = panel["offer_volume_energy"] - panel["bid_volume_energy"]
    panel["bess_offer_share"] = panel["bess_offer_volume"].div(panel["offer_volume_energy"].replace(0, pd.NA))
    panel["bess_bid_share"] = panel["bess_bid_volume"].div(panel["bid_volume_energy"].abs().replace(0, pd.NA))

    marginal_audit = panel[
        SP_KEY
        + [
            "has_energy_offer",
            "offer_volume_energy",
            "offer_max_price_energy",
            "marginal_price_energy_offer",
            "marginal_bmu_energy_offer",
            "marginal_tech_energy_offer",
            "marginal_defined_energy_offer",
        ]
    ].copy()
    marginal_audit.to_csv(MARGINAL_AUDIT_CSV, index=False)

    ordered_cols = [
        "settlementDate",
        "settlementPeriod",
        "niv",
        "offer_volume_total",
        "offer_volume_energy",
        "offer_volume_system",
        "offer_n_bmus",
        "offer_n_energy_actions",
        "offer_n_system_actions",
        "offer_max_price_energy",
        "bid_volume_total",
        "bid_volume_energy",
        "bid_volume_system",
        "bid_n_bmus",
        "bid_n_energy_actions",
        "bid_n_system_actions",
        "bid_min_price_energy",
    ] + [f"vol_{tech.lower().replace(' ', '_')}" for tech in TECH_BUCKETS] + [
        "bess_offer_volume",
        "bess_bid_volume",
        "bess_offer_share",
        "bess_bid_share",
        "marginal_price_energy_offer",
        "marginal_bmu_energy_offer",
        "marginal_tech_energy_offer",
        "marginal_defined_energy_offer",
        "offer_system_share",
        "gross_bm_volume_energy",
        "has_any_offer",
        "has_any_bid",
        "has_energy_offer",
        "has_energy_bid",
        "has_system_offer",
        "has_system_bid",
    ]

    panel = panel[ordered_cols].sort_values(SP_KEY).reset_index(drop=True)
    return panel, build_log, audit


def main() -> None:
    panel, build_log, audit = build_panel()

    panel.to_csv(OUTPUT_CSV, index=False)
    panel.to_parquet(OUTPUT_PARQUET, index=False)

    missingness = panel.isna().sum().rename("missingValues").reset_index()
    missingness.columns = ["column", "missingValues"]
    missingness.to_csv(MISSINGNESS_CSV, index=False)
    pd.DataFrame(build_log).to_csv(BUILD_LOG_CSV, index=False)

    print("Built separate 2023-2025 energy-offer stack panel")
    print(f"shape = {panel.shape}")
    print(f"duplicate SP keys = {int(panel.duplicated(SP_KEY).sum())}")
    print(f"rows with undefined marginal = {int((panel['marginal_defined_energy_offer'] == 0).sum())}")
    print(f"unmatched BMU IDs = {int(audit['is_unmatched'].sum())}")
    print(f"saved csv = {OUTPUT_CSV}")
    print(f"saved parquet = {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
