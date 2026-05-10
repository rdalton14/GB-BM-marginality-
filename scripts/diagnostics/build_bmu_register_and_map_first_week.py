from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

BMU_ENDPOINT = "https://data.elexon.co.uk/bmrs/api/v1/reference/bmunits/all"
STACK_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "q1_2026" / "bid_offer_stack_first_week_energy"
    / "bid_offer_stack_energy_first_week_q1_2026_niv_marginal.parquet"
)
OUT_REGISTER = PROJECT_ROOT / "data" / "raw" / "reference" / "bmu_register_elexon_current.csv"
OUT_MAP = (
    PROJECT_ROOT
    / "data" / "processed" / "q1_2026" / "bid_offer_stack_first_week_energy"
    / "bid_offer_stack_energy_first_week_q1_2026_bmu_map.csv"
)

KEEP_COLS = [
    "elexonBmUnit", "nationalGridBmUnit", "bmUnitName", "fuelType",
    "leadPartyName", "leadPartyId", "bmUnitType", "productionOrConsumptionFlag",
    "generationCapacity", "demandCapacity", "transmissionLossFactor",
    "gspGroupId", "gspGroupName", "interconnectorId", "eic",
]

BMU_FUEL_TYP_PATH = PROJECT_ROOT / "data" / "raw" / "reference" / "BMUFuelTyp.xlsx"

PARTY_FUEL_MAP = {
    "INFINIS LIMITED": "GAS",
    "OAKTREE POWER LIMITED": "DIESEL",
    "GRIDBEYOND LIMITED": "LOAD RESPONSE",
    "EQUIWATT LIMITED": "LOAD RESPONSE",
}

# ── Named-BMU inference rules (applied in priority order for NaN fuel types) ──
# Level 1: bmUnitName keyword → fuel type
UNIT_NAME_KEYWORDS: list[tuple[str, str]] = [
    ("bess", "BATTERY"),
    ("battery", "BATTERY"),
    ("storage", "BATTERY"),
]

# Level 2: leadPartyName substring → fuel type (lowercase match)
PARTY_SUBSTRING_MAP: list[tuple[str, str]] = [
    # Battery operators
    ("tesla", "BATTERY"),
    ("bp gas marketing", "BATTERY"),       # BP's BM units are BESS (Pillswood, etc.)
    ("arenko", "BATTERY"),
    ("pivot power", "BATTERY"),
    ("pivoted power", "BATTERY"),
    ("bess holdco", "BATTERY"),
    ("centrica business solutions", "BATTERY"),
    ("cbs energy storage", "BATTERY"),
    ("roosecote", "BATTERY"),
    ("statkraft", "BATTERY"),              # Statkraft secondary 2__ units = battery portfolio
    ("sp renewables", "BATTERY"),
    # Demand/load response aggregators
    ("flexitricity", "LOAD RESPONSE"),
    ("limejump", "LOAD RESPONSE"),
    ("octopus energy", "LOAD RESPONSE"),
    ("gridbeyond", "LOAD RESPONSE"),
    ("sse energy supply", "LOAD RESPONSE"),
    ("edf energy customers", "LOAD RESPONSE"),
    # OCGT/diesel
    ("uk power reserve", "OCGT"),
    ("conrad energy", "DIESEL"),
]

# Sibling-resolved overrides: manually confirmed from family-member lookup
SIBLING_OVERRIDES: dict[str, str] = {
    "E_PETEM3":   "CCGT",   # sibling E_PETEM1 (Peterborough, River Nene) = CCGT in both registers
    "2__DLOND001": "OCGT",  # sibling 2__DLOND002 (EDFE Aggregate MANWEB02) = OCGT in both registers
}


def infer_fuel_type(row: pd.Series) -> str | None:
    """Infer fuel type from bmUnitName keywords then leadPartyName substrings."""
    name = str(row.get("bmUnitName", "") or "").lower()
    for kw, ft in UNIT_NAME_KEYWORDS:
        if kw in name:
            return ft
    party = str(row.get("leadPartyName", "") or "").lower()
    for substr, ft in PARTY_SUBSTRING_MAP:
        if substr in party:
            return ft
    return None


def fetch_bmu_register() -> pd.DataFrame:
    print(f"GET {BMU_ENDPOINT} ...")
    for attempt in range(3):
        try:
            r = requests.get(BMU_ENDPOINT, headers={"Accept": "application/json"}, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    else:
        raise RuntimeError("Failed to fetch BMU register after 3 attempts")

    df = pd.DataFrame(data)
    print(f"  {len(df):,} BMUs returned")
    return df


def load_fuel_typ() -> pd.DataFrame:
    """Load BMUFuelTyp.xlsx — primary curated fuel type source for named BMUs."""
    df = pd.read_excel(BMU_FUEL_TYP_PATH)[
        ["SETT UNIT ID", "REG FUEL TYPE", "BMRS FUEL TYPE", "CRM UNIT CAT", "REG TYPE"]
    ]
    df = df.dropna(subset=["SETT UNIT ID"]).rename(columns={
        "SETT UNIT ID": "elexonBmUnit",
        "REG FUEL TYPE": "reg_fuel_type",
        "BMRS FUEL TYPE": "bmrs_fuel_type",
        "CRM UNIT CAT": "crm_unit_cat",
        "REG TYPE": "reg_type",
    })
    df["elexonBmUnit"] = df["elexonBmUnit"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["elexonBmUnit"], keep="first")
    return df


def main() -> None:
    # ── 1. Pull fresh register ────────────────────────────────────────────────
    reg = fetch_bmu_register()

    available = [c for c in KEEP_COLS if c in reg.columns]
    reg = reg[available].copy()
    reg["elexonBmUnit"] = reg["elexonBmUnit"].astype(str).str.strip()

    OUT_REGISTER.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(OUT_REGISTER, index=False)
    print(f"Saved register -> {OUT_REGISTER}  ({len(reg):,} rows, {len(available)} cols)")

    # ── 2. Load stack, get unique BMUs ────────────────────────────────────────
    stack = pd.read_parquet(STACK_PATH)
    unique_bmus = (
        stack[["resolved_bmu_id", "is_numeric_id", "identity_source"]]
        .drop_duplicates(subset=["resolved_bmu_id"])
        .sort_values("resolved_bmu_id")
        .reset_index(drop=True)
    )
    print(f"\nUnique BMUs in stack: {len(unique_bmus):,}")
    print(f"  Named   : {(~unique_bmus['is_numeric_id']).sum():,}")
    print(f"  Numeric : {unique_bmus['is_numeric_id'].sum():,}")

    # ── 3. Load curated fuel type from BMUFuelTyp.xlsx ───────────────────────
    fuel_typ = load_fuel_typ()
    print(f"\nBMUFuelTyp.xlsx: {len(fuel_typ):,} rows with SETT UNIT ID")

    # ── 4. Map named BMUs ─────────────────────────────────────────────────────
    named = unique_bmus[~unique_bmus["is_numeric_id"]].copy()

    # Join fresh Elexon register for metadata (capacity, party, GSP, etc.)
    mapped = named.merge(reg, left_on="resolved_bmu_id", right_on="elexonBmUnit", how="left")

    # Join BMUFuelTyp for curated fuel type (primary source)
    mapped = mapped.merge(fuel_typ[["elexonBmUnit", "reg_fuel_type", "bmrs_fuel_type", "crm_unit_cat", "reg_type"]],
                          on="elexonBmUnit", how="left")

    # Resolved fuel type: prefer curated BMUFuelTyp, fall back to fresh register fuelType
    mapped["fuel_type_resolved"] = mapped["reg_fuel_type"].where(
        mapped["reg_fuel_type"].notna(), mapped["fuelType"]
    )

    # Inference pass: fill remaining NaN fuel types from bmUnitName / leadPartyName
    still_null = mapped["fuel_type_resolved"].isna()
    if still_null.any():
        inferred = mapped[still_null].apply(infer_fuel_type, axis=1)
        mapped.loc[still_null, "fuel_type_inferred"] = inferred
        mapped.loc[still_null & inferred.notna(), "fuel_type_resolved"] = inferred[inferred.notna()]

    # Remaining NaN → OTHER (catch-all for unknown aggregators/traders)
    mapped["fuel_type_resolved"] = mapped["fuel_type_resolved"].fillna("OTHER")

    # Named interconnectors (I_I* BMUs appearing directly in ISPSTACK): bmUnitType='I' → INTERCONNECTOR
    ic_named_mask = mapped["bmUnitType"] == "I"
    mapped.loc[ic_named_mask, "fuel_type_resolved"] = "INTERCONNECTOR"

    # Refinement: promote Elexon fuelType sub-type (CCGT/OCGT) where reg_fuel_type is only 'GAS'
    # The Elexon register carries plant-level technology for T_* and some E_* BMUs;
    # BMUFuelTyp uses 'GAS' as a coarser umbrella — use the Elexon value where it's more specific.
    ELEXON_SUBTYPES = {"CCGT", "OCGT"}
    gas_mask = mapped["fuel_type_resolved"] == "GAS"
    elexon_specific = mapped["fuelType"].isin(ELEXON_SUBTYPES)
    promoted = gas_mask & elexon_specific
    mapped.loc[promoted, "fuel_type_resolved"] = mapped.loc[promoted, "fuelType"]
    n_promoted = int(promoted.sum())

    # Sibling-resolved overrides (two confirmed from family-member lookup)
    sibling_mask = mapped["elexonBmUnit"].isin(SIBLING_OVERRIDES)
    mapped.loc[sibling_mask, "fuel_type_resolved"] = mapped.loc[sibling_mask, "elexonBmUnit"].map(SIBLING_OVERRIDES)
    n_sibling = int(sibling_mask.sum())

    n_from_register = mapped["reg_fuel_type"].notna().sum()
    n_from_elexon = (mapped["reg_fuel_type"].isna() & mapped["fuelType"].notna()).sum()
    n_inferred = mapped.get("fuel_type_inferred", pd.Series(dtype=str)).notna().sum()
    n_other = (mapped["fuel_type_resolved"] == "OTHER").sum()
    print(f"\nNamed BMU fuel type coverage ({len(mapped):,} BMUs):")
    print(f"  BMUFuelTyp.xlsx      : {n_from_register:,}")
    print(f"  Elexon fuelType      : {n_from_elexon:,}")
    print(f"  Inferred (name/party): {n_inferred:,}")
    print(f"  Fallback OTHER       : {n_other:,}")
    print(f"  GAS->CCGT/OCGT promoted (Elexon sub-type): {n_promoted:,}")
    print(f"  Sibling-resolved overrides               : {n_sibling:,}")

    # ── 5. Map numeric BMUs ───────────────────────────────────────────────────
    numeric = unique_bmus[unique_bmus["is_numeric_id"]].copy()

    # Numeric BMUs with a resolved_bmu_id (e.g. I_I* interconnectors from DISBSAD)
    numeric_with_id = numeric[numeric["resolved_bmu_id"].notna()].copy()
    numeric_no_id = numeric[numeric["resolved_bmu_id"].isna()].copy()

    numeric_with_id = numeric_with_id.merge(
        reg[["elexonBmUnit", "bmUnitType", "leadPartyName", "leadPartyId"]],
        left_on="resolved_bmu_id", right_on="elexonBmUnit", how="left",
    )
    # bmUnitType='I' → interconnector; anything else unresolved → UNKNOWN_NUMERIC
    numeric_with_id["fuel_type_resolved"] = numeric_with_id["bmUnitType"].apply(
        lambda t: "INTERCONNECTOR" if t == "I" else "UNKNOWN_NUMERIC"
    )

    # Numeric BMUs with no resolved ID: infer from leadPartyId via DISBSAD party map
    numeric_no_id = numeric_no_id.merge(
        stack[stack["is_numeric_id"] & stack["resolved_bmu_id"].isna()]
        [["id", "disbsad_partyId"]]
        .drop_duplicates(),
        left_on="resolved_bmu_id", right_on="id", how="left",
    ) if "disbsad_partyId" in stack.columns else numeric_no_id.assign(disbsad_partyId=pd.NA)

    numeric_no_id["fuel_type_resolved"] = (
        numeric_no_id.get("disbsad_partyId", pd.Series(dtype=str))
        .map(PARTY_FUEL_MAP)
        .fillna("UNKNOWN_NUMERIC")
    )

    numeric = pd.concat([numeric_with_id, numeric_no_id], ignore_index=True)

    n_interconnectors = (numeric["fuel_type_resolved"] == "INTERCONNECTOR").sum()
    print(f"\nNumeric BMU resolution:")
    print(f"  Interconnectors identified : {n_interconnectors:,}")
    print(f"  Still unknown              : {(numeric['fuel_type_resolved'] == 'UNKNOWN_NUMERIC').sum():,}")

    full_map = pd.concat([mapped, numeric], ignore_index=True).sort_values("resolved_bmu_id")

    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    full_map.to_csv(OUT_MAP, index=False)
    print(f"\nSaved BMU map -> {OUT_MAP}  ({len(full_map):,} rows)")

    # ── 6. Summary by resolved fuel type ─────────────────────────────────────
    print("\nFuel type breakdown (all BMUs in stack):")
    print(
        full_map.groupby("fuel_type_resolved", dropna=False)
        .size()
        .rename("n_bmus")
        .sort_values(ascending=False)
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
