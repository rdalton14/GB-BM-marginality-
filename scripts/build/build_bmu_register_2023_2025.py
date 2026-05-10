from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

BMU_MAP_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "full_2023_2025" / "bid_offer_stack_2023_2025"
    / "bid_offer_stack_energy_2023_2025_bmu_map.csv"
)
BMU_FUEL_TYP_PATH = PROJECT_ROOT / "data" / "raw" / "reference" / "BMUFuelTyp.xlsx"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "reference" / "bmu_register_2023_2025.csv"

RENAME = {
    "resolved_bmu_id":    "elexon_bmu_id",
    "bmUnitName":         "bmu_name",
    "fuel_type_resolved": "fuel_type",
    "leadPartyName":      "lead_party",
    "bmUnitType":         "bmu_type",
    "generationCapacity": "gen_capacity_mw",
    "gspGroupName":       "gsp_group",
    "is_numeric_id":      "is_numeric_id",
    "identity_source":    "identity_source",
}

# Abbreviated label appended to family_id for mixed-fuel families
FUEL_ABBREV: dict[str, str] = {
    "BATTERY":       "B",
    "CCGT":          "CC",
    "OCGT":          "OC",
    "GAS":           "G",
    "DIESEL":        "D",
    "WIND":          "W",
    "NPSHYD":        "HY",
    "PS":            "PS",
    "BIOMASS":       "BIO",
    "LOAD RESPONSE": "LR",
    "INTERCONNECTOR":"INT",
    "OTHER":         "OTH",
}

# BMU ID 2-letter prefix (after I_I) → interconnector name
# D suffix = foreign/Denmark side; G suffix = GB side
BMU_PREFIX_IC_NAME: dict[str, str] = {
    "2D": "IFA2",     "2G": "IFA2",
    "BD": "BritNed",  "BG": "BritNed",
    "FD": "IFA",      "FG": "IFA",
    "LD": "Eleclink", "LG": "Eleclink",
    "ND": "NEMO",     "NG": "NEMO",
    "VD": "Viking",   "VG": "Viking",
}

# BMRS interconnector code → proper interconnector name
BMRS_IC_NAME: dict[str, str] = {
    "INTFR":    "IFA",
    "INTIFA2":  "IFA2",
    "INTNED":   "BritNed",
    "INTNEM":   "NEMO",
    "INTELEC":  "Eleclink",
    "INTVKL":   "Viking",
    "INTIRL":   "Moyle",
    "INTEW":    "EastWest",
    "INTNSL":   "NSL",
    "INTGRNL":  "Greenlink",
}


def extract_family(bmu_id: str) -> str:
    """Strip trailing hyphen-digits or bare digits to get the plant family prefix."""
    return re.sub(r"-?\d+$", "", str(bmu_id))


def load_ic_bmrs_map() -> dict[str, str]:
    """Return elexon_bmu_id -> interconnector name for all I_I* BMUs."""
    ft = pd.read_excel(BMU_FUEL_TYP_PATH)[["SETT UNIT ID", "BMRS FUEL TYPE"]].dropna(subset=["SETT UNIT ID"])
    ft["SETT UNIT ID"] = ft["SETT UNIT ID"].astype(str).str.strip()
    ft = ft[ft["SETT UNIT ID"].str.startswith("I_")]
    ft["ic_name"] = ft["BMRS FUEL TYPE"].map(BMRS_IC_NAME)
    ft = ft.dropna(subset=["ic_name"]).drop_duplicates("SETT UNIT ID")
    return dict(zip(ft["SETT UNIT ID"], ft["ic_name"]))


def main() -> None:
    bmu_map = pd.read_csv(BMU_MAP_PATH)

    reg = bmu_map[list(RENAME)].rename(columns=RENAME).copy()
    reg = reg.dropna(subset=["elexon_bmu_id"]).copy()
    reg.insert(1, "family_id", reg["elexon_bmu_id"].map(extract_family))

    # ── 1. Resolve interconnector fuel_type to specific link name ─────────────
    ic_map = load_ic_bmrs_map()
    ic_mask = reg["fuel_type"] == "INTERCONNECTOR"
    reg.loc[ic_mask, "fuel_type"] = reg.loc[ic_mask, "elexon_bmu_id"].map(ic_map).fillna("INTERCONNECTOR")

    n_ic_named = int(reg.loc[ic_mask, "fuel_type"].ne("INTERCONNECTOR").sum())
    print(f"Interconnectors named from BMRS: {n_ic_named} / {ic_mask.sum()}")

    # Fallback: infer name from 2-letter BMU ID prefix for any still-unnamed ICs
    still_unnamed_idx = reg.index[ic_mask & (reg["fuel_type"] == "INTERCONNECTOR")]
    if len(still_unnamed_idx):
        prefix_series = reg.loc[still_unnamed_idx, "elexon_bmu_id"].str.extract(r"^I_I(..)")[0]
        inferred = prefix_series.map(BMU_PREFIX_IC_NAME)
        resolved_idx = inferred.dropna().index
        reg.loc[resolved_idx, "fuel_type"] = inferred.loc[resolved_idx]
        print(f"Interconnectors named from BMU ID prefix: {len(resolved_idx)} / {len(still_unnamed_idx)}")

    print(reg.loc[ic_mask, ["elexon_bmu_id", "fuel_type"]].groupby("fuel_type").size().rename("n").to_string())
    print()

    # ── 2. Suffix family_id with fuel type abbrev for mixed families ──────────
    family_types = reg.groupby("family_id")["fuel_type"].nunique()
    mixed_families = set(family_types[family_types > 1].index)

    def make_family_id(row: pd.Series) -> str:
        if row["family_id"] not in mixed_families:
            return row["family_id"]
        abbrev = FUEL_ABBREV.get(row["fuel_type"], "OTH")
        return f"{row['family_id']}_{abbrev}"

    reg["family_id"] = reg.apply(make_family_id, axis=1)

    n_mixed_bmus = (
        reg["family_id"] != reg["elexon_bmu_id"].map(extract_family)
    ).sum()
    print(f"Mixed-family BMUs with suffixed family_id: {n_mixed_bmus}")
    print()

    reg = reg.sort_values(["family_id", "elexon_bmu_id"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(OUT_PATH, index=False)

    print(f"BMU register saved -> {OUT_PATH}")
    print(f"  Rows            : {len(reg):,}")
    print(f"  Unique BMUs     : {reg['elexon_bmu_id'].nunique():,}")
    print(f"  Unique families : {reg['family_id'].nunique():,}")
    print()
    print("Fuel type breakdown:")
    print(
        reg.groupby("fuel_type", dropna=False)
        .agg(n_bmus=("elexon_bmu_id", "count"), n_families=("family_id", "nunique"))
        .sort_values("n_bmus", ascending=False)
        .to_string()
    )
    print()
    print("Mixed families (sample):")
    mixed_fam_summary = (
        reg[reg["family_id"].apply(lambda f: f not in set(family_types[family_types == 1].index)
                                             and not f.startswith("I_"))]
        .groupby("family_id")
        .agg(n=("elexon_bmu_id", "count"), fuel_type=("fuel_type", "first"))
        .sort_values("family_id")
    )
    print(mixed_fam_summary.to_string())


if __name__ == "__main__":
    main()
