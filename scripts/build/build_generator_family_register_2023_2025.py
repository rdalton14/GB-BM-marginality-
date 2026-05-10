from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTER_CSV = PROJECT_ROOT / "data" / "interim" / "ispstack" / "marginal_bmu_supplementary_register_2023_2025.csv"

OUT_REGISTER_CSV = PROJECT_ROOT / "data" / "interim" / "ispstack" / "generator_family_register_2023_2025.csv"
OUT_REGISTER_PARQUET = PROJECT_ROOT / "data" / "interim" / "ispstack" / "generator_family_register_2023_2025.parquet"

AUDIT_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "generator_family_register_2023_2025_audit.csv"
REVIEW_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "generator_family_register_2023_2025_review_queue.csv"
SUMMARY_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "generator_family_register_2023_2025_summary.csv"


def family_stem_from_bmu(bmu_id: object) -> str:
    if pd.isna(bmu_id):
        return ""
    s = str(bmu_id).strip().upper()
    s = re.sub(r"^[A-Z0-9]+_+", "", s)
    s = re.sub(r"[-_]\d+$", "", s)
    return s


def generator_id_from_stem(stem: str) -> str:
    return f"GEN_{stem}" if stem else ""


def tech_slug(tech: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(tech).strip().upper())


def has_clean_sister_pattern(ids: list[str]) -> bool:
    if len(ids) < 2:
        return False
    ok = 0
    for bmu_id in ids:
        s = str(bmu_id).strip().upper()
        if re.search(r"[-_]\d+$", s):
            ok += 1
    return ok == len(ids)


def classify_family(member_ids: list[str], tech_set: list[str]) -> tuple[str, str, int, str]:
    if len(member_ids) == 1:
        return "keep_separate", "singleton", 0, "Single BMU family"

    if len(tech_set) > 1:
        return "split_by_tech", "mixed_tech_family_split", 0, "Family has multiple technology labels, so split into separate generator families by tech"

    if has_clean_sister_pattern(member_ids):
        return "auto_collapse", "clean_sister_suffix_same_tech", 0, "Multi-BMU family with consistent tech and numeric sister suffixes"

    return "auto_collapse", "close_match_same_tech", 0, "Multi-BMU family with consistent tech and closely related IDs"


print("Loading supplementary marginal BMU register ...")
register = pd.read_csv(REGISTER_CSV)
register["elexon_bmu_id"] = register["elexon_bmu_id"].astype(str).str.strip()
register["reg_fuel_type"] = register["reg_fuel_type"].astype(str).str.strip()
register["family_stem_candidate"] = register["elexon_bmu_id"].map(family_stem_from_bmu)
register["generator_id_candidate"] = register["family_stem_candidate"].map(generator_id_from_stem)
register["generator_label_candidate"] = register["family_stem_candidate"]

print("Building family-level audit ...")
family_rows: list[dict[str, object]] = []

for stem, grp in register.groupby("family_stem_candidate", dropna=False):
    member_ids = sorted(grp["elexon_bmu_id"].tolist())
    tech_set = sorted(grp["reg_fuel_type"].dropna().unique().tolist())
    collapse_decision, collapse_rule, needs_manual_review, notes = classify_family(member_ids, tech_set)
    family_rows.append(
        {
            "family_stem_candidate": stem,
            "generator_id_candidate": generator_id_from_stem(stem),
            "generator_label_candidate": stem,
            "family_size": len(member_ids),
            "member_bmus": "; ".join(member_ids),
            "tech_set": ", ".join(tech_set),
            "family_tech_consistent": int(len(tech_set) <= 1),
            "collapse_decision": collapse_decision,
            "collapse_rule": collapse_rule,
            "needs_manual_review": needs_manual_review,
            "notes": notes,
        }
    )

family_audit = pd.DataFrame(family_rows).sort_values(["needs_manual_review", "family_size", "family_stem_candidate"], ascending=[False, False, True]).reset_index(drop=True)

print("Projecting family decisions to BMU-level register ...")
family_cols = [
    "family_stem_candidate",
    "family_size",
    "member_bmus",
    "tech_set",
    "family_tech_consistent",
    "collapse_decision",
    "collapse_rule",
    "needs_manual_review",
    "notes",
]
generator_register = register.merge(family_audit[family_cols], on="family_stem_candidate", how="left")
mixed_mask = generator_register["collapse_decision"].eq("split_by_tech")
generator_register["generator_id_final"] = generator_register["generator_id_candidate"]
generator_register["generator_label_final"] = generator_register["generator_label_candidate"]
generator_register.loc[mixed_mask, "generator_id_final"] = generator_register.loc[mixed_mask].apply(
    lambda x: f"{x['generator_id_candidate']}_{tech_slug(x['reg_fuel_type'])}", axis=1
)
generator_register.loc[mixed_mask, "generator_label_final"] = generator_register.loc[mixed_mask].apply(
    lambda x: f"{x['generator_label_candidate']} ({x['reg_fuel_type']})", axis=1
)
generator_register["source"] = "bmu_id_pattern"
generator_register.loc[mixed_mask, "source"] = "bmu_id_pattern_split_by_tech"

generator_register = generator_register[
    [
        "elexon_bmu_id",
        "reg_fuel_type",
        "family_stem_candidate",
        "generator_id_candidate",
        "generator_label_candidate",
        "family_size",
        "member_bmus",
        "tech_set",
        "family_tech_consistent",
        "collapse_decision",
        "collapse_rule",
        "needs_manual_review",
        "generator_id_final",
        "generator_label_final",
        "notes",
        "source",
    ]
].sort_values(["family_size", "family_stem_candidate", "elexon_bmu_id"], ascending=[False, True, True]).reset_index(drop=True)

review_queue = family_audit.loc[family_audit["needs_manual_review"] == 1].copy()

summary = pd.DataFrame(
    [
        {"metric": "total_bmus", "value": register["elexon_bmu_id"].nunique()},
        {"metric": "unique_candidate_families", "value": family_audit["family_stem_candidate"].nunique()},
        {"metric": "singleton_families", "value": int((family_audit["family_size"] == 1).sum())},
        {"metric": "auto_collapse_families", "value": int((family_audit["collapse_decision"] == "auto_collapse").sum())},
        {"metric": "split_by_tech_families", "value": int((family_audit["collapse_decision"] == "split_by_tech").sum())},
        {"metric": "review_required_families", "value": int((family_audit["collapse_decision"] == "review_required").sum())},
        {"metric": "auto_collapse_bmus", "value": int(generator_register.loc[generator_register["collapse_decision"] == "auto_collapse", "elexon_bmu_id"].nunique())},
        {"metric": "split_by_tech_bmus", "value": int(generator_register.loc[generator_register["collapse_decision"] == "split_by_tech", "elexon_bmu_id"].nunique())},
        {"metric": "review_required_bmus", "value": int(generator_register.loc[generator_register["needs_manual_review"] == 1, "elexon_bmu_id"].nunique())},
        {"metric": "largest_family_size", "value": int(family_audit["family_size"].max())},
    ]
)

print("Writing outputs ...")
OUT_REGISTER_CSV.parent.mkdir(parents=True, exist_ok=True)
AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

generator_register.to_csv(OUT_REGISTER_CSV, index=False)
generator_register.to_parquet(OUT_REGISTER_PARQUET, index=False)
family_audit.to_csv(AUDIT_CSV, index=False)
review_queue.to_csv(REVIEW_CSV, index=False)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"Saved register -> {OUT_REGISTER_CSV}")
print(f"Saved register -> {OUT_REGISTER_PARQUET}")
print(f"Saved audit    -> {AUDIT_CSV}")
print(f"Saved review   -> {REVIEW_CSV}")
print(f"Saved summary  -> {SUMMARY_CSV}")
print()
print(summary.to_string(index=False))
