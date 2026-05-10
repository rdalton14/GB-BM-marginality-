from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PANEL = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_iso.csv"
REFERENCE = PROJECT_ROOT / "data" / "interim" / "ispstack" / "bmu_reference_canonical_2023_2025.csv"

OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_refined_stack.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_refined_stack.parquet"

AUDIT_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "master_panel_2023_2025_cleaned_refined_stack_reassignment_audit.csv"
SUMMARY_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "master_panel_2023_2025_cleaned_refined_stack_summary.csv"


def family_stem(bmu_id: object) -> str:
    if pd.isna(bmu_id):
        return ""
    stem = str(bmu_id).strip().upper()
    stem = re.sub(r"^[A-Z0-9]+_", "", stem)
    stem = re.sub(r"-\d+$", "", stem)
    return stem


def unified_bucket(tech: object) -> str:
    if pd.isna(tech):
        return "Unknown"
    tech = str(tech).strip().upper()
    if tech in {"GAS", "CCGT", "OCGT"}:
        return "Gas"
    if tech == "BATTERY":
        return "BESS"
    return "Other"


print("Loading cleaned stack panel ...")
panel = pd.read_csv(INPUT_PANEL, parse_dates=["settlementDate"])
print(f"  rows: {len(panel):,}")

print("Loading canonical BMU reference ...")
ref = pd.read_csv(REFERENCE)
ref["family_stem"] = ref["id"].map(family_stem)

print("Building conservative family-level gas refinement map ...")
family_rows = []
refine_map: dict[str, str] = {}

for stem, grp in ref.groupby("family_stem", dropna=False):
    labels = sorted(set(grp["tech_final"].dropna().astype(str)))
    ids = sorted(set(grp["id"].dropna().astype(str)))
    if "GAS" not in labels:
        continue

    specific = sorted({x for x in labels if x in {"CCGT", "OCGT"}})
    if len(specific) == 1:
        target = specific[0]
        refine_map[stem] = target
        decision = "reassign"
        rationale = f"GAS family has unambiguous sibling label {target}"
    elif len(specific) > 1:
        target = "KEEP_GAS"
        decision = "ambiguous_keep"
        rationale = "Family contains both CCGT and OCGT siblings"
    else:
        target = "KEEP_GAS"
        decision = "no_specific_keep"
        rationale = "Family contains GAS only with no specific sibling evidence"

    family_rows.append(
        {
            "family_stem": stem,
            "decision": decision,
            "target_label": target,
            "family_labels": ", ".join(labels),
            "n_bmus": len(ids),
            "family_ids": "; ".join(ids),
            "rationale": rationale,
        }
    )

audit = pd.DataFrame(family_rows).sort_values(["decision", "family_stem"]).reset_index(drop=True)

print("Applying refinement to marginal technology ...")
panel["marginal_family_stem"] = panel["marginal_bmu"].map(family_stem)
panel["marginal_tech_refined"] = panel["marginal_tech"]
panel["marginal_tech_refined_rule"] = "original"

gas_mask = panel["marginal_tech"].eq("GAS")
reassign_mask = gas_mask & panel["marginal_family_stem"].isin(refine_map)

panel.loc[reassign_mask, "marginal_tech_refined"] = panel.loc[reassign_mask, "marginal_family_stem"].map(refine_map)
panel.loc[reassign_mask, "marginal_tech_refined_rule"] = "family_reassigned_from_gas"

panel["marginal_tech_refined_changed"] = panel["marginal_tech_refined"].ne(panel["marginal_tech"]).astype(int)
panel["marginal_tech_refined_unified"] = panel["marginal_tech_refined"].map(unified_bucket)
panel["gas_marginal_refined"] = panel["marginal_tech_refined_unified"].eq("Gas").astype(int)

summary = pd.DataFrame(
    [
        {"metric": "rows_total", "value": len(panel)},
        {"metric": "gas_rows_original", "value": int(gas_mask.sum())},
        {"metric": "gas_rows_reassigned", "value": int(reassign_mask.sum())},
        {
            "metric": "gas_rows_reassigned_pct_of_original_gas",
            "value": round(100 * reassign_mask.sum() / gas_mask.sum(), 4) if gas_mask.sum() else 0.0,
        },
        {"metric": "families_reassigned", "value": int(len(refine_map))},
        {"metric": "families_audited_with_gas", "value": int(len(audit))},
    ]
)

print("Writing outputs ...")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

panel.to_csv(OUTPUT_CSV, index=False)
panel.to_parquet(OUTPUT_PARQUET, index=False)
audit.to_csv(AUDIT_CSV, index=False)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"Saved panel  -> {OUTPUT_CSV}")
print(f"Saved panel  -> {OUTPUT_PARQUET}")
print(f"Saved audit  -> {AUDIT_CSV}")
print(f"Saved audit  -> {SUMMARY_CSV}")
print()
print("Refinement summary:")
print(summary.to_string(index=False))
