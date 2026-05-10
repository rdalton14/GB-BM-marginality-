from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STACK_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_refined_stack.csv"
REGISTER_CSV = PROJECT_ROOT / "data" / "interim" / "ispstack" / "marginal_bmu_supplementary_register_2023_2025.csv"

OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_refined_stack_manual_register.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_refined_stack_manual_register.parquet"

CHANGE_AUDIT_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "master_panel_2023_2025_cleaned_refined_stack_manual_register_changes.csv"
SUMMARY_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "master_panel_2023_2025_cleaned_refined_stack_manual_register_summary.csv"


def unified_bucket(tech: object) -> str:
    if pd.isna(tech):
        return "Unknown"
    tech = str(tech).strip().upper()
    if tech in {"GAS", "CCGT", "OCGT"}:
        return "Gas"
    if tech == "BATTERY":
        return "BESS"
    return "Other"


print("Loading stack panel ...")
stack = pd.read_csv(STACK_CSV, parse_dates=["settlementDate"])

print("Loading manual marginal register ...")
register = pd.read_csv(REGISTER_CSV)
register["elexon_bmu_id"] = register["elexon_bmu_id"].astype(str).str.strip()
register["reg_fuel_type"] = register["reg_fuel_type"].astype(str).str.strip()

dup_count = int(register.duplicated(subset=["elexon_bmu_id"]).sum())
if dup_count:
    raise ValueError(f"Manual register has {dup_count} duplicate BMU ids; resolve before applying.")

register_lookup = register.rename(
    columns={
        "elexon_bmu_id": "marginal_bmu",
        "reg_fuel_type": "marginal_tech_manual_register",
        "tech_assignment_rule": "marginal_tech_manual_register_rule",
    }
)

print("Applying manual register across stack ...")
panel = stack.merge(
    register_lookup[["marginal_bmu", "marginal_tech_manual_register", "marginal_tech_manual_register_rule"]],
    on="marginal_bmu",
    how="left",
)

panel["marginal_tech_final"] = panel["marginal_tech_manual_register"].fillna(panel["marginal_tech_refined"])
panel["marginal_tech_final_rule"] = panel["marginal_tech_manual_register"].notna().map(
    {True: "manual_register", False: "fallback_refined_stack"}
)
panel["marginal_tech_final_changed_vs_refined"] = panel["marginal_tech_final"].ne(panel["marginal_tech_refined"]).astype(int)
panel["marginal_tech_final_changed_vs_original"] = panel["marginal_tech_final"].ne(panel["marginal_tech"]).astype(int)
panel["marginal_tech_final_unified"] = panel["marginal_tech_final"].map(unified_bucket)
panel["gas_marginal_final"] = panel["marginal_tech_final_unified"].eq("Gas").astype(int)

change_audit = (
    panel.loc[
        panel["marginal_tech_final_changed_vs_refined"] == 1,
        [
            "settlementDate",
            "settlementPeriod",
            "marginal_bmu",
            "marginal_tech",
            "marginal_tech_refined",
            "marginal_tech_manual_register",
            "marginal_tech_final",
            "marginal_tech_manual_register_rule",
            "marginal_tech_final_rule",
        ],
    ]
    .sort_values(["settlementDate", "settlementPeriod", "marginal_bmu"])
    .reset_index(drop=True)
)

summary = pd.DataFrame(
    [
        {"metric": "stack_rows", "value": len(panel)},
        {"metric": "unique_stack_bmus", "value": panel["marginal_bmu"].nunique()},
        {"metric": "manual_register_bmus", "value": register["elexon_bmu_id"].nunique()},
        {"metric": "rows_with_manual_register_match", "value": int(panel["marginal_tech_manual_register"].notna().sum())},
        {"metric": "rows_changed_vs_refined", "value": int(panel["marginal_tech_final_changed_vs_refined"].sum())},
        {"metric": "rows_changed_vs_original", "value": int(panel["marginal_tech_final_changed_vs_original"].sum())},
        {"metric": "unique_bmus_changed_vs_refined", "value": change_audit["marginal_bmu"].nunique()},
    ]
)

print("Writing outputs ...")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
CHANGE_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

panel.to_csv(OUTPUT_CSV, index=False)
panel.to_parquet(OUTPUT_PARQUET, index=False)
change_audit.to_csv(CHANGE_AUDIT_CSV, index=False)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"Saved panel  -> {OUTPUT_CSV}")
print(f"Saved panel  -> {OUTPUT_PARQUET}")
print(f"Saved audit  -> {CHANGE_AUDIT_CSV}")
print(f"Saved audit  -> {SUMMARY_CSV}")
print()
print(summary.to_string(index=False))
