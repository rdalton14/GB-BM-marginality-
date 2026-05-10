from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STACK_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_refined_stack_manual_register.csv"
GENERATOR_REGISTER_CSV = PROJECT_ROOT / "data" / "interim" / "ispstack" / "generator_family_register_2023_2025.csv"

OUT_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_generator_collapsed.csv"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_generator_collapsed.parquet"

CHANGE_AUDIT_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "master_panel_2023_2025_generator_collapsed_changes.csv"
SUMMARY_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "master_panel_2023_2025_generator_collapsed_summary.csv"


print("Loading manual-register-applied stack panel ...")
stack = pd.read_csv(STACK_CSV, parse_dates=["settlementDate"])

print("Loading generator family register ...")
generator_register = pd.read_csv(GENERATOR_REGISTER_CSV)
generator_register["elexon_bmu_id"] = generator_register["elexon_bmu_id"].astype(str).str.strip()

dup_count = int(generator_register.duplicated(subset=["elexon_bmu_id"]).sum())
if dup_count:
    raise ValueError(f"Generator family register has {dup_count} duplicate BMU ids; resolve before applying.")

lookup = generator_register.rename(
    columns={
        "elexon_bmu_id": "marginal_bmu",
        "generator_id_final": "marginal_generator_id",
        "generator_label_final": "marginal_generator_label",
        "collapse_decision": "marginal_generator_collapse_decision",
        "collapse_rule": "marginal_generator_collapse_rule",
        "family_stem_candidate": "marginal_generator_family_stem",
        "reg_fuel_type": "marginal_generator_register_tech",
    }
)

print("Applying generator-family mapping across stack ...")
panel = stack.merge(
    lookup[
        [
            "marginal_bmu",
            "marginal_generator_id",
            "marginal_generator_label",
            "marginal_generator_collapse_decision",
            "marginal_generator_collapse_rule",
            "marginal_generator_family_stem",
            "marginal_generator_register_tech",
        ]
    ],
    on="marginal_bmu",
    how="left",
)

panel["marginal_generator_id"] = panel["marginal_generator_id"].fillna(panel["marginal_bmu"])
panel["marginal_generator_label"] = panel["marginal_generator_label"].fillna(panel["marginal_bmu"])
panel["marginal_generator_collapse_decision"] = panel["marginal_generator_collapse_decision"].fillna("unmapped_keep_bmu")
panel["marginal_generator_collapse_rule"] = panel["marginal_generator_collapse_rule"].fillna("unmapped_keep_bmu")
panel["marginal_generator_family_stem"] = panel["marginal_generator_family_stem"].fillna("")
panel["marginal_generator_register_tech"] = panel["marginal_generator_register_tech"].fillna(panel["marginal_tech_final"])

panel["marginal_generator_changed_from_bmu"] = panel["marginal_generator_id"].ne(panel["marginal_bmu"]).astype(int)

change_audit = (
    panel.loc[
        panel["marginal_generator_changed_from_bmu"] == 1,
        [
            "settlementDate",
            "settlementPeriod",
            "marginal_bmu",
            "marginal_tech_final",
            "marginal_generator_id",
            "marginal_generator_label",
            "marginal_generator_collapse_decision",
            "marginal_generator_collapse_rule",
            "marginal_generator_family_stem",
        ],
    ]
    .sort_values(["settlementDate", "settlementPeriod", "marginal_bmu"])
    .reset_index(drop=True)
)

summary = pd.DataFrame(
    [
        {"metric": "stack_rows", "value": len(panel)},
        {"metric": "unique_marginal_bmus", "value": panel["marginal_bmu"].nunique()},
        {"metric": "unique_marginal_generators", "value": panel["marginal_generator_id"].nunique()},
        {"metric": "rows_with_generator_collapse", "value": int(panel["marginal_generator_changed_from_bmu"].sum())},
        {"metric": "rows_with_generator_collapse_pct", "value": round(100 * panel["marginal_generator_changed_from_bmu"].mean(), 4)},
        {"metric": "unique_bmus_collapsed", "value": change_audit["marginal_bmu"].nunique()},
    ]
)

print("Writing outputs ...")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
CHANGE_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

panel.to_csv(OUT_CSV, index=False)
panel.to_parquet(OUT_PARQUET, index=False)
change_audit.to_csv(CHANGE_AUDIT_CSV, index=False)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"Saved panel  -> {OUT_CSV}")
print(f"Saved panel  -> {OUT_PARQUET}")
print(f"Saved audit  -> {CHANGE_AUDIT_CSV}")
print(f"Saved audit  -> {SUMMARY_CSV}")
print()
print(summary.to_string(index=False))
