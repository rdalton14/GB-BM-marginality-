from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_refined_stack.csv"

OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "ispstack" / "marginal_bmu_supplementary_register_2023_2025.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "ispstack" / "marginal_bmu_supplementary_register_2023_2025.parquet"
MULTITECH_AUDIT_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "marginal_bmu_supplementary_register_2023_2025_multitech_audit.csv"
SUMMARY_CSV = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "marginal_bmu_supplementary_register_2023_2025_summary.csv"


print("Loading refined stack panel ...")
df = pd.read_csv(
    INPUT_CSV,
    usecols=[
        "marginal_bmu",
        "marginal_tech_refined",
        "marginal_family_stem",
        "marginal_tech_refined_rule",
    ],
)

df = df.dropna(subset=["marginal_bmu"]).copy()
df["marginal_bmu"] = df["marginal_bmu"].astype(str).str.strip()
df["marginal_tech_refined"] = df["marginal_tech_refined"].fillna("UNKNOWN").astype(str).str.strip()
df["marginal_family_stem"] = df["marginal_family_stem"].fillna("").astype(str).str.strip()
df["marginal_tech_refined_rule"] = df["marginal_tech_refined_rule"].fillna("unknown").astype(str).str.strip()

print("Building BMU-tech counts ...")
pair_counts = (
    df.groupby(
        ["marginal_bmu", "marginal_tech_refined", "marginal_family_stem", "marginal_tech_refined_rule"],
        dropna=False,
    )
    .size()
    .reset_index(name="marginal_sp_count")
)

print("Resolving one row per BMU ...")
pair_counts = pair_counts.sort_values(
    ["marginal_bmu", "marginal_sp_count", "marginal_tech_refined"],
    ascending=[True, False, True],
).reset_index(drop=True)

dominant = pair_counts.drop_duplicates(subset=["marginal_bmu"], keep="first").copy()
tech_counts = pair_counts.groupby("marginal_bmu")["marginal_tech_refined"].nunique().reset_index(name="n_unique_marginal_techs")
dominant = dominant.merge(tech_counts, on="marginal_bmu", how="left")
dominant["has_multiple_marginal_techs"] = dominant["n_unique_marginal_techs"].gt(1).astype(int)

dominant = dominant.rename(
    columns={
        "marginal_bmu": "elexon_bmu_id",
        "marginal_tech_refined": "reg_fuel_type",
        "marginal_family_stem": "family_stem",
        "marginal_tech_refined_rule": "tech_assignment_rule",
    }
)

dominant["source"] = "master_panel_2023_2025_cleaned_refined_stack"
dominant = dominant[
    [
        "elexon_bmu_id",
        "reg_fuel_type",
        "family_stem",
        "tech_assignment_rule",
        "marginal_sp_count",
        "n_unique_marginal_techs",
        "has_multiple_marginal_techs",
        "source",
    ]
].sort_values(["reg_fuel_type", "elexon_bmu_id"]).reset_index(drop=True)

print("Building multitech audit ...")
multitech_bmus = tech_counts.loc[tech_counts["n_unique_marginal_techs"] > 1, "marginal_bmu"]
multitech_audit = (
    pair_counts[pair_counts["marginal_bmu"].isin(multitech_bmus)]
    .sort_values(["marginal_bmu", "marginal_sp_count"], ascending=[True, False])
    .rename(columns={"marginal_bmu": "elexon_bmu_id", "marginal_tech_refined": "reg_fuel_type"})
    .reset_index(drop=True)
)

summary = pd.DataFrame(
    [
        {"metric": "rows_in_source_panel", "value": len(df)},
        {"metric": "unique_bmus_in_register", "value": dominant["elexon_bmu_id"].nunique()},
        {"metric": "unique_bmu_tech_pairs", "value": pair_counts[["marginal_bmu", "marginal_tech_refined"]].drop_duplicates().shape[0]},
        {"metric": "bmus_with_multiple_techs", "value": int(dominant["has_multiple_marginal_techs"].sum())},
    ]
)

print("Writing outputs ...")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
MULTITECH_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

dominant.to_csv(OUTPUT_CSV, index=False)
dominant.to_parquet(OUTPUT_PARQUET, index=False)
multitech_audit.to_csv(MULTITECH_AUDIT_CSV, index=False)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"Saved register -> {OUTPUT_CSV}")
print(f"Saved register -> {OUTPUT_PARQUET}")
print(f"Saved audit    -> {MULTITECH_AUDIT_CSV}")
print(f"Saved summary  -> {SUMMARY_CSV}")
print()
print(summary.to_string(index=False))
