from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPORTANCE_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "offer_side_generator_matrices"
    / "rf_fullsample_strict_filtered"
    / "offer_side_generator_strict_filtered_rf_feature_importance.csv"
)
LOOKUP_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "generator_lookup"
    / "offer_side_generator_label_lookup.csv"
)
OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "offer_side_generator_matrices"
    / "rf_fullsample_strict_filtered"
    / "offer_side_generator_strict_filtered_rf_feature_importance_enriched.csv"
)


def main() -> None:
    importance = pd.read_csv(IMPORTANCE_CSV)
    lookup = pd.read_csv(LOOKUP_CSV)

    enriched = importance.merge(
        lookup,
        left_on="generator_label",
        right_on="generator_label_final",
        how="left",
        validate="many_to_one",
    )

    preferred_cols = [
        "generator_label",
        "feature_importance",
        "importance_share_pct",
        "cumulative_importance_pct",
        "marginal_rows",
        "row_share_pct",
        "generator_ids",
        "bmus",
        "offer_plants",
        "techs",
        "register_plant_names",
        "register_techs",
    ]
    preferred_present = [c for c in preferred_cols if c in enriched.columns]
    ordered = preferred_present + [c for c in enriched.columns if c not in preferred_present]
    enriched = enriched[ordered]
    enriched.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved enriched strict importance file to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
