from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "marginal_action_sp_2023_2025_identity_unknown_numeric.parquet"
)
OUT_DIR = INPUT_PATH.parent

OUT_PANEL = OUT_DIR / "marginal_action_sp_2023_2025_modelling_identity.parquet"
OUT_PANEL_MAIN = OUT_DIR / "marginal_action_sp_2023_2025_modelling_identity_main.parquet"
OUT_PANEL_ROBUST = OUT_DIR / "marginal_action_sp_2023_2025_modelling_identity_robustness.parquet"
OUT_AUDIT = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_modelling_identity_2023_2025_audit.csv"
OUT_DROP_AUDIT = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_modelling_identity_2023_2025_main_drop_audit.csv"
OUT_SUMMARY = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_modelling_identity_2023_2025_summary.json"


def main() -> None:
    df = pd.read_parquet(INPUT_PATH).copy()

    raw_label = df["marginal_generator_label_final"].fillna("")
    unknown_stack_mask = raw_label.str.startswith("UNKNOWN_STACK_ID_") | raw_label.eq("UNKNOWN_NUMERIC")

    df["marginal_generator_label_model"] = df["marginal_generator_label_final"]
    df.loc[unknown_stack_mask, "marginal_generator_label_model"] = "UNKNOWN_STACK"

    df["marginal_generator_id_model"] = df["marginal_generator_id_final"]
    df.loc[unknown_stack_mask, "marginal_generator_id_model"] = "GEN_UNKNOWN_STACK"

    df["marginal_bmu_model"] = df["marginal_bmu_final"]
    df.loc[unknown_stack_mask, "marginal_bmu_model"] = "UNKNOWN_STACK"

    df["marginal_identity_placeholder_raw"] = unknown_stack_mask
    df["marginal_identity_model_rule"] = "keep_final_identity"
    df.loc[unknown_stack_mask, "marginal_identity_model_rule"] = "pool_unknown_stack_ids"

    df["marginal_identity_main_keep"] = ~unknown_stack_mask
    df["marginal_identity_main_rule"] = "keep_non_placeholder_generator"
    df.loc[unknown_stack_mask, "marginal_identity_main_rule"] = "drop_unknown_stack"

    audit = (
        df.groupby(["marginal_side_winner", "marginal_identity_model_rule"])
        .agg(
            rows=("settlementDate", "size"),
            unique_final_labels=("marginal_generator_label_final", "nunique"),
            unique_model_labels=("marginal_generator_label_model", "nunique"),
        )
        .reset_index()
        .sort_values(["marginal_side_winner", "marginal_identity_model_rule"])
    )

    drop_audit = (
        df.groupby(["marginal_side_winner", "marginal_identity_main_keep"])
        .agg(
            rows=("settlementDate", "size"),
            unique_final_labels=("marginal_generator_label_final", "nunique"),
        )
        .reset_index()
        .sort_values(["marginal_side_winner", "marginal_identity_main_keep"], ascending=[True, False])
    )

    df_main = df[df["marginal_identity_main_keep"]].copy()
    df_robust = df.copy()

    summary = {
        "rows_total": int(len(df)),
        "placeholder_rows_total": int(unknown_stack_mask.sum()),
        "placeholder_rows_share": float(unknown_stack_mask.mean()),
        "placeholder_rows_by_side": {
            str(k): int(v) for k, v in df.loc[unknown_stack_mask, "marginal_side_winner"].value_counts(dropna=False).to_dict().items()
        },
        "top_placeholder_labels": {
            str(k): int(v) for k, v in df.loc[unknown_stack_mask, "marginal_generator_label_final"].value_counts().head(20).to_dict().items()
        },
        "unique_generator_labels_final": int(df["marginal_generator_label_final"].nunique(dropna=True)),
        "unique_generator_labels_model": int(df["marginal_generator_label_model"].nunique(dropna=True)),
        "main_rows_total": int(len(df_main)),
        "main_rows_dropped": int(unknown_stack_mask.sum()),
        "main_rows_dropped_share": float(unknown_stack_mask.mean()),
        "main_rows_by_side": {
            str(k): int(v) for k, v in df_main["marginal_side_winner"].value_counts(dropna=False).to_dict().items()
        },
        "robust_rows_total": int(len(df_robust)),
    }

    df.to_parquet(OUT_PANEL, index=False)
    df.to_csv(OUT_PANEL.with_suffix(".csv"), index=False)
    df_main.to_parquet(OUT_PANEL_MAIN, index=False)
    df_main.to_csv(OUT_PANEL_MAIN.with_suffix(".csv"), index=False)
    df_robust.to_parquet(OUT_PANEL_ROBUST, index=False)
    df_robust.to_csv(OUT_PANEL_ROBUST.with_suffix(".csv"), index=False)
    audit.to_csv(OUT_AUDIT, index=False)
    drop_audit.to_csv(OUT_DROP_AUDIT, index=False)
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("2023-2025 ISPSTACK Modelling Identity Panel Built")
    print("=" * 72)
    print(f"Rows total              : {len(df):,}")
    print(f"Placeholder rows pooled : {int(unknown_stack_mask.sum()):,}")
    print(f"Main rows kept          : {len(df_main):,}")
    print(f"Main rows dropped       : {int(unknown_stack_mask.sum()):,}")
    print(f"Unique labels (final)   : {summary['unique_generator_labels_final']:,}")
    print(f"Unique labels (model)   : {summary['unique_generator_labels_model']:,}")
    print(f"Panel                   : {OUT_PANEL}")
    print(f"Main panel              : {OUT_PANEL_MAIN}")
    print(f"Robustness panel        : {OUT_PANEL_ROBUST}")
    print(f"Audit                   : {OUT_AUDIT}")


if __name__ == "__main__":
    main()
