from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "marginal_action_sp_2023_2025_modelling_identity_main.parquet"
)
SYSTEM_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "fundamentals"
    / "system_price_niv_2023_2025.csv"
)
OUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "marginal_action_sp_2023_2025_modelling_identity_main_with_system_price.parquet"
)
AUDIT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "align_system_price_to_2023_2025_main_model_df_audit.json"

KEYS = ["settlementDate", "settlementPeriod"]


def main() -> None:
    model_df = pd.read_parquet(MODEL_PATH)
    system_df = pd.read_csv(
        SYSTEM_PATH,
        usecols=KEYS + ["systemPrice", "systemBuyPrice", "systemSellPrice", "netImbalanceVolume", "systemLongShort"],
    )

    model_dup = int(model_df.duplicated(KEYS).sum())
    system_dup = int(system_df.duplicated(KEYS).sum())
    if model_dup != 0:
        raise ValueError(f"Main model df has {model_dup} duplicate settlement-period keys.")
    if system_dup != 0:
        raise ValueError(f"System-price source has {system_dup} duplicate settlement-period keys.")

    aligned = model_df.merge(system_df, on=KEYS, how="left", validate="one_to_one")
    missing_system_price = int(aligned["systemPrice"].isna().sum())
    if missing_system_price != 0:
        raise ValueError(f"Aligned dataframe still has {missing_system_price} missing systemPrice values.")

    audit = {
        "model_rows": int(len(model_df)),
        "system_rows": int(len(system_df)),
        "aligned_rows": int(len(aligned)),
        "model_duplicate_keys": model_dup,
        "system_duplicate_keys": system_dup,
        "missing_system_price_after_merge": missing_system_price,
        "date_min": str(aligned["settlementDate"].min()),
        "date_max": str(aligned["settlementDate"].max()),
        "system_price_mean": float(aligned["systemPrice"].mean()),
        "system_price_median": float(aligned["systemPrice"].median()),
    }

    aligned.to_parquet(OUT_PATH, index=False)
    aligned.to_csv(OUT_PATH.with_suffix(".csv"), index=False)
    with AUDIT_PATH.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("=" * 72)
    print("System Price Aligned To 2023-2025 Main Model DF")
    print("=" * 72)
    print(f"Rows aligned                  : {len(aligned):,}")
    print(f"Missing systemPrice after join: {missing_system_price:,}")
    print(f"Output                        : {OUT_PATH}")
    print(f"Audit                         : {AUDIT_PATH}")


if __name__ == "__main__":
    main()
