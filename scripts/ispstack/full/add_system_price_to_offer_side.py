from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path("data/processed/full_2023_2025/ispstack_marginal_action_2023_2025")
OFFER_CSV = BASE / "marginal_action_sp_2023_2025_offer_side.csv"
OFFER_PARQUET = BASE / "marginal_action_sp_2023_2025_offer_side.parquet"
OUTPUT_CSV = BASE / "marginal_action_sp_2023_2025_offer_side_with_system_price.csv"
OUTPUT_PARQUET = BASE / "marginal_action_sp_2023_2025_offer_side_with_system_price.parquet"
MAIN_WITH_PRICE_CSV = BASE / "marginal_action_sp_2023_2025_modelling_identity_main_with_system_price.csv"
SYSTEM_CSV = Path("data/processed/full_2023_2025/fundamentals/system_price_niv_2023_2025.csv")
AUDIT_JSON = Path("data/diagnostics/audits/add_system_price_to_offer_side_audit.json")

KEYS = ["settlementDate", "settlementPeriod"]
PRICE_COLS = [
    "systemBuyPrice",
    "systemSellPrice",
    "systemPrice",
    "netImbalanceVolume",
    "systemLongShort",
]


def main() -> None:
    offer_df = pd.read_csv(OFFER_CSV)
    system_df = pd.read_csv(SYSTEM_CSV, usecols=KEYS + PRICE_COLS)

    duplicate_offer_keys = int(offer_df.duplicated(KEYS).sum())
    duplicate_system_keys = int(system_df.duplicated(KEYS).sum())

    system_lookup = system_df.drop_duplicates(KEYS)
    offer_without_price = offer_df.drop(columns=[c for c in PRICE_COLS if c in offer_df.columns])
    merged = offer_without_price.merge(system_lookup, on=KEYS, how="left", validate="one_to_one")

    missing_system_price = int(merged["systemPrice"].isna().sum())

    merged.to_csv(OUTPUT_CSV, index=False)
    merged.to_parquet(OUTPUT_PARQUET, index=False)

    audit = {
        "offer_rows": int(len(offer_df)),
        "system_rows": int(len(system_df)),
        "offer_duplicate_keys": duplicate_offer_keys,
        "system_duplicate_keys": duplicate_system_keys,
        "missing_systemPrice_after_merge": missing_system_price,
        "columns_added": PRICE_COLS,
        "source_offer_csv": str(OFFER_CSV),
        "source_offer_parquet": str(OFFER_PARQUET),
        "source_system_csv": str(SYSTEM_CSV),
        "output_csv": str(OUTPUT_CSV),
        "output_parquet": str(OUTPUT_PARQUET),
    }

    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
