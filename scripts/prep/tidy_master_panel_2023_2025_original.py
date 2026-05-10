from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025"
AUDITS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "audits"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

CLEANED_CSV = PROCESSED_DIR / "master_panel_2023_2025_cleaned.csv"
CLEANED_PARQUET = PROCESSED_DIR / "master_panel_2023_2025_cleaned.parquet"
CLEANED_ISO_CSV = PROCESSED_DIR / "master_panel_2023_2025_cleaned_iso.csv"
CLEANED_ISO_PARQUET = PROCESSED_DIR / "master_panel_2023_2025_cleaned_iso.parquet"

MISSINGNESS_AUDIT_CSV = AUDITS_DIR / "master_panel_2023_2025_missingness_audit.csv"
IMPUTATION_AUDIT_CSV = AUDITS_DIR / "master_panel_2023_2025_imputation_audit.csv"
KEY_INTEGRITY_AUDIT_CSV = AUDITS_DIR / "master_panel_2023_2025_key_integrity_audit.csv"
VARIABLE_DICTIONARY_CSV = REPORTS_DIR / "master_panel_2023_2025_variable_dictionary.csv"
DATA_NOTE_MD = REPORTS_DIR / "master_panel_2023_2025_data_note.md"


def build_variable_dictionary() -> pd.DataFrame:
    rows = [
        ("settlementDate", "Settlement date of the BM action panel row.", "date", "ISP stack panel", "no", "Original file stored dates in MM/DD/YYYY format."),
        ("settlementPeriod", "Settlement period number within settlementDate.", "integer", "ISP stack panel", "no", "May include DST edge cases in raw system calendars."),
        ("offer_volume_total", "Total accepted offer volume across all offer actions in the SP.", "MWh", "ISP stack raw", "no", "Includes energy and system offers."),
        ("offer_n_bmus", "Number of distinct BMUs with accepted offers in the SP.", "count", "ISP stack raw", "no", ""),
        ("offer_n_energy", "Count of accepted non-SO-flagged offer actions.", "count", "ISP stack raw", "no", "Energy-side offers only."),
        ("offer_n_system", "Count of accepted SO-flagged offer actions.", "count", "ISP stack raw", "no", "System actions only."),
        ("offer_volume_energy", "Accepted offer volume from non-SO-flagged offer actions.", "MWh", "ISP stack raw", "no", "Missing in original file can mean no energy offers in that SP."),
        ("offer_volume_system", "Accepted offer volume from SO-flagged system offer actions.", "MWh", "ISP stack raw", "no", "Missing in original file can mean no system offers in that SP."),
        ("offer_max_price", "Maximum accepted non-repriced energy-offer price used in the original pipeline.", "GBP/MWh", "ISP stack raw", "yes", "Within-day filled in the original pipeline where absent."),
        ("systemPrice", "System price companion field present in the saved original CSV.", "GBP/MWh", "Original saved file", "unknown", "Not created by the current stack-processing script; retained as found in the canonical analysis file."),
        ("bid_volume_total", "Total accepted bid volume across all bid actions in the SP.", "MWh", "ISP stack raw", "no", "Includes energy and system bids."),
        ("bid_n_bmus", "Number of distinct BMUs with accepted bids in the SP.", "count", "ISP stack raw", "no", ""),
        ("bid_n_energy", "Count of accepted non-SO-flagged bid actions.", "count", "ISP stack raw", "no", "Energy-side bids only."),
        ("bid_n_system", "Count of accepted SO-flagged bid actions.", "count", "ISP stack raw", "no", "System actions only."),
        ("bid_volume_energy", "Accepted bid volume from non-SO-flagged bid actions.", "MWh", "ISP stack raw", "no", "Missing in original file can mean no energy bids in that SP."),
        ("bid_volume_system", "Accepted bid volume from SO-flagged system bid actions.", "MWh", "ISP stack raw", "no", "Missing in original file can mean no system bids in that SP."),
        ("bid_min_price", "Minimum accepted energy-bid price in the SP.", "GBP/MWh", "ISP stack raw", "no", "Missing where no energy bid exists."),
        ("bess_bid_volume", "Accepted bid volume attributed to battery BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", "Original file leaves some NaN where no bid-side data exists."),
        ("vol_battery", "Accepted energy-offer volume attributed to battery BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_biomass", "Accepted energy-offer volume attributed to biomass BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_ccgt", "Accepted energy-offer volume attributed to CCGT BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_coal", "Accepted energy-offer volume attributed to coal BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_diesel", "Accepted energy-offer volume attributed to diesel BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_gas", "Accepted energy-offer volume attributed to gas BMUs not otherwise bucketed.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_load_response", "Accepted energy-offer volume attributed to load-response BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_npshyd", "Accepted energy-offer volume attributed to non-pumped-storage hydro BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_ocgt", "Accepted energy-offer volume attributed to OCGT BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_other", "Accepted energy-offer volume attributed to BMUs grouped as other.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_ps", "Accepted energy-offer volume attributed to pumped-storage BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_solar", "Accepted energy-offer volume attributed to solar BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_unclassified", "Accepted energy-offer volume attributed to unmatched/unclassified BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("vol_wind", "Accepted energy-offer volume attributed to wind BMUs.", "MWh", "ISP stack raw + BMU mapping", "no", ""),
        ("marginal_price", "Marginal price field from the original pipeline.", "GBP/MWh", "Original stack pipeline", "yes", "Filled within day in the original pipeline."),
        ("marginal_bmu", "BMU id corresponding to marginal_price in the original pipeline.", "string", "Original stack pipeline", "yes", "Filled within day in the original pipeline."),
        ("marginal_tech", "Technology corresponding to marginal_bmu in the original pipeline.", "string", "Original stack pipeline", "yes", "Filled within day in the original pipeline."),
        ("niv", "Net imbalance volume merged from BMRS system-prices data.", "MWh", "BMRS system-prices", "no", ""),
        ("offer_system_share", "Share of offer actions that were SO-flagged system actions.", "ratio", "Derived", "no", "Undefined when no offers exist."),
        ("gross_bm_volume", "Energy-only offer volume minus energy-only bid volume.", "MWh", "Derived", "no", "Undefined where both energy offers and energy bids are absent."),
        ("bess_offer_volume", "Convenience alias for battery energy-offer volume.", "MWh", "Derived from vol_battery", "no", ""),
        ("bess_offer_share", "Battery share of accepted energy-offer volume.", "ratio", "Derived", "no", "Undefined when offer_volume_energy is zero or missing."),
        ("bess_bid_share", "Battery share of accepted energy-bid volume.", "ratio", "Derived", "no", "Undefined when bid_volume_energy is zero or missing."),
        ("price_imputed", "Indicator that key price/marginal fields were imputed within day in the original pipeline.", "0/1", "Original stack pipeline", "yes", "Central for interpreting marginal coverage in the original file."),
        ("marginal_tech_unified", "Collapsed marginal technology grouping used for analysis.", "categorical", "Derived", "yes", "Derived after marginal imputation."),
        ("gas_marginal", "Indicator that the unified marginal technology is Gas.", "0/1", "Derived", "yes", ""),
    ]
    return pd.DataFrame(rows, columns=["variable", "definition", "unit", "source", "imputed", "notes"])


def build_data_note(summary: dict[str, int | float]) -> str:
    return f"""# master_panel_2023_2025 Data Note

This note documents the original `master_panel_2023_2025.csv` stack dataframe retained as the dissertation analysis base.

## Canonical File

- Original file kept unchanged: `data/processed/full_2023_2025/master_panel_2023_2025.csv`
- Cleaned companion copy: `data/processed/full_2023_2025/master_panel_2023_2025_cleaned.csv`
- Cleaned ISO-date copy: `data/processed/full_2023_2025/master_panel_2023_2025_cleaned_iso.csv`

## Key Tidy-Up Findings

- Original rows: {summary['original_rows']}
- Malformed key rows removed in cleaned copy: {summary['malformed_key_rows']}
- Cleaned rows: {summary['cleaned_rows']}
- Duplicate SP keys after removing malformed rows: {summary['duplicate_keys_after_cleaning']}
- Rows with any missing value after removing malformed rows: {summary['rows_with_any_missing_after_cleaning']}
- `price_imputed == 1` rows: {summary['price_imputed_rows']}
- Rows with `offer_volume_energy` missing but marginal label still present: {summary['energy_missing_but_marginal_present']}

## Interpretation Notes

- The original file contains malformed trailing rows with missing `settlementDate` and `settlementPeriod`. These rows are removed in the cleaned companion copies.
- The original stack pipeline uses within-day imputation for marginal fields (`marginal_price`, `marginal_bmu`, `marginal_tech`) and `offer_max_price`.
- As a result, some settlement periods have a marginal label even where there is no observed energy-offer volume in that SP.
- The cleaned copies in this tidy pass do **not** change the economic logic of the original panel; they only remove malformed rows and standardize dates in the ISO version.

## Recommended Usage

- Use the original file as the frozen dissertation base if that is already tied to analysis decisions.
- Use the cleaned or ISO copy for safer merging, diagnostics, and inspection.
- Use the audit CSVs alongside the file in methods writing so the imputation and malformed-row issues are explicit.
"""


def main() -> None:
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    original_rows = len(df)

    malformed_mask = df["settlementDate"].isna() | df["settlementPeriod"].isna()
    malformed_key_rows = int(malformed_mask.sum())

    cleaned = df.loc[~malformed_mask].copy()
    cleaned["settlementPeriod"] = pd.to_numeric(cleaned["settlementPeriod"], errors="coerce")
    duplicate_keys_after_cleaning = int(cleaned.duplicated(["settlementDate", "settlementPeriod"]).sum())
    rows_with_any_missing_after_cleaning = int(cleaned.isna().any(axis=1).sum())

    cleaned.to_csv(CLEANED_CSV, index=False)
    cleaned.to_parquet(CLEANED_PARQUET, index=False)

    iso = cleaned.copy()
    iso["settlementDate"] = pd.to_datetime(iso["settlementDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    iso.to_csv(CLEANED_ISO_CSV, index=False)
    iso.to_parquet(CLEANED_ISO_PARQUET, index=False)

    missingness = (
        cleaned.isna()
        .sum()
        .rename("missingValues")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missingness["missingPctOfRows"] = (100.0 * missingness["missingValues"] / len(cleaned)).round(4)
    missingness.to_csv(MISSINGNESS_AUDIT_CSV, index=False)

    price_imputed_rows = int((cleaned["price_imputed"] == 1).sum()) if "price_imputed" in cleaned.columns else 0
    energy_missing_but_marginal_present = int(
        (
            cleaned["offer_volume_energy"].isna().fillna(False)
            & cleaned["marginal_tech"].notna().fillna(False)
        ).sum()
    )
    imputation_audit = pd.DataFrame(
        [
            {"metric": "price_imputed_rows", "value": price_imputed_rows},
            {"metric": "price_imputed_pct_of_rows", "value": round(100.0 * price_imputed_rows / len(cleaned), 4)},
            {"metric": "energy_missing_but_marginal_present_rows", "value": energy_missing_but_marginal_present},
            {
                "metric": "energy_missing_but_marginal_present_pct_of_rows",
                "value": round(100.0 * energy_missing_but_marginal_present / len(cleaned), 4),
            },
        ]
    )
    imputation_audit.to_csv(IMPUTATION_AUDIT_CSV, index=False)

    key_integrity = pd.DataFrame(
        [
            {"metric": "original_rows", "value": original_rows},
            {"metric": "malformed_key_rows", "value": malformed_key_rows},
            {"metric": "cleaned_rows", "value": len(cleaned)},
            {"metric": "duplicate_keys_after_cleaning", "value": duplicate_keys_after_cleaning},
            {"metric": "rows_with_any_missing_after_cleaning", "value": rows_with_any_missing_after_cleaning},
            {"metric": "iso_null_dates_after_conversion", "value": int(iso["settlementDate"].isna().sum())},
        ]
    )
    key_integrity.to_csv(KEY_INTEGRITY_AUDIT_CSV, index=False)

    variable_dictionary = build_variable_dictionary()
    variable_dictionary.to_csv(VARIABLE_DICTIONARY_CSV, index=False)

    note = build_data_note(
        {
            "original_rows": original_rows,
            "malformed_key_rows": malformed_key_rows,
            "cleaned_rows": len(cleaned),
            "duplicate_keys_after_cleaning": duplicate_keys_after_cleaning,
            "rows_with_any_missing_after_cleaning": rows_with_any_missing_after_cleaning,
            "price_imputed_rows": price_imputed_rows,
            "energy_missing_but_marginal_present": energy_missing_but_marginal_present,
        }
    )
    DATA_NOTE_MD.write_text(note, encoding="utf-8")

    print("Tidied original master_panel_2023_2025 companion files")
    print(f"original rows = {original_rows}")
    print(f"malformed key rows removed = {malformed_key_rows}")
    print(f"cleaned rows = {len(cleaned)}")
    print(f"duplicate keys after cleaning = {duplicate_keys_after_cleaning}")
    print(f"rows with any missing after cleaning = {rows_with_any_missing_after_cleaning}")
    print(f"saved cleaned csv = {CLEANED_CSV}")
    print(f"saved iso csv = {CLEANED_ISO_CSV}")


if __name__ == "__main__":
    main()
