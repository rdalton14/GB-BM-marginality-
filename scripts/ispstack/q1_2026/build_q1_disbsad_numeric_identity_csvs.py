from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 3, 31)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/DISBSAD"
SLEEP_SECONDS = 0.05
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

Q1_IDENTITY_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "q1_2026"
    / "ispstack_marginal_action_q1_2026"
    / "marginal_action_sp_q1_2026_identity_unknown_numeric.parquet"
)
BMU_REF_PATH = PROJECT_ROOT / "data" / "raw" / "reference" / "bmu_reference.csv"

DISBSAD_DIR = PROJECT_ROOT / "data" / "raw" / "ispstack" / "disbsad_q1_2026"
RAW_JSON_DIR = DISBSAD_DIR / "raw_json"
DISBSAD_CSV = DISBSAD_DIR / "disbsad_q1_2026.csv"
DISBSAD_PARQUET = DISBSAD_DIR / "disbsad_q1_2026.parquet"
DISBSAD_REQUEST_SUMMARY = DISBSAD_DIR / "disbsad_q1_2026_request_summary.json"

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "q1_2026_disbsad_numeric_identity_rebuild"
OUT_PANEL_CSV = OUT_DIR / "marginal_action_sp_q1_2026_identity_disbsad_numeric_rebuilt.csv"
OUT_ENERGY_ONLY_PANEL_CSV = OUT_DIR / "marginal_action_sp_q1_2026_identity_disbsad_energy_only_rebuilt.csv"
OUT_NUMERIC_AUDIT_CSV = OUT_DIR / "q1_numeric_winner_disbsad_resolution_audit.csv"
OUT_ENERGY_ONLY_NUMERIC_AUDIT_CSV = OUT_DIR / "q1_numeric_winner_disbsad_energy_only_resolution_audit.csv"
OUT_NON_ENERGY_EXCLUSION_CSV = OUT_DIR / "q1_numeric_winner_disbsad_non_energy_exclusion_audit.csv"
OUT_MANUAL_QUEUE_CSV = OUT_DIR / "q1_disbsad_energy_asset_bmu_manual_imputation_queue.csv"
OUT_ASSET_BMU_MAP_CSV = OUT_DIR / "q1_disbsad_energy_asset_bmu_reference_match_audit.csv"
OUT_SUMMARY_JSON = OUT_DIR / "q1_disbsad_numeric_identity_rebuild_summary.json"

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def daterange(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def extract_data(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def fetch_disbsad_day(day: date) -> dict[str, Any]:
    params = {
        "from": day.isoformat(),
        "to": day.isoformat(),
        "settlementPeriodFrom": 1,
        "settlementPeriodTo": 48,
        "format": "json",
    }
    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
            if response.status_code == 429:
                last_error = f"429 rate limited on attempt {attempt}"
                time.sleep(attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            return {
                "status": "ok",
                "status_code": response.status_code,
                "settlementDate": day.isoformat(),
                "url": response.url,
                "payload": payload,
                "rows": extract_data(payload),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(attempt)
    return {
        "status": "error",
        "status_code": None,
        "settlementDate": day.isoformat(),
        "url": BASE_URL,
        "payload": None,
        "rows": [],
        "error": last_error,
    }


def save_raw_payload(result: dict[str, Any]) -> None:
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_JSON_DIR / f"{result['settlementDate']}.json"
    payload = {
        "status": result["status"],
        "status_code": result["status_code"],
        "settlementDate": result["settlementDate"],
        "url": result["url"],
        "error": result["error"],
        "payload": result["payload"],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_or_fetch_disbsad() -> pd.DataFrame:
    DISBSAD_DIR.mkdir(parents=True, exist_ok=True)
    if DISBSAD_PARQUET.exists():
        return pd.read_parquet(DISBSAD_PARQUET)

    request_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for day in daterange(START_DATE, END_DATE):
        print(f"Fetching DISBSAD {day.isoformat()}", flush=True)
        result = fetch_disbsad_day(day)
        save_raw_payload(result)
        request_rows.append(
            {
                "settlementDate": day.isoformat(),
                "status": result["status"],
                "status_code": result["status_code"],
                "row_count": len(result["rows"]),
                "error": result["error"],
                "url": result["url"],
            }
        )
        for row in result["rows"]:
            record = dict(row)
            record.setdefault("settlementDate", day.isoformat())
            rows.append(record)
        time.sleep(SLEEP_SECONDS)

    disbsad = pd.DataFrame(rows)
    disbsad.to_csv(DISBSAD_CSV, index=False)
    disbsad.to_parquet(DISBSAD_PARQUET, index=False)
    DISBSAD_REQUEST_SUMMARY.write_text(json.dumps(request_rows, indent=2), encoding="utf-8")
    return disbsad


def prepare_disbsad(disbsad: pd.DataFrame) -> pd.DataFrame:
    out = disbsad.copy()
    out["settlementDate"] = pd.to_datetime(out["settlementDate"]).dt.strftime("%Y-%m-%d")
    out["settlementPeriod"] = pd.to_numeric(out["settlementPeriod"], errors="coerce").astype("Int64")
    out["raw_id"] = out["id"].astype("string").str.strip()
    out["disbsad_cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out["disbsad_volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["disbsad_implied_price"] = np.where(
        out["disbsad_volume"].abs().gt(0),
        out["disbsad_cost"] / out["disbsad_volume"],
        np.nan,
    )
    out["disbsad_direction_from_volume"] = np.where(out["disbsad_volume"].lt(0), "bid", "offer")
    out = out.rename(
        columns={
            "partyId": "disbsad_partyId",
            "assetId": "disbsad_assetId",
            "service": "disbsad_service",
            "soFlag": "disbsad_soFlag",
            "storFlag": "disbsad_storFlag",
            "isTendered": "disbsad_isTendered",
        }
    )
    return out


def enrich_disbsad_with_bmu_reference(disbsad: pd.DataFrame) -> pd.DataFrame:
    bmu_ref_cols = [
        "nationalGridBmUnit",
        "elexonBmUnit",
        "eic",
        "fuelType",
        "leadPartyName",
        "leadPartyId",
        "bmUnitType",
        "bmUnitName",
        "demandCapacity",
        "generationCapacity",
        "productionOrConsumptionFlag",
        "gspGroupId",
        "gspGroupName",
        "interconnectorId",
    ]
    bmu_ref = pd.read_csv(BMU_REF_PATH, dtype=str)
    bmu_ref = bmu_ref[[c for c in bmu_ref_cols if c in bmu_ref.columns]].drop_duplicates()
    out = disbsad.merge(
        bmu_ref.add_prefix("bmu_ref_"),
        left_on="disbsad_assetId",
        right_on="bmu_ref_nationalGridBmUnit",
        how="left",
    )
    out["disbsad_bmu_reference_match"] = out["bmu_ref_nationalGridBmUnit"].notna()
    out["disbsad_mapped_bmu"] = out["bmu_ref_elexonBmUnit"].fillna(out["disbsad_assetId"])
    out["disbsad_mapped_generator_id"] = out["bmu_ref_elexonBmUnit"].fillna(out["disbsad_assetId"])
    out["disbsad_mapped_generator_label"] = out["bmu_ref_bmUnitName"].fillna(
        out["disbsad_partyId"].fillna("UNKNOWN_PARTY") + " | " + out["disbsad_assetId"].fillna("UNKNOWN_ASSET")
    )
    out["disbsad_mapped_tech"] = out["bmu_ref_fuelType"]
    interconnector_mask = out["bmu_ref_bmUnitType"].eq("I") | out["bmu_ref_interconnectorId"].notna()
    out.loc[interconnector_mask, "disbsad_mapped_tech"] = "INTERCONNECTOR"
    out["disbsad_mapped_tech"] = out["disbsad_mapped_tech"].fillna(
        "DISBSAD_" + out["disbsad_service"].fillna("UNKNOWN").astype(str).str.upper().str.replace(r"\W+", "_", regex=True)
    )
    return out


def is_numeric_only(series: pd.Series) -> pd.Series:
    return series.astype("string").str.fullmatch(r"\d+").fillna(False)


def load_identity_panel() -> pd.DataFrame:
    df = pd.read_parquet(Q1_IDENTITY_PANEL)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"]).dt.strftime("%Y-%m-%d")
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    df["marginal_identity_raw_winner_id"] = df["marginal_identity_raw_winner_id"].astype("string").str.strip()
    return df


def build_rebuilt_identity_panel(panel: pd.DataFrame, disbsad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["settlementDate", "settlementPeriod", "raw_id"]
    dis_cols = [
        "settlementDate",
        "settlementPeriod",
        "raw_id",
        "disbsad_cost",
        "disbsad_volume",
        "disbsad_implied_price",
        "disbsad_direction_from_volume",
        "disbsad_partyId",
        "disbsad_assetId",
        "disbsad_service",
        "disbsad_soFlag",
        "disbsad_storFlag",
        "disbsad_isTendered",
        "disbsad_bmu_reference_match",
        "disbsad_mapped_bmu",
        "disbsad_mapped_generator_id",
        "disbsad_mapped_generator_label",
        "disbsad_mapped_tech",
        "bmu_ref_nationalGridBmUnit",
        "bmu_ref_elexonBmUnit",
        "bmu_ref_bmUnitType",
        "bmu_ref_bmUnitName",
        "bmu_ref_interconnectorId",
        "bmu_ref_fuelType",
        "bmu_ref_leadPartyName",
    ]
    dis_lookup = disbsad[dis_cols].drop_duplicates(keys)

    out = panel.copy()
    out["raw_id"] = out["marginal_identity_raw_winner_id"]
    out["q1_raw_winner_is_numeric"] = is_numeric_only(out["raw_id"])
    out = out.merge(dis_lookup, on=keys, how="left")
    out["q1_numeric_winner_disbsad_matched"] = out["q1_raw_winner_is_numeric"] & out["disbsad_assetId"].notna()
    out["q1_numeric_winner_disbsad_direction_match"] = out["marginal_side_winner"].eq(out["disbsad_direction_from_volume"])
    out["q1_numeric_winner_disbsad_abs_volume_match"] = np.isclose(
        pd.to_numeric(out["marginal_abs_volume_winner"], errors="coerce"),
        pd.to_numeric(out["disbsad_volume"], errors="coerce").abs(),
        atol=0.001,
        equal_nan=False,
    )
    out["q1_numeric_winner_disbsad_price_match"] = np.isclose(
        pd.to_numeric(out["marginal_price_winner"], errors="coerce"),
        pd.to_numeric(out["disbsad_implied_price"], errors="coerce"),
        atol=0.01,
        equal_nan=False,
    )

    out["q1_rebuilt_marginal_id_final"] = out["marginal_id_final"]
    out["q1_rebuilt_marginal_bmu_final"] = out["marginal_bmu_final"]
    out["q1_rebuilt_marginal_generator_id_final"] = out["marginal_generator_id_final"]
    out["q1_rebuilt_marginal_generator_label_final"] = out["marginal_generator_label_final"]
    out["q1_rebuilt_marginal_tech_final"] = out["marginal_tech_final"]
    out["q1_rebuilt_marginal_identity_resolution_rule_final"] = out["marginal_identity_resolution_rule_final"]
    out["q1_rebuilt_marginal_identity_unknown_numeric"] = out["marginal_identity_unknown_numeric"]

    matched = out["q1_numeric_winner_disbsad_matched"].fillna(False).astype(bool)
    ref_matched = out["disbsad_bmu_reference_match"].fillna(False).astype(bool)
    bmu_matched = matched & ref_matched
    manual_needed = matched & ~ref_matched

    out.loc[matched, "q1_rebuilt_marginal_id_final"] = out.loc[matched, "disbsad_assetId"]
    out.loc[matched, "q1_rebuilt_marginal_bmu_final"] = out.loc[matched, "disbsad_mapped_bmu"]
    out.loc[matched, "q1_rebuilt_marginal_generator_id_final"] = out.loc[matched, "disbsad_mapped_generator_id"]
    out.loc[matched, "q1_rebuilt_marginal_generator_label_final"] = out.loc[matched, "disbsad_mapped_generator_label"]
    out.loc[matched, "q1_rebuilt_marginal_tech_final"] = out.loc[matched, "disbsad_mapped_tech"]
    out.loc[bmu_matched, "q1_rebuilt_marginal_identity_resolution_rule_final"] = (
        "numeric_winner_resolved_via_disbsad_asset_bmu_reference"
    )
    out.loc[manual_needed, "q1_rebuilt_marginal_identity_resolution_rule_final"] = (
        "numeric_winner_resolved_via_disbsad_asset_manual_impute_needed"
    )
    out.loc[matched, "q1_rebuilt_marginal_identity_unknown_numeric"] = False

    numeric_audit_cols = [
        "settlementDate",
        "settlementPeriod",
        "marginal_side_winner",
        "marginal_identity_raw_winner_id",
        "marginal_price_winner",
        "marginal_volume_winner",
        "marginal_abs_volume_winner",
        "marginal_identity_substituted",
        "marginal_id_final",
        "marginal_bmu_final",
        "marginal_generator_label_final",
        "marginal_tech_final",
        "marginal_identity_resolution_rule_final",
        "q1_numeric_winner_disbsad_matched",
        "q1_numeric_winner_disbsad_direction_match",
        "q1_numeric_winner_disbsad_abs_volume_match",
        "q1_numeric_winner_disbsad_price_match",
        "disbsad_assetId",
        "disbsad_partyId",
        "disbsad_service",
        "disbsad_cost",
        "disbsad_volume",
        "disbsad_implied_price",
        "disbsad_bmu_reference_match",
        "bmu_ref_elexonBmUnit",
        "bmu_ref_nationalGridBmUnit",
        "bmu_ref_bmUnitType",
        "bmu_ref_bmUnitName",
        "bmu_ref_interconnectorId",
        "q1_rebuilt_marginal_id_final",
        "q1_rebuilt_marginal_bmu_final",
        "q1_rebuilt_marginal_generator_label_final",
        "q1_rebuilt_marginal_tech_final",
        "q1_rebuilt_marginal_identity_resolution_rule_final",
        "q1_rebuilt_marginal_identity_unknown_numeric",
    ]
    numeric_audit = out[out["q1_raw_winner_is_numeric"]][numeric_audit_cols].copy()
    return out.drop(columns=["raw_id"]), numeric_audit


def build_energy_asset_manual_queue(disbsad: pd.DataFrame, numeric_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    used_assets = numeric_audit.loc[
        numeric_audit["q1_numeric_winner_disbsad_matched"] & numeric_audit["disbsad_service"].eq("Energy"),
        ["disbsad_assetId", "disbsad_partyId"],
    ].drop_duplicates()

    energy = disbsad[disbsad["disbsad_service"].eq("Energy")].copy()
    energy_assets = (
        energy.groupby(["disbsad_assetId", "disbsad_partyId"], dropna=False)
        .agg(
            disbsad_energy_row_count=("raw_id", "size"),
            disbsad_energy_numeric_ids=("raw_id", "nunique"),
            first_settlement_date=("settlementDate", "min"),
            last_settlement_date=("settlementDate", "max"),
            settlement_period_count=("settlementPeriod", "nunique"),
            total_disbsad_volume=("disbsad_volume", "sum"),
            total_disbsad_abs_volume=("disbsad_volume", lambda s: pd.to_numeric(s, errors="coerce").abs().sum()),
            total_disbsad_cost=("disbsad_cost", "sum"),
            mean_disbsad_implied_price=("disbsad_implied_price", "mean"),
            bmu_reference_match=("disbsad_bmu_reference_match", "max"),
            bmu_ref_elexonBmUnit=("bmu_ref_elexonBmUnit", "first"),
            bmu_ref_nationalGridBmUnit=("bmu_ref_nationalGridBmUnit", "first"),
            bmu_ref_bmUnitName=("bmu_ref_bmUnitName", "first"),
            bmu_ref_interconnectorId=("bmu_ref_interconnectorId", "first"),
        )
        .reset_index()
    )
    energy_assets = energy_assets.merge(
        used_assets.assign(appears_as_numeric_marginal_winner=True),
        on=["disbsad_assetId", "disbsad_partyId"],
        how="left",
    )
    energy_assets["appears_as_numeric_marginal_winner"] = energy_assets["appears_as_numeric_marginal_winner"].fillna(False)
    energy_assets = energy_assets.sort_values(["bmu_reference_match", "total_disbsad_abs_volume"], ascending=[False, False])
    manual_queue = energy_assets[~energy_assets["bmu_reference_match"].fillna(False).astype(bool)].copy()
    manual_queue = manual_queue.sort_values(
        ["appears_as_numeric_marginal_winner", "total_disbsad_abs_volume"],
        ascending=[False, False],
    )
    return energy_assets, manual_queue


def build_energy_only_outputs(rebuilt: pd.DataFrame, numeric_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    energy_numeric_mask = (
        rebuilt["q1_raw_winner_is_numeric"].fillna(False).astype(bool)
        & rebuilt["q1_numeric_winner_disbsad_matched"].fillna(False).astype(bool)
        & rebuilt["disbsad_service"].eq("Energy")
    )
    non_energy_numeric_mask = (
        rebuilt["q1_raw_winner_is_numeric"].fillna(False).astype(bool)
        & rebuilt["q1_numeric_winner_disbsad_matched"].fillna(False).astype(bool)
        & ~rebuilt["disbsad_service"].eq("Energy")
    )

    out = rebuilt.copy()
    out["q1_energy_only_disbsad_numeric_retained"] = energy_numeric_mask
    out["q1_energy_only_disbsad_numeric_excluded"] = non_energy_numeric_mask

    # Start from the pre-DISBSAD final identity and only apply DISBSAD resolution for Energy-service rows.
    out["q1_energy_only_marginal_id_final"] = out["marginal_id_final"]
    out["q1_energy_only_marginal_bmu_final"] = out["marginal_bmu_final"]
    out["q1_energy_only_marginal_generator_id_final"] = out["marginal_generator_id_final"]
    out["q1_energy_only_marginal_generator_label_final"] = out["marginal_generator_label_final"]
    out["q1_energy_only_marginal_tech_final"] = out["marginal_tech_final"]
    out["q1_energy_only_marginal_identity_resolution_rule_final"] = out["marginal_identity_resolution_rule_final"]
    out["q1_energy_only_marginal_identity_unknown_numeric"] = out["marginal_identity_unknown_numeric"]

    energy_bmu_matched = energy_numeric_mask & out["disbsad_bmu_reference_match"].fillna(False).astype(bool)
    energy_manual = energy_numeric_mask & ~out["disbsad_bmu_reference_match"].fillna(False).astype(bool)

    out.loc[energy_numeric_mask, "q1_energy_only_marginal_id_final"] = out.loc[energy_numeric_mask, "disbsad_assetId"]
    out.loc[energy_numeric_mask, "q1_energy_only_marginal_bmu_final"] = out.loc[energy_numeric_mask, "disbsad_mapped_bmu"]
    out.loc[energy_numeric_mask, "q1_energy_only_marginal_generator_id_final"] = out.loc[
        energy_numeric_mask, "disbsad_mapped_generator_id"
    ]
    out.loc[energy_numeric_mask, "q1_energy_only_marginal_generator_label_final"] = out.loc[
        energy_numeric_mask, "disbsad_mapped_generator_label"
    ]
    out.loc[energy_numeric_mask, "q1_energy_only_marginal_tech_final"] = out.loc[energy_numeric_mask, "disbsad_mapped_tech"]
    out.loc[energy_bmu_matched, "q1_energy_only_marginal_identity_resolution_rule_final"] = (
        "numeric_winner_energy_resolved_via_disbsad_asset_bmu_reference"
    )
    out.loc[energy_manual, "q1_energy_only_marginal_identity_resolution_rule_final"] = (
        "numeric_winner_energy_resolved_via_disbsad_asset_manual_impute_needed"
    )
    out.loc[energy_numeric_mask, "q1_energy_only_marginal_identity_unknown_numeric"] = False
    out.loc[non_energy_numeric_mask, "q1_energy_only_marginal_identity_resolution_rule_final"] = (
        "numeric_winner_disbsad_non_energy_excluded_from_energy_only"
    )

    energy_audit = numeric_audit[numeric_audit["disbsad_service"].eq("Energy")].copy()
    non_energy_audit = numeric_audit[
        numeric_audit["q1_numeric_winner_disbsad_matched"].fillna(False).astype(bool)
        & ~numeric_audit["disbsad_service"].eq("Energy")
    ].copy()
    return out, energy_audit, non_energy_audit


def build_summary(panel: pd.DataFrame, rebuilt: pd.DataFrame, numeric_audit: pd.DataFrame, energy_assets: pd.DataFrame, manual_queue: pd.DataFrame) -> dict[str, Any]:
    matched = numeric_audit["q1_numeric_winner_disbsad_matched"].fillna(False)
    return {
        "q1_window": {"start_date": START_DATE.isoformat(), "end_date": END_DATE.isoformat()},
        "input_identity_panel": str(Q1_IDENTITY_PANEL),
        "rows": int(len(panel)),
        "numeric_raw_winner_rows": int(len(numeric_audit)),
        "numeric_raw_winner_rows_disbsad_matched": int(matched.sum()),
        "numeric_raw_winner_disbsad_match_share": float(matched.mean()) if len(numeric_audit) else None,
        "previous_unknown_numeric_rows": int(panel["marginal_identity_unknown_numeric"].fillna(False).sum()),
        "rebuilt_unknown_numeric_rows": int(rebuilt["q1_rebuilt_marginal_identity_unknown_numeric"].fillna(False).sum()),
        "numeric_raw_winner_bmu_reference_matched_rows": int(
            numeric_audit["disbsad_bmu_reference_match"].fillna(False).astype(bool).sum()
        ),
        "numeric_raw_winner_manual_impute_rows": int(
            (matched & ~numeric_audit["disbsad_bmu_reference_match"].fillna(False).astype(bool)).sum()
        ),
        "numeric_raw_winner_services": numeric_audit["disbsad_service"].value_counts(dropna=False).to_dict(),
        "energy_asset_bmu_reference": {
            "unique_energy_assets": int(energy_assets["disbsad_assetId"].nunique(dropna=True)),
            "matched_energy_assets": int(energy_assets["bmu_reference_match"].fillna(False).astype(bool).sum()),
            "unmatched_energy_assets": int(manual_queue["disbsad_assetId"].nunique(dropna=True)),
            "asset_match_share": float(energy_assets["bmu_reference_match"].fillna(False).astype(bool).mean())
            if len(energy_assets)
            else None,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_disbsad = load_or_fetch_disbsad()
    disbsad = enrich_disbsad_with_bmu_reference(prepare_disbsad(raw_disbsad))
    panel = load_identity_panel()
    rebuilt, numeric_audit = build_rebuilt_identity_panel(panel, disbsad)
    energy_only_panel, energy_only_numeric_audit, non_energy_exclusion = build_energy_only_outputs(rebuilt, numeric_audit)
    energy_assets, manual_queue = build_energy_asset_manual_queue(disbsad, numeric_audit)
    summary = build_summary(panel, rebuilt, numeric_audit, energy_assets, manual_queue)
    summary["energy_only_outputs"] = {
        "numeric_winner_energy_rows_retained": int(len(energy_only_numeric_audit)),
        "numeric_winner_non_energy_rows_excluded": int(len(non_energy_exclusion)),
        "energy_numeric_winner_bmu_reference_matched_rows": int(
            energy_only_numeric_audit["disbsad_bmu_reference_match"].fillna(False).astype(bool).sum()
        ),
        "energy_numeric_winner_manual_impute_rows": int(
            (~energy_only_numeric_audit["disbsad_bmu_reference_match"].fillna(False).astype(bool)).sum()
        ),
    }

    rebuilt.to_csv(OUT_PANEL_CSV, index=False)
    energy_only_panel.to_csv(OUT_ENERGY_ONLY_PANEL_CSV, index=False)
    numeric_audit.to_csv(OUT_NUMERIC_AUDIT_CSV, index=False)
    energy_only_numeric_audit.to_csv(OUT_ENERGY_ONLY_NUMERIC_AUDIT_CSV, index=False)
    non_energy_exclusion.to_csv(OUT_NON_ENERGY_EXCLUSION_CSV, index=False)
    energy_assets.to_csv(OUT_ASSET_BMU_MAP_CSV, index=False)
    manual_queue.to_csv(OUT_MANUAL_QUEUE_CSV, index=False)
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Q1 DISBSAD numeric identity CSV rebuild complete")
    print("=" * 72)
    print(f"Panel rows                         : {len(rebuilt):,}")
    print(f"Numeric raw winner rows            : {summary['numeric_raw_winner_rows']:,}")
    print(f"Numeric winners matched to DISBSAD : {summary['numeric_raw_winner_rows_disbsad_matched']:,}")
    print(f"Energy numeric winners retained    : {summary['energy_only_outputs']['numeric_winner_energy_rows_retained']:,}")
    print(f"Non-energy numeric winners excluded: {summary['energy_only_outputs']['numeric_winner_non_energy_rows_excluded']:,}")
    print(f"Previous UNKNOWN_NUMERIC rows      : {summary['previous_unknown_numeric_rows']:,}")
    print(f"Rebuilt UNKNOWN_NUMERIC rows       : {summary['rebuilt_unknown_numeric_rows']:,}")
    print(f"Energy asset match share           : {summary['energy_asset_bmu_reference']['asset_match_share']:.2%}")
    print(f"Rebuilt panel CSV                  : {OUT_PANEL_CSV}")
    print(f"Energy-only panel CSV              : {OUT_ENERGY_ONLY_PANEL_CSV}")
    print(f"Energy-only numeric audit CSV      : {OUT_ENERGY_ONLY_NUMERIC_AUDIT_CSV}")
    print(f"Non-energy exclusion CSV           : {OUT_NON_ENERGY_EXCLUSION_CSV}")
    print(f"Numeric audit CSV                  : {OUT_NUMERIC_AUDIT_CSV}")
    print(f"Manual queue CSV                   : {OUT_MANUAL_QUEUE_CSV}")


if __name__ == "__main__":
    main()
