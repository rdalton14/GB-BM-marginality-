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

RUN_LABEL = "2026_01_05_to_2026_01_11"
START_DATE = date(2026, 1, 5)
END_DATE = date(2026, 1, 11)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/DISBSAD"
SLEEP_SECONDS = 0.10
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

ISPSTACK_DIR = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / f"ispstack_one_week_probe_{RUN_LABEL}"
ISPSTACK_PATH = ISPSTACK_DIR / "ispstack_one_week_long.parquet"

OUT_DIR = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / f"disbsad_probe_{RUN_LABEL}"
RAW_JSON_DIR = OUT_DIR / "raw_json"
LONG_CSV = OUT_DIR / "disbsad_long_2026_01_05_2026_01_11.csv"
LONG_PARQUET = OUT_DIR / "disbsad_long_2026_01_05_2026_01_11.parquet"
SCHEMA_CSV = OUT_DIR / "disbsad_schema_summary.csv"

AUDIT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "audits" / f"disbsad_ispstack_numeric_audit_{RUN_LABEL}"
MATCH_AUDIT_CSV = AUDIT_DIR / "disbsad_ispstack_numeric_match_audit.csv"
UNIQUE_ID_MAP_CSV = AUDIT_DIR / "disbsad_numeric_id_asset_party_lookup.csv"
ENRICHED_NUMERIC_CSV = AUDIT_DIR / "ispstack_numeric_rows_enriched_with_disbsad.csv"
ENRICHED_NUMERIC_PARQUET = AUDIT_DIR / "ispstack_numeric_rows_enriched_with_disbsad.parquet"
ENRICHED_FULL_CSV = AUDIT_DIR / "ispstack_one_week_with_disbsad_enrichment.csv"
ENRICHED_FULL_PARQUET = AUDIT_DIR / "ispstack_one_week_with_disbsad_enrichment.parquet"
ENRICHMENT_SUMMARY_CSV = AUDIT_DIR / "disbsad_enrichment_summary_by_service_party.csv"
ENERGY_ASSET_BMU_MAP_CSV = AUDIT_DIR / "disbsad_energy_asset_bmu_reference_match_audit.csv"
ENERGY_ASSET_MANUAL_QUEUE_CSV = AUDIT_DIR / "disbsad_energy_asset_bmu_manual_imputation_queue.csv"
SUMMARY_JSON = AUDIT_DIR / "disbsad_ispstack_numeric_summary.json"

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


def fetch_day(day: date) -> dict[str, Any]:
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


def normalise_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in result["rows"]:
        record = dict(row)
        record.setdefault("settlementDate", result["settlementDate"])
        rows.append(record)
    return rows


def is_numeric_only(series: pd.Series) -> pd.Series:
    return series.astype("string").str.fullmatch(r"\d+").fillna(False)


def build_schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        non_null = df[col].dropna()
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_share": float(df[col].isna().mean()) if len(df) else None,
                "n_unique_non_null": int(non_null.nunique()),
                "sample_values": " | ".join(non_null.astype(str).head(5).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_share", "column"], ascending=[False, True])


def write_csv_or_locked_copy(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"[WARN] Could not overwrite locked file: {path}")
        print(f"[WARN] Wrote updated copy instead: {fallback}")
        return fallback


def prepare_disbsad(disbsad: pd.DataFrame) -> pd.DataFrame:
    out = disbsad.copy()
    out["settlementDate"] = out["settlementDate"].astype(str)
    out["settlementPeriod"] = pd.to_numeric(out["settlementPeriod"], errors="coerce").astype("Int64")
    out["id"] = out["id"].astype("string").str.strip()
    out["cost_num"] = pd.to_numeric(out["cost"], errors="coerce")
    out["volume_num"] = pd.to_numeric(out["volume"], errors="coerce")
    out["implied_price"] = np.where(
        out["volume_num"].abs().gt(0),
        out["cost_num"] / out["volume_num"],
        np.nan,
    )
    out["inferred_direction_from_volume"] = np.where(out["volume_num"].lt(0), "bid", "offer")
    return out


def prepare_numeric_ispstack(ispstack: pd.DataFrame) -> pd.DataFrame:
    out = ispstack[is_numeric_only(ispstack["id"])].copy()
    out["settlementDate"] = out["settlementDate"].astype(str)
    out["settlementPeriod"] = pd.to_numeric(out["settlementPeriod"], errors="coerce").astype("Int64")
    out["id"] = out["id"].astype("string").str.strip()
    out["ispstack_final_price"] = pd.to_numeric(out["finalPrice"], errors="coerce")
    out["ispstack_original_price"] = pd.to_numeric(out["originalPrice"], errors="coerce")
    out["ispstack_volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["ispstack_cost_from_final_price"] = out["ispstack_final_price"] * out["ispstack_volume"]
    out["ispstack_abs_volume"] = out["ispstack_volume"].abs()
    return out


def build_match_audit(isp_numeric: pd.DataFrame, disbsad: pd.DataFrame) -> pd.DataFrame:
    dis_cols = [
        "settlementDate",
        "settlementPeriod",
        "id",
        "cost_num",
        "volume_num",
        "implied_price",
        "soFlag",
        "storFlag",
        "partyId",
        "assetId",
        "isTendered",
        "service",
        "inferred_direction_from_volume",
    ]
    audit = isp_numeric.merge(
        disbsad[dis_cols],
        on=["settlementDate", "settlementPeriod", "id"],
        how="left",
        indicator=True,
        suffixes=("_ispstack", "_disbsad"),
    )
    audit["key_match"] = audit["_merge"].eq("both")
    audit["direction_match"] = audit["direction"].eq(audit["inferred_direction_from_volume"])
    audit["volume_match"] = np.isclose(audit["ispstack_volume"], audit["volume_num"], atol=0.001, equal_nan=False)
    audit["abs_volume_match"] = np.isclose(
        audit["ispstack_abs_volume"],
        audit["volume_num"].abs(),
        atol=0.001,
        equal_nan=False,
    )
    audit["final_price_match"] = np.isclose(
        audit["ispstack_final_price"],
        audit["implied_price"],
        atol=0.01,
        equal_nan=False,
    )
    audit["original_price_match"] = np.isclose(
        audit["ispstack_original_price"],
        audit["implied_price"],
        atol=0.01,
        equal_nan=False,
    )
    audit["cost_match"] = np.isclose(
        audit["ispstack_cost_from_final_price"],
        audit["cost_num"],
        atol=0.05,
        equal_nan=False,
    )
    audit["strict_value_match"] = (
        audit["key_match"]
        & audit["abs_volume_match"]
        & (audit["final_price_match"] | audit["original_price_match"])
    )
    return audit


def build_unique_id_lookup(disbsad: pd.DataFrame) -> pd.DataFrame:
    lookup = (
        disbsad.groupby(["id", "partyId", "assetId", "service", "isTendered", "storFlag"], dropna=False)
        .agg(
            row_count=("id", "size"),
            first_settlement_date=("settlementDate", "min"),
            last_settlement_date=("settlementDate", "max"),
            settlement_period_count=("settlementPeriod", "nunique"),
            total_volume=("volume_num", "sum"),
            total_cost=("cost_num", "sum"),
            mean_implied_price=("implied_price", "mean"),
        )
        .reset_index()
        .sort_values(["id", "row_count"], ascending=[True, False])
    )
    return lookup


def enrich_numeric_rows(isp_numeric: pd.DataFrame, disbsad: pd.DataFrame) -> pd.DataFrame:
    dis_cols = [
        "settlementDate",
        "settlementPeriod",
        "id",
        "cost_num",
        "volume_num",
        "implied_price",
        "soFlag",
        "storFlag",
        "partyId",
        "assetId",
        "isTendered",
        "service",
        "inferred_direction_from_volume",
    ]
    enriched = isp_numeric.merge(
        disbsad[dis_cols],
        on=["settlementDate", "settlementPeriod", "id"],
        how="left",
        suffixes=("", "_disbsad"),
    )
    enriched = enriched.rename(
        columns={
            "cost_num": "disbsad_cost",
            "volume_num": "disbsad_volume",
            "implied_price": "disbsad_implied_price",
            "soFlag_disbsad": "disbsad_soFlag",
            "storFlag": "disbsad_storFlag",
            "partyId": "disbsad_partyId",
            "assetId": "disbsad_assetId",
            "isTendered": "disbsad_isTendered",
            "service": "disbsad_service",
            "inferred_direction_from_volume": "disbsad_direction_from_volume",
        }
    )
    enriched["disbsad_key_matched"] = enriched["disbsad_cost"].notna()
    enriched["disbsad_abs_volume_match"] = np.isclose(
        enriched["ispstack_abs_volume"],
        enriched["disbsad_volume"].abs(),
        atol=0.001,
        equal_nan=False,
    )
    enriched["disbsad_original_price_match"] = np.isclose(
        enriched["ispstack_original_price"],
        enriched["disbsad_implied_price"],
        atol=0.01,
        equal_nan=False,
    )
    enriched["disbsad_final_price_match"] = np.isclose(
        enriched["ispstack_final_price"],
        enriched["disbsad_implied_price"],
        atol=0.01,
        equal_nan=False,
    )
    enriched["disbsad_strict_value_match"] = (
        enriched["disbsad_key_matched"]
        & enriched["disbsad_abs_volume_match"]
        & (enriched["disbsad_original_price_match"] | enriched["disbsad_final_price_match"])
    )
    return enriched


def enrich_full_ispstack(ispstack: pd.DataFrame, enriched_numeric: pd.DataFrame) -> pd.DataFrame:
    enrichment_cols = [
        "settlementDate",
        "settlementPeriod",
        "id",
        "direction",
        "sequenceNumber",
        "disbsad_cost",
        "disbsad_volume",
        "disbsad_implied_price",
        "disbsad_soFlag",
        "disbsad_storFlag",
        "disbsad_partyId",
        "disbsad_assetId",
        "disbsad_isTendered",
        "disbsad_service",
        "disbsad_direction_from_volume",
        "disbsad_key_matched",
        "disbsad_strict_value_match",
    ]
    out = ispstack.copy()
    out["is_numeric_id"] = is_numeric_only(out["id"])
    out = out.merge(
        enriched_numeric[enrichment_cols],
        on=["settlementDate", "settlementPeriod", "id", "direction", "sequenceNumber"],
        how="left",
    )
    out["stack_row_class"] = np.where(out["is_numeric_id"], "disbsad_numeric", "bm_stack_named")
    return out


def build_enrichment_summary(enriched_numeric: pd.DataFrame) -> pd.DataFrame:
    summary = (
        enriched_numeric.groupby(["disbsad_service", "disbsad_partyId"], dropna=False)
        .agg(
            row_count=("id", "size"),
            unique_numeric_ids=("id", "nunique"),
            unique_assets=("disbsad_assetId", "nunique"),
            total_ispstack_volume=("ispstack_volume", "sum"),
            total_disbsad_volume=("disbsad_volume", "sum"),
            total_disbsad_cost=("disbsad_cost", "sum"),
            mean_disbsad_implied_price=("disbsad_implied_price", "mean"),
            so_flag_share=("disbsad_soFlag", lambda s: pd.Series(s).astype("boolean").mean()),
        )
        .reset_index()
        .sort_values(["row_count", "total_disbsad_cost"], ascending=[False, False])
    )
    return summary


def build_energy_asset_bmu_match(enriched_numeric: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bmu_ref_path = PROJECT_ROOT / "data" / "raw" / "reference" / "bmu_reference.csv"
    bmu_ref = pd.read_csv(bmu_ref_path, dtype=str)
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
    bmu_ref = bmu_ref[[c for c in bmu_ref_cols if c in bmu_ref.columns]].copy()

    energy = enriched_numeric[enriched_numeric["disbsad_service"].eq("Energy")].copy()
    asset_usage = (
        energy.groupby(["disbsad_assetId", "disbsad_partyId"], dropna=False)
        .agg(
            row_count=("id", "size"),
            unique_numeric_ids=("id", "nunique"),
            first_settlement_date=("settlementDate", "min"),
            last_settlement_date=("settlementDate", "max"),
            settlement_period_count=("settlementPeriod", "nunique"),
            total_volume=("ispstack_volume", "sum"),
            total_abs_volume=("ispstack_volume", lambda s: pd.to_numeric(s, errors="coerce").abs().sum()),
            total_cost=("disbsad_cost", "sum"),
            mean_implied_price=("disbsad_implied_price", "mean"),
            min_implied_price=("disbsad_implied_price", "min"),
            max_implied_price=("disbsad_implied_price", "max"),
        )
        .reset_index()
        .rename(columns={"disbsad_assetId": "assetId", "disbsad_partyId": "partyId"})
    )

    matched = asset_usage.merge(
        bmu_ref.add_prefix("bmu_ref_"),
        left_on="assetId",
        right_on="bmu_ref_nationalGridBmUnit",
        how="left",
    )
    matched["bmu_reference_match"] = matched["bmu_ref_nationalGridBmUnit"].notna()
    matched["manual_imputation_needed"] = ~matched["bmu_reference_match"]
    matched["suggested_manual_key"] = matched["assetId"]
    matched["suggested_identity_label"] = np.where(
        matched["bmu_reference_match"],
        matched["bmu_ref_elexonBmUnit"],
        matched["partyId"].fillna("UNKNOWN_PARTY") + " | " + matched["assetId"].fillna("UNKNOWN_ASSET"),
    )

    manual_queue = matched[matched["manual_imputation_needed"]].copy()
    manual_queue = manual_queue.sort_values(["total_abs_volume", "row_count"], ascending=[False, False])
    matched = matched.sort_values(["bmu_reference_match", "total_abs_volume"], ascending=[False, False])
    return matched, manual_queue


def summarise(disbsad: pd.DataFrame, isp_numeric: pd.DataFrame, audit: pd.DataFrame, request_rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = audit[audit["key_match"]].copy()
    strict = audit[audit["strict_value_match"]].copy()
    summary: dict[str, Any] = {
        "one_week_window": {"start_date": START_DATE.isoformat(), "end_date": END_DATE.isoformat()},
        "requests": {
            "total": len(request_rows),
            "ok": int(sum(r["status"] == "ok" for r in request_rows)),
            "error": int(sum(r["status"] == "error" for r in request_rows)),
            "rows_by_date": {r["settlementDate"]: int(r["row_count"]) for r in request_rows},
            "errors": [r for r in request_rows if r["status"] != "ok"],
        },
        "disbsad": {
            "rows": int(len(disbsad)),
            "unique_ids": int(disbsad["id"].nunique(dropna=True)),
            "unique_asset_ids": int(disbsad["assetId"].nunique(dropna=True)) if "assetId" in disbsad.columns else None,
            "rows_by_service": disbsad["service"].value_counts(dropna=False).to_dict() if "service" in disbsad.columns else {},
            "rows_by_so_flag": disbsad["soFlag"].value_counts(dropna=False).to_dict() if "soFlag" in disbsad.columns else {},
        },
        "ispstack_numeric": {
            "rows": int(len(isp_numeric)),
            "unique_numeric_ids": int(isp_numeric["id"].nunique(dropna=True)),
            "rows_by_direction": isp_numeric["direction"].value_counts(dropna=False).to_dict(),
            "rows_by_so_flag": isp_numeric["soFlag"].value_counts(dropna=False).to_dict(),
        },
        "crosscheck": {
            "key_matched_rows": int(audit["key_match"].sum()),
            "key_matched_share": float(audit["key_match"].mean()) if len(audit) else None,
            "strict_value_matched_rows": int(audit["strict_value_match"].sum()),
            "strict_value_matched_share": float(audit["strict_value_match"].mean()) if len(audit) else None,
            "matched_final_price_match_share": float(matched["final_price_match"].mean()) if len(matched) else None,
            "matched_original_price_match_share": float(matched["original_price_match"].mean()) if len(matched) else None,
            "matched_abs_volume_match_share": float(matched["abs_volume_match"].mean()) if len(matched) else None,
            "matched_cost_match_share": float(matched["cost_match"].mean()) if len(matched) else None,
            "matched_direction_match_share": float(matched["direction_match"].mean()) if len(matched) else None,
            "strict_matched_unique_asset_ids": int(strict["assetId"].nunique(dropna=True)) if len(strict) else 0,
            "strict_matched_rows_by_service": strict["service"].value_counts(dropna=False).to_dict() if len(strict) else {},
            "unmatched_numeric_rows": int((~audit["key_match"]).sum()),
        },
        "conclusion": {
            "numeric_ispstack_ids_are_disbsad": bool(len(audit) and audit["strict_value_match"].mean() > 0.95),
            "disbsad_adds_asset_party_resolution": bool(len(strict) and strict["assetId"].notna().mean() > 0.5),
        },
    }
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    request_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for day in daterange(START_DATE, END_DATE):
        print(f"Fetching DISBSAD {day.isoformat()}", flush=True)
        result = fetch_day(day)
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
        rows.extend(normalise_rows(result))
        time.sleep(SLEEP_SECONDS)

    disbsad_raw = pd.DataFrame(rows)
    if disbsad_raw.empty:
        SUMMARY_JSON.write_text(
            json.dumps({"message": "No DISBSAD rows returned.", "requests": request_rows}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError("No DISBSAD rows returned for the one-week probe.")

    schema_summary = build_schema_summary(disbsad_raw)
    write_csv_or_locked_copy(schema_summary, SCHEMA_CSV)

    disbsad = prepare_disbsad(disbsad_raw)
    write_csv_or_locked_copy(disbsad, LONG_CSV)
    disbsad.to_parquet(LONG_PARQUET, index=False)

    ispstack = pd.read_parquet(ISPSTACK_PATH)
    isp_numeric = prepare_numeric_ispstack(ispstack)
    audit = build_match_audit(isp_numeric, disbsad)
    enriched_numeric = enrich_numeric_rows(isp_numeric, disbsad)
    enriched_full = enrich_full_ispstack(ispstack, enriched_numeric)
    enrichment_summary = build_enrichment_summary(enriched_numeric)
    energy_asset_bmu_match, energy_asset_manual_queue = build_energy_asset_bmu_match(enriched_numeric)
    lookup = build_unique_id_lookup(disbsad)
    summary = summarise(disbsad, isp_numeric, audit, request_rows)
    summary["energy_asset_bmu_reference_match"] = {
        "energy_unique_assets": int(energy_asset_bmu_match["assetId"].nunique(dropna=True)),
        "energy_assets_matched_to_bmu_reference": int(energy_asset_bmu_match["bmu_reference_match"].sum()),
        "energy_assets_unmatched": int(energy_asset_manual_queue["assetId"].nunique(dropna=True)),
        "energy_asset_match_share": float(energy_asset_bmu_match["bmu_reference_match"].mean())
        if len(energy_asset_bmu_match)
        else None,
        "energy_rows_covered_by_matched_assets": int(
            energy_asset_bmu_match.loc[energy_asset_bmu_match["bmu_reference_match"], "row_count"].sum()
        ),
        "energy_rows_unmatched": int(energy_asset_manual_queue["row_count"].sum()) if len(energy_asset_manual_queue) else 0,
        "energy_row_match_share": float(
            energy_asset_bmu_match.loc[energy_asset_bmu_match["bmu_reference_match"], "row_count"].sum()
            / energy_asset_bmu_match["row_count"].sum()
        )
        if len(energy_asset_bmu_match) and energy_asset_bmu_match["row_count"].sum()
        else None,
    }

    write_csv_or_locked_copy(audit, MATCH_AUDIT_CSV)
    write_csv_or_locked_copy(enriched_numeric, ENRICHED_NUMERIC_CSV)
    enriched_numeric.to_parquet(ENRICHED_NUMERIC_PARQUET, index=False)
    write_csv_or_locked_copy(enriched_full, ENRICHED_FULL_CSV)
    enriched_full.to_parquet(ENRICHED_FULL_PARQUET, index=False)
    write_csv_or_locked_copy(enrichment_summary, ENRICHMENT_SUMMARY_CSV)
    write_csv_or_locked_copy(energy_asset_bmu_match, ENERGY_ASSET_BMU_MAP_CSV)
    write_csv_or_locked_copy(energy_asset_manual_queue, ENERGY_ASSET_MANUAL_QUEUE_CSV)
    write_csv_or_locked_copy(lookup, UNIQUE_ID_MAP_CSV)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("DISBSAD x ISPSTACK NUMERIC-ID CROSS-CHECK")
    print("=" * 72)
    print(f"DISBSAD rows fetched             : {len(disbsad):,}")
    print(f"ISPStack numeric rows checked    : {len(isp_numeric):,}")
    print(f"Key-matched numeric rows         : {summary['crosscheck']['key_matched_rows']:,} ({summary['crosscheck']['key_matched_share']:.2%})")
    print(f"Strict value-matched numeric rows: {summary['crosscheck']['strict_value_matched_rows']:,} ({summary['crosscheck']['strict_value_matched_share']:.2%})")
    print(f"Matched price agreement          : {summary['crosscheck']['matched_final_price_match_share']:.2%}")
    print(f"Matched abs-volume agreement     : {summary['crosscheck']['matched_abs_volume_match_share']:.2%}")
    print(f"Matched direction agreement      : {summary['crosscheck']['matched_direction_match_share']:.2%}")
    print(f"Energy asset BMU match share     : {summary['energy_asset_bmu_reference_match']['energy_asset_match_share']:.2%}")
    print(f"Energy row BMU match share       : {summary['energy_asset_bmu_reference_match']['energy_row_match_share']:.2%}")
    print(f"Energy manual queue              : {ENERGY_ASSET_MANUAL_QUEUE_CSV}")
    print(f"Enriched numeric rows            : {ENRICHED_NUMERIC_CSV}")
    print(f"Enriched full ISPStack           : {ENRICHED_FULL_PARQUET}")
    print(f"Summary JSON                     : {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
