from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "acceptances_all_q1_2026"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LONG_PARQUET = PROCESSED_DIR / "acceptances_all_q1_2026_long.parquet"
LONG_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_long.csv"
SCHEMA_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_schema_summary.csv"
COMPLETENESS_SP_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_completeness_by_sp.csv"
COMPLETENESS_DATE_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_completeness_by_date.csv"
PULL_ERRORS_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_pull_errors.csv"
CANDIDATE_FIELDS_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_candidate_fields.csv"
IDENTIFIER_QUALITY_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_identifier_quality.csv"
TOP_IDENTIFIERS_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_top_identifiers.csv"
UNRESOLVED_IDENTIFIERS_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_unresolved_identifiers.csv"
STACK_SUMMARY_SP_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_stack_summary_sp.csv"
MARGINAL_CANDIDATES_SP_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_marginal_candidates_sp.csv"
MARGINAL_IDENTIFIER_QUALITY_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_marginal_identifier_quality.csv"
TOP_MARGINAL_IDENTIFIERS_CSV = PROCESSED_DIR / "acceptances_all_q1_2026_top_marginal_identifiers.csv"
SUMMARY_JSON = PROCESSED_DIR / "acceptances_all_q1_2026_summary.json"

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/acceptances/all"
SOURCE_ENDPOINT = "acceptances_all"
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 3, 31)
SETTLEMENT_PERIODS = list(range(1, 49))
SLEEP_SECONDS = 0.10
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def generate_dates(start: date, end: date) -> list[date]:
    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def save_raw_response(settlement_date: str, settlement_period: int, result: dict[str, Any]) -> None:
    day_dir = RAW_DIR / settlement_date
    day_dir.mkdir(parents=True, exist_ok=True)
    out_path = day_dir / f"SP_{settlement_period}.json"
    payload = {
        "status": result["status"],
        "status_code": result["status_code"],
        "settlementDate": settlement_date,
        "settlementPeriod": settlement_period,
        "url": result["url"],
        "error": result["error"],
        "pull_timestamp": result["pull_timestamp"],
        "payload": result["payload"],
        "row_count": len(result["rows"]),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def api_request(settlement_date: str, settlement_period: int) -> dict[str, Any]:
    url = f"{BASE_URL}/{settlement_date}/{settlement_period}"
    params = {"format": "json"}
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        pull_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = SESSION.get(url, params=params, timeout=TIMEOUT_SECONDS)

            if response.status_code == 404:
                return {
                    "status": "not_found",
                    "status_code": 404,
                    "url": response.url,
                    "payload": {"note": "404 not found"},
                    "rows": [],
                    "error": "404 not found",
                    "pull_timestamp": pull_timestamp,
                }

            if response.status_code == 429:
                last_error = f"429 rate limited on attempt {attempt}"
                time.sleep(attempt)
                continue

            if response.status_code >= 500:
                last_error = f"{response.status_code} server error on attempt {attempt}"
                time.sleep(attempt)
                continue

            response.raise_for_status()
            try:
                payload = response.json()
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "malformed_json",
                    "status_code": response.status_code,
                    "url": response.url,
                    "payload": {"raw_text": response.text[:5000]},
                    "rows": [],
                    "error": f"malformed JSON: {exc}",
                    "pull_timestamp": pull_timestamp,
                }

            rows = extract_rows(payload)
            return {
                "status": "ok" if rows else "empty",
                "status_code": response.status_code,
                "url": response.url,
                "payload": payload if rows else {"metadata": payload.get("metadata") if isinstance(payload, dict) else None, "note": "empty data"},
                "rows": rows,
                "error": None,
                "pull_timestamp": pull_timestamp,
            }
        except requests.Timeout:
            last_error = f"timeout on attempt {attempt}"
            time.sleep(attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(attempt)

    return {
        "status": "error",
        "status_code": None,
        "url": url,
        "payload": {"note": "request failed after retries"},
        "rows": [],
        "error": last_error,
        "pull_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def flatten_rows(settlement_date: str, settlement_period: int, rows: list[dict[str, Any]], pull_timestamp: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["settlementDate"] = record.get("settlementDate", settlement_date)
        record["settlementPeriod"] = int(record.get("settlementPeriod", settlement_period))
        record["source_endpoint"] = SOURCE_ENDPOINT
        record["pull_timestamp"] = pull_timestamp
        out.append(record)
    return out


def normalize_long_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "settlementDate" in out.columns:
        out["settlementDate"] = out["settlementDate"].astype("string")
    if "settlementPeriod" in out.columns:
        out["settlementPeriod"] = pd.to_numeric(out["settlementPeriod"], errors="coerce").astype("Int64")
    for col in ["acceptanceNumber", "bidOfferPairId"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in ["bidPrice", "offerPrice", "acceptanceVolume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        non_null = df[col].dropna()
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_percentage": float(df[col].isna().mean()),
                "n_unique": int(non_null.nunique()),
                "example_values": " | ".join(non_null.astype(str).head(5).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("column")


def completeness_diagnostics(request_log: pd.DataFrame, long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sp_df = request_log.copy()
    sp_df["had_rows"] = sp_df["row_count"].gt(0)
    sp_df["request_error"] = sp_df["status"].isin(["error", "malformed_json", "not_found"])
    sp_df["missing_or_nonstandard_sp"] = ~sp_df["settlementPeriod"].isin(SETTLEMENT_PERIODS)

    date_summary = (
        sp_df.groupby("settlementDate", as_index=False)
        .agg(
            n_requests=("settlementPeriod", "size"),
            n_successful_requests=("status", lambda s: int(s.isin(["ok", "empty"]).sum())),
            n_non_empty=("had_rows", "sum"),
            n_empty=("status", lambda s: int((s == "empty").sum())),
            n_errors=("request_error", "sum"),
            rows_returned=("row_count", "sum"),
        )
    )
    date_summary["missing_settlement_periods"] = date_summary["settlementDate"].map(
        sp_df.groupby("settlementDate")["settlementPeriod"]
        .apply(lambda s: ";".join(str(sp) for sp in sorted(set(SETTLEMENT_PERIODS) - set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()))))
        .to_dict()
    )
    median_rows_by_date = date_summary["rows_returned"].median()
    date_summary["unusually_low_row_count"] = date_summary["rows_returned"] < (0.5 * median_rows_by_date)

    sp_summary = (
        sp_df.groupby("settlementPeriod", as_index=False)
        .agg(
            n_requests=("settlementDate", "size"),
            n_non_empty=("had_rows", "sum"),
            n_empty=("status", lambda s: int((s == "empty").sum())),
            n_errors=("request_error", "sum"),
            rows_returned=("row_count", "sum"),
        )
    )
    median_rows_by_sp = sp_summary["rows_returned"].median()
    sp_summary["unusually_low_row_count"] = sp_summary["rows_returned"] < (0.5 * median_rows_by_sp)

    errors = sp_df[sp_df["request_error"] | sp_df["status"].eq("empty")].copy()

    if not long_df.empty:
        dup_mask = long_df.duplicated(keep=False)
        duplicate_rows = int(dup_mask.sum())
        date_summary["duplicate_rows_q1_total"] = duplicate_rows
        sp_summary["duplicate_rows_q1_total"] = duplicate_rows

    return sp_summary, date_summary, errors


def candidate_field_mapping(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    lower = {c: c.lower() for c in cols}

    def sample(col: str) -> str:
        return " | ".join(df[col].dropna().astype(str).head(5).tolist()) if col in df.columns else ""

    candidates = {
        "BMU ID": [c for c in cols if "bmunit" in lower[c] and "nationalgrid" not in lower[c]],
        "national grid BMU ID": [c for c in cols if "nationalgridbmunit" in lower[c]],
        "plant / asset identifier": [c for c in cols if any(k in lower[c] for k in ["asset", "unit", "bmu"])],
        "technology / fuel type": [c for c in cols if any(k in lower[c] for k in ["fuel", "tech", "technology"])],
        "bid/offer side": [c for c in cols if any(k in lower[c] for k in ["side", "direction", "type"])],
        "bid/offer price": [c for c in cols if "price" in lower[c]],
        "bid/offer volume": [c for c in cols if "volume" in lower[c]],
        "acceptance number": [c for c in cols if "acceptance" in lower[c] and "number" in lower[c]],
        "acceptance time": [c for c in cols if "acceptance" in lower[c] and "time" in lower[c]],
        "pair number or identifier": [c for c in cols if "pair" in lower[c]],
        "numeric-only stack or generator identifier": [c for c in cols if any(k in lower[c] for k in ["id", "number"])],
    }

    rows: list[dict[str, Any]] = []
    for role, candidate_cols in candidates.items():
        if not candidate_cols:
            rows.append({"role": role, "candidate_column": None, "sample_values": None})
            continue
        for col in candidate_cols:
            rows.append({"role": role, "candidate_column": col, "sample_values": sample(col)})
    return pd.DataFrame(rows)


def choose_identifier_column(df: pd.DataFrame) -> str | None:
    preferred = ["bmUnit", "nationalGridBmUnit", "id", "assetId"]
    for col in preferred:
        if col in df.columns:
            return col
    return None


def is_numeric_only(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text.isdigit() if text else False


def identifier_quality(df: pd.DataFrame, candidate_field_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identifier_col = choose_identifier_column(df)
    rows: list[dict[str, Any]] = []

    if identifier_col is None:
        rows.append({"metric": "identifier_column_found", "value": False})
        return pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame()

    ident = df[identifier_col].astype("string")
    missing_mask = ident.isna() | ident.eq("")
    numeric_mask = ident.map(is_numeric_only)

    rows.extend(
        [
            {"metric": "identifier_column", "value": identifier_col},
            {"metric": "unique_identifiers", "value": int(ident.nunique(dropna=True))},
            {"metric": "missing_identifier_rows", "value": int(missing_mask.sum())},
            {"metric": "numeric_only_identifier_rows", "value": int(numeric_mask.sum())},
            {"metric": "missing_identifier_share", "value": float(missing_mask.mean())},
            {"metric": "numeric_only_identifier_share", "value": float(numeric_mask.mean())},
        ]
    )

    sp_all_recognisable = (
        df.assign(identifier_missing=missing_mask, identifier_numeric=numeric_mask)
        .groupby(["settlementDate", "settlementPeriod"])
        .apply(lambda g: int((~g["identifier_missing"] & ~g["identifier_numeric"]).all()))
    )
    rows.append(
        {
            "metric": "settlement_periods_all_actions_have_recognisable_identifier_share",
            "value": float(sp_all_recognisable.mean()),
        }
    )

    top_identifiers = (
        ident.fillna("MISSING_IDENTIFIER")
        .value_counts()
        .rename_axis("identifier")
        .reset_index(name="row_count")
        .head(50)
    )
    unresolved = (
        ident[numeric_mask | missing_mask]
        .fillna("MISSING_IDENTIFIER")
        .value_counts()
        .rename_axis("identifier")
        .reset_index(name="row_count")
        .head(50)
    )

    return pd.DataFrame(rows), top_identifiers, unresolved


def infer_stack_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "offerPrice" in out.columns and "bidPrice" in out.columns:
        offer_rows = out.copy()
        offer_rows["inferred_side"] = "offer"
        offer_rows["accepted_price"] = pd.to_numeric(offer_rows["offerPrice"], errors="coerce")
        if "acceptanceVolume" in offer_rows.columns:
            offer_rows["accepted_volume"] = pd.to_numeric(offer_rows["acceptanceVolume"], errors="coerce")
        else:
            offer_rows["accepted_volume"] = pd.NA

        bid_rows = out.copy()
        bid_rows["inferred_side"] = "bid"
        bid_rows["accepted_price"] = pd.to_numeric(bid_rows["bidPrice"], errors="coerce")
        if "acceptanceVolume" in bid_rows.columns:
            bid_rows["accepted_volume"] = pd.to_numeric(bid_rows["acceptanceVolume"], errors="coerce")
        else:
            bid_rows["accepted_volume"] = pd.NA

        stacked = pd.concat([offer_rows, bid_rows], ignore_index=True)
        stacked = stacked[stacked["accepted_price"].notna()].copy()
        return stacked

    out["inferred_side"] = pd.NA
    out["accepted_price"] = pd.NA
    out["accepted_volume"] = pd.NA
    return out


def stack_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stacked = infer_stack_rows(df)
    identifier_col = choose_identifier_column(df)
    if identifier_col is None:
        identifier_col = "bmUnit"
        stacked[identifier_col] = pd.NA

    sp_summary_records: list[dict[str, Any]] = []
    marginal_records: list[dict[str, Any]] = []

    for (settlement_date, settlement_period), grp in stacked.groupby(["settlementDate", "settlementPeriod"], sort=True):
        offers = grp[grp["inferred_side"] == "offer"].copy()
        bids = grp[grp["inferred_side"] == "bid"].copy()

        sp_summary_records.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": int(settlement_period),
                "n_offer_actions": int(len(offers)),
                "n_bid_actions": int(len(bids)),
                "total_offer_volume": float(pd.to_numeric(offers["accepted_volume"], errors="coerce").sum()) if not offers.empty else None,
                "total_bid_volume": float(pd.to_numeric(bids["accepted_volume"], errors="coerce").sum()) if not bids.empty else None,
            }
        )

        offer_top = None
        if not offers.empty:
            offer_top = offers.sort_values(["accepted_price", "acceptanceNumber"], ascending=[False, True], kind="stable").iloc[0]

        bid_top = None
        if not bids.empty:
            # Convention: choose the highest accepted bid price as the edge-of-stack bid-side candidate.
            bid_top = bids.sort_values(["accepted_price", "acceptanceNumber"], ascending=[False, True], kind="stable").iloc[0]

        marginal_records.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": int(settlement_period),
                "marginal_offer_identifier": None if offer_top is None else offer_top.get(identifier_col),
                "marginal_offer_price": None if offer_top is None else float(offer_top["accepted_price"]),
                "marginal_offer_volume": None if offer_top is None or pd.isna(offer_top.get("accepted_volume")) else float(offer_top["accepted_volume"]),
                "marginal_bid_identifier": None if bid_top is None else bid_top.get(identifier_col),
                "marginal_bid_price": None if bid_top is None else float(bid_top["accepted_price"]),
                "marginal_bid_volume": None if bid_top is None or pd.isna(bid_top.get("accepted_volume")) else float(bid_top["accepted_volume"]),
                "number_of_accepted_offer_actions": int(len(offers)),
                "number_of_accepted_bid_actions": int(len(bids)),
                "bid_candidate_convention": "highest accepted bid price",
            }
        )

    return pd.DataFrame(sp_summary_records), pd.DataFrame(marginal_records)


def marginal_identifier_quality(marginal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def missing_mask(series: pd.Series) -> pd.Series:
        s = series.astype("string")
        return s.isna() | s.eq("")

    def numeric_mask(series: pd.Series) -> pd.Series:
        return series.astype("string").map(is_numeric_only)

    offer_id = marginal_df["marginal_offer_identifier"].astype("string")
    bid_id = marginal_df["marginal_bid_identifier"].astype("string")

    overall = pd.concat(
        [
            offer_id.rename("identifier"),
            bid_id.rename("identifier"),
        ],
        ignore_index=True,
    )

    rows = [
        {"metric": "marginal_offer_identifier_missing_count", "value": int(missing_mask(offer_id).sum())},
        {"metric": "marginal_offer_identifier_numeric_only_count", "value": int(numeric_mask(offer_id).sum())},
        {"metric": "marginal_bid_identifier_missing_count", "value": int(missing_mask(bid_id).sum())},
        {"metric": "marginal_bid_identifier_numeric_only_count", "value": int(numeric_mask(bid_id).sum())},
    ]

    top_offer = (
        offer_id.fillna("MISSING_IDENTIFIER").value_counts().rename_axis("identifier").reset_index(name="offer_count").head(50)
    )
    top_bid = (
        bid_id.fillna("MISSING_IDENTIFIER").value_counts().rename_axis("identifier").reset_index(name="bid_count").head(50)
    )
    top_overall = (
        overall.fillna("MISSING_IDENTIFIER").value_counts().rename_axis("identifier").reset_index(name="overall_count").head(50)
    )

    top = top_offer.merge(top_bid, on="identifier", how="outer").merge(top_overall, on="identifier", how="outer")
    return pd.DataFrame(rows), top


def build_summary(
    request_log: pd.DataFrame,
    long_df: pd.DataFrame,
    schema_df: pd.DataFrame,
    candidate_fields_df: pd.DataFrame,
    identifier_quality_df: pd.DataFrame,
    stack_summary_df: pd.DataFrame,
    marginal_df: pd.DataFrame,
) -> dict[str, Any]:
    expected = len(generate_dates(START_DATE, END_DATE)) * len(SETTLEMENT_PERIODS)
    successful = int(request_log["status"].isin(["ok", "empty"]).sum())
    non_empty = int(request_log["status"].eq("ok").sum())
    empty = int(request_log["status"].eq("empty").sum())
    errors = int(request_log["status"].isin(["error", "malformed_json", "not_found"]).sum())

    dates_low = []
    if not long_df.empty:
        by_date = long_df.groupby("settlementDate").size()
        threshold = 0.5 * by_date.median()
        dates_low = by_date[by_date < threshold].index.tolist()

    sp_low = []
    if not long_df.empty:
        by_sp = long_df.groupby("settlementPeriod").size()
        threshold_sp = 0.5 * by_sp.median()
        sp_low = [int(x) for x in by_sp[by_sp < threshold_sp].index.tolist()]

    cols = set(long_df.columns)
    has_price = "bidPrice" in cols and "offerPrice" in cols
    has_volume = any("volume" in str(c).lower() for c in cols)
    has_side = any(k in cols for k in ["side", "direction", "acceptanceType", "bidOfferType"])
    has_identifier = any(k in cols for k in ["bmUnit", "nationalGridBmUnit"])

    ident_map = dict(zip(identifier_quality_df["metric"], identifier_quality_df["value"])) if not identifier_quality_df.empty else {}

    summary = {
        "coverage": {
            "expected_date_sp_combinations": expected,
            "successfully_requested": successful,
            "non_empty": non_empty,
            "empty": empty,
            "request_errors": errors,
            "low_row_dates": dates_low,
            "low_row_settlement_periods": sp_low,
        },
        "schema": {
            "contains_price": has_price,
            "contains_volume": has_volume,
            "contains_acceptance_number": "acceptanceNumber" in cols,
            "contains_acceptance_time": "acceptanceTime" in cols,
            "contains_bid_offer_side_field": has_side,
            "contains_bmu_or_plant_identifier": has_identifier,
            "contains_soflag": "soFlag" in cols,
        },
        "identifier_quality": {
            "best_identifier_column": ident_map.get("identifier_column"),
            "numeric_only_identifier_share": ident_map.get("numeric_only_identifier_share"),
            "missing_identifier_share": ident_map.get("missing_identifier_share"),
        },
        "stack_usability": {
            "can_support_one_row_per_sp_reconstruction": bool(not stack_summary_df.empty),
            "can_identify_marginal_offer_bid_candidates": bool(not marginal_df.empty),
        },
        "caveat": {
            "contains_soflag": "soFlag" in cols,
            "interpretation": "This endpoint does not include soFlag, so marginal candidates reflect the highest-priced accepted actions rather than a fully SO-clean economic merit-order marginal unit.",
        },
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("ACCEPTANCES ALL Q1 2026 QUALITY ASSESSMENT")
    print("=" * 72)
    cov = summary["coverage"]
    sch = summary["schema"]
    ident = summary["identifier_quality"]
    stack = summary["stack_usability"]

    print("A. Coverage")
    print(f"- Expected date-SP combinations: {cov['expected_date_sp_combinations']:,}")
    print(f"- Successfully requested: {cov['successfully_requested']:,}")
    print(f"- Non-empty: {cov['non_empty']:,}")
    print(f"- Empty: {cov['empty']:,}")
    print(f"- Request errors: {cov['request_errors']:,}")
    print(f"- Low-row dates: {cov['low_row_dates'][:10]}")
    print(f"- Low-row settlement periods: {cov['low_row_settlement_periods'][:10]}")

    print("\nB. Schema")
    print(f"- Has price fields: {sch['contains_price']}")
    print(f"- Has volume field: {sch['contains_volume']}")
    print(f"- Has acceptance number: {sch['contains_acceptance_number']}")
    print(f"- Has acceptance time: {sch['contains_acceptance_time']}")
    print(f"- Has explicit bid/offer side field: {sch['contains_bid_offer_side_field']}")
    print(f"- Has BMU/plant identifier: {sch['contains_bmu_or_plant_identifier']}")

    print("\nC. Identifier quality")
    print(f"- Best identifier column: {ident['best_identifier_column']}")
    print(f"- Numeric-only identifier share: {ident['numeric_only_identifier_share']}")
    print(f"- Missing identifier share: {ident['missing_identifier_share']}")

    print("\nD. Stack usability")
    print(f"- Can support one-row-per-SP reconstruction: {stack['can_support_one_row_per_sp_reconstruction']}")
    print(f"- Can identify marginal offer and bid candidates: {stack['can_identify_marginal_offer_bid_candidates']}")

    print("\nE. Caveat")
    print(f"- Contains soFlag: {summary['caveat']['contains_soflag']}")
    print(f"- {summary['caveat']['interpretation']}")

    print("\nF. Recommendation")
    should_expand = (
        cov["successfully_requested"] >= 0.98 * cov["expected_date_sp_combinations"]
        and sch["contains_price"]
        and sch["contains_acceptance_number"]
        and sch["contains_acceptance_time"]
        and sch["contains_bmu_or_plant_identifier"]
    )
    print(f"- Expand into main Q1 2026 modelling pipeline? {'Yes, as a candidate stack source for further testing' if should_expand else 'Not yet'}")
    print("- Pair with BOALF later for SO-flag enrichment? Yes")
    print("- Continue with the older SO-filtered but less complete source? Still likely needed for SO-clean interpretations")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    request_log_rows: list[dict[str, Any]] = []

    dates = generate_dates(START_DATE, END_DATE)
    tasks = [(d.isoformat(), sp) for d in dates for sp in SETTLEMENT_PERIODS]
    iterator = tqdm(tasks, desc="Pulling acceptances/all Q1 2026") if tqdm is not None else tasks

    for settlement_date, settlement_period in iterator:
        result = api_request(settlement_date, settlement_period)
        save_raw_response(settlement_date, settlement_period, result)
        request_log_rows.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": settlement_period,
                "status": result["status"],
                "status_code": result["status_code"],
                "error": result["error"],
                "url": result["url"],
                "pull_timestamp": result["pull_timestamp"],
                "row_count": len(result["rows"]),
            }
        )
        all_rows.extend(flatten_rows(settlement_date, settlement_period, result["rows"], result["pull_timestamp"]))
        time.sleep(SLEEP_SECONDS)

    request_log = pd.DataFrame(request_log_rows)
    long_df = normalize_long_df(pd.DataFrame(all_rows))

    if not long_df.empty:
        long_df.to_parquet(LONG_PARQUET, index=False)
        long_df.to_csv(LONG_CSV, index=False)
    else:
        pd.DataFrame().to_csv(LONG_CSV, index=False)

    schema_df = schema_summary(long_df) if not long_df.empty else pd.DataFrame(columns=["column", "dtype", "non_null_count", "null_percentage", "n_unique", "example_values"])
    schema_df.to_csv(SCHEMA_CSV, index=False)

    sp_summary_df, date_summary_df, errors_df = completeness_diagnostics(request_log, long_df)
    sp_summary_df.to_csv(COMPLETENESS_SP_CSV, index=False)
    date_summary_df.to_csv(COMPLETENESS_DATE_CSV, index=False)
    errors_df.to_csv(PULL_ERRORS_CSV, index=False)

    candidate_fields_df = candidate_field_mapping(long_df) if not long_df.empty else pd.DataFrame(columns=["role", "candidate_column", "sample_values"])
    candidate_fields_df.to_csv(CANDIDATE_FIELDS_CSV, index=False)

    identifier_quality_df, top_identifiers_df, unresolved_identifiers_df = identifier_quality(long_df, candidate_fields_df) if not long_df.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    identifier_quality_df.to_csv(IDENTIFIER_QUALITY_CSV, index=False)
    top_identifiers_df.to_csv(TOP_IDENTIFIERS_CSV, index=False)
    unresolved_identifiers_df.to_csv(UNRESOLVED_IDENTIFIERS_CSV, index=False)

    stack_summary_df, marginal_df = stack_diagnostics(long_df) if not long_df.empty else (pd.DataFrame(), pd.DataFrame())
    stack_summary_df.to_csv(STACK_SUMMARY_SP_CSV, index=False)
    marginal_df.to_csv(MARGINAL_CANDIDATES_SP_CSV, index=False)

    marginal_identifier_quality_df, top_marginal_identifiers_df = marginal_identifier_quality(marginal_df) if not marginal_df.empty else (pd.DataFrame(), pd.DataFrame())
    marginal_identifier_quality_df.to_csv(MARGINAL_IDENTIFIER_QUALITY_CSV, index=False)
    top_marginal_identifiers_df.to_csv(TOP_MARGINAL_IDENTIFIERS_CSV, index=False)

    summary = build_summary(
        request_log=request_log,
        long_df=long_df,
        schema_df=schema_df,
        candidate_fields_df=candidate_fields_df,
        identifier_quality_df=identifier_quality_df,
        stack_summary_df=stack_summary_df,
        marginal_df=marginal_df,
    )
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
