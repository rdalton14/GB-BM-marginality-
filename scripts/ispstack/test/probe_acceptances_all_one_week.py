from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
OUT_DIR = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / "acceptances_all_probe_2026_01_05_to_2026_01_11"
RAW_JSON_DIR = OUT_DIR / "raw_json"
LONG_CSV = OUT_DIR / "acceptances_all_long_2026_01_05_2026_01_11.csv"
LONG_PARQUET = OUT_DIR / "acceptances_all_long_2026_01_05_2026_01_11.parquet"
SCHEMA_CSV = OUT_DIR / "acceptances_all_schema_summary.csv"
MARGINAL_TEST_CSV = OUT_DIR / "marginal_action_test_2026_01_05_2026_01_11.csv"
IDENTIFIER_QUALITY_CSV = OUT_DIR / "generator_identifier_quality_summary.csv"
REQUEST_SUMMARY_JSON = OUT_DIR / "request_summary.json"

START_DATE = date(2026, 1, 5)
END_DATE = date(2026, 1, 11)
SETTLEMENT_PERIODS = list(range(1, 49))
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/acceptances/all"
SOURCE_ENDPOINT = "acceptances_all"
SLEEP_SECONDS = 0.10
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60

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


def fetch_period(settlement_date: str, settlement_period: int) -> dict[str, Any]:
    url = f"{BASE_URL}/{settlement_date}/{settlement_period}"
    params = {"format": "json"}
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=TIMEOUT_SECONDS)
            status_code = response.status_code

            if status_code == 404:
                return {
                    "status": "not_found",
                    "status_code": status_code,
                    "settlementDate": settlement_date,
                    "settlementPeriod": settlement_period,
                    "url": response.url,
                    "payload": None,
                    "rows": [],
                    "error": "404 not found",
                }

            if status_code == 429:
                wait_seconds = attempt
                time.sleep(wait_seconds)
                last_error = f"429 rate limited on attempt {attempt}"
                continue

            response.raise_for_status()
            payload = response.json()
            rows = extract_data(payload)
            return {
                "status": "ok",
                "status_code": status_code,
                "settlementDate": settlement_date,
                "settlementPeriod": settlement_period,
                "url": response.url,
                "payload": payload,
                "rows": rows,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(attempt)

    return {
        "status": "error",
        "status_code": None,
        "settlementDate": settlement_date,
        "settlementPeriod": settlement_period,
        "url": url,
        "payload": None,
        "rows": [],
        "error": last_error,
    }


def save_raw_payload(result: dict[str, Any]) -> None:
    day_dir = RAW_JSON_DIR / result["settlementDate"]
    day_dir.mkdir(parents=True, exist_ok=True)
    out_path = day_dir / f"sp_{int(result['settlementPeriod']):02d}.json"
    payload = {
        "status": result["status"],
        "status_code": result["status_code"],
        "settlementDate": result["settlementDate"],
        "settlementPeriod": result["settlementPeriod"],
        "url": result["url"],
        "error": result["error"],
        "payload": result["payload"],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result["rows"]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record.setdefault("settlementDate", result["settlementDate"])
        record.setdefault("settlementPeriod", result["settlementPeriod"])
        record["settlementDate"] = result["settlementDate"]
        record["settlementPeriod"] = int(result["settlementPeriod"])
        record["source_endpoint"] = SOURCE_ENDPOINT
        normalized.append(record)
    return normalized


def is_numeric_only(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip()
    return bool(text) and bool(re.fullmatch(r"\d+", text))


def identify_candidate_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = [str(c) for c in df.columns]
    lowered = {c: c.lower() for c in cols}

    def pick(*keywords: str) -> list[str]:
        out = []
        for col in cols:
            name = lowered[col]
            if all(k in name for k in keywords):
                out.append(col)
        return out

    candidates = {
        "bmu_or_asset_id": [c for c in cols if any(k in lowered[c] for k in ["bmunit", "bm_unit", "asset", "unit", "id"])],
        "acceptance_number": [c for c in cols if "acceptance" in lowered[c] and any(k in lowered[c] for k in ["number", "id"])],
        "acceptance_time": [c for c in cols if "acceptance" in lowered[c] and "time" in lowered[c]],
        "bid_offer_pair_number": [c for c in cols if "bidofferpair" in lowered[c] or ("pair" in lowered[c] and "bid" in lowered[c])],
        "bid_offer_side": [c for c in cols if any(k in lowered[c] for k in ["type", "side", "bid", "offer"])],
        "accepted_price": [c for c in cols if "price" in lowered[c]],
        "accepted_volume": [c for c in cols if "volume" in lowered[c]],
        "action_type_flag": [c for c in cols if any(k in lowered[c] for k in ["flag", "type", "dataset"])],
        "stack_or_numeric_identifier": [c for c in cols if any(k in lowered[c] for k in ["id", "number", "pair"])],
    }
    return candidates


def print_candidate_samples(df: pd.DataFrame, candidates: dict[str, list[str]]) -> None:
    print("\nCandidate columns and sample values:")
    for role, cols in candidates.items():
        if not cols:
            print(f"- {role}: none found")
            continue
        print(f"- {role}:")
        for col in cols[:8]:
            samples = df[col].dropna().astype(str).head(5).tolist()
            print(f"    {col}: {samples}")


def infer_identifier_column(df: pd.DataFrame) -> str | None:
    preferred = ["bmUnit", "bmUnitId", "assetId", "id", "nationalGridBmUnit"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if "unit" in str(col).lower() or str(col).lower().endswith("id"):
            return str(col)
    return None


def infer_side_column(df: pd.DataFrame) -> str | None:
    preferred = ["acceptanceType", "bidOfferType", "direction", "side"]
    for col in preferred:
        if col in df.columns:
            return col
    return None


def infer_side(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    out[text.str.contains("offer", na=False)] = "offer"
    out[text.str.contains("bid", na=False)] = "bid"
    return out


def choose_price_column(df: pd.DataFrame) -> str | None:
    preferred = ["bidOfferPrice", "acceptedPrice", "finalPrice", "price"]
    for col in preferred:
        if col in df.columns:
            return col
    price_cols = [c for c in df.columns if "price" in str(c).lower()]
    return price_cols[0] if price_cols else None


def choose_volume_column(df: pd.DataFrame) -> str | None:
    preferred = ["acceptanceVolume", "acceptedVolume", "volume"]
    for col in preferred:
        if col in df.columns:
            return col
    volume_cols = [c for c in df.columns if "volume" in str(c).lower()]
    return volume_cols[0] if volume_cols else None


def build_schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        non_null = df[col].dropna()
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missingCount": int(df[col].isna().sum()),
                "missingShare": float(df[col].isna().mean()),
                "nUniqueNonNull": int(non_null.nunique()),
                "sampleValues": " | ".join(non_null.astype(str).head(5).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["missingShare", "column"], ascending=[False, True])


def build_identifier_quality_summary(df: pd.DataFrame, identifier_col: str | None, price_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    if identifier_col is None:
        rows.append({"metric": "identifier_column_found", "value": "False"})
        return pd.DataFrame(rows), pd.DataFrame()

    identifier = df[identifier_col].astype(str)
    numeric_mask = identifier.map(is_numeric_only)

    rows.extend(
        [
            {"metric": "identifier_column", "value": identifier_col},
            {"metric": "unique_identifiers", "value": int(identifier.nunique())},
            {"metric": "numeric_only_unique_identifiers", "value": int(identifier[numeric_mask].nunique())},
            {"metric": "numeric_only_row_count", "value": int(numeric_mask.sum())},
            {"metric": "numeric_only_row_share", "value": float(numeric_mask.mean())},
        ]
    )

    offer_numeric_sp_share = None
    if "inferred_action_side" in df.columns and price_col is not None:
        offers = df[df["inferred_action_side"] == "offer"].copy()
        if not offers.empty:
            offers["_numeric_only_id"] = offers[identifier_col].map(is_numeric_only)
            offers["_price_numeric"] = pd.to_numeric(offers[price_col], errors="coerce")
            offers = offers[offers["_price_numeric"].notna()]
            if not offers.empty:
                top_offer = (
                    offers.sort_values(["settlementDate", "settlementPeriod", "_price_numeric"], ascending=[True, True, False])
                    .groupby(["settlementDate", "settlementPeriod"], as_index=False)
                    .first()
                )
                offer_numeric_sp_share = float(top_offer["_numeric_only_id"].mean())
                rows.append({"metric": "share_of_sps_where_highest_priced_offer_id_is_numeric_only", "value": offer_numeric_sp_share})

    top_identifiers = (
        df.assign(identifier_numeric_only=df[identifier_col].map(is_numeric_only))
        .groupby([identifier_col, "identifier_numeric_only"])
        .size()
        .reset_index(name="rowCount")
        .sort_values("rowCount", ascending=False)
        .head(50)
    )
    return pd.DataFrame(rows), top_identifiers


def build_marginal_action_test(df: pd.DataFrame, identifier_col: str | None, price_col: str | None, volume_col: str | None) -> tuple[pd.DataFrame, str]:
    if identifier_col is None or price_col is None or volume_col is None or "inferred_action_side" not in df.columns:
        return pd.DataFrame(), "Not enough side/price/volume/identifier information to reconstruct marginal actions."

    work = df.copy()
    work["_price_numeric"] = pd.to_numeric(work[price_col], errors="coerce")
    work["_volume_numeric"] = pd.to_numeric(work[volume_col], errors="coerce")
    work = work[work["_price_numeric"].notna()]

    records: list[dict[str, Any]] = []
    for (settlement_date, settlement_period), grp in work.groupby(["settlementDate", "settlementPeriod"], sort=True):
        offers = grp[grp["inferred_action_side"] == "offer"].sort_values("_price_numeric", ascending=False)
        bids = grp[grp["inferred_action_side"] == "bid"].sort_values("_price_numeric", ascending=False)

        offer_top = offers.iloc[0] if not offers.empty else None
        bid_top = bids.iloc[0] if not bids.empty else None

        records.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": int(settlement_period),
                "marginal_offer_identifier": None if offer_top is None else offer_top[identifier_col],
                "marginal_offer_price": None if offer_top is None else float(offer_top["_price_numeric"]),
                "marginal_offer_volume": None if offer_top is None else float(offer_top["_volume_numeric"]) if pd.notna(offer_top["_volume_numeric"]) else None,
                "marginal_bid_identifier": None if bid_top is None else bid_top[identifier_col],
                "marginal_bid_price": None if bid_top is None else float(bid_top["_price_numeric"]),
                "marginal_bid_volume": None if bid_top is None else float(bid_top["_volume_numeric"]) if pd.notna(bid_top["_volume_numeric"]) else None,
                "number_of_accepted_offer_actions": int(len(offers)),
                "number_of_accepted_bid_actions": int(len(bids)),
            }
        )

    convention = (
        "Bid-side convention used: highest accepted bid-side price within the inferred bid subset, "
        "to mirror the edge-of-accepted-stack logic used in the earlier ISPSTACK prototype."
    )
    return pd.DataFrame(records), convention


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)

    request_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []

    all_days = daterange(START_DATE, END_DATE)
    for current_day in all_days:
        date_str = current_day.isoformat()
        for sp in SETTLEMENT_PERIODS:
            print(f"Fetching {date_str} SP {sp:02d}/48", end="\r", flush=True)
            result = fetch_period(date_str, sp)
            save_raw_payload(result)
            request_rows.append(
                {
                    "settlementDate": date_str,
                    "settlementPeriod": sp,
                    "status": result["status"],
                    "status_code": result["status_code"],
                    "row_count": len(result["rows"]),
                    "error": result["error"],
                    "url": result["url"],
                }
            )
            normalized_rows.extend(normalize_rows(result))
            time.sleep(SLEEP_SECONDS)
    print()

    long_df = pd.DataFrame(normalized_rows)
    request_df = pd.DataFrame(request_rows)

    if not long_df.empty:
        long_df.to_csv(LONG_CSV, index=False)
        long_df.to_parquet(LONG_PARQUET, index=False)
    else:
        pd.DataFrame().to_csv(LONG_CSV, index=False)

    print("=" * 72)
    print("ACCEPTANCES ALL ONE-WEEK PROBE")
    print("=" * 72)
    print(f"Rows returned: {len(long_df):,}")
    print(f"Settlement periods with rows: {request_df.loc[request_df['row_count'] > 0, ['settlementDate', 'settlementPeriod']].shape[0]:,}")

    if long_df.empty:
        with REQUEST_SUMMARY_JSON.open("w", encoding="utf-8") as f:
            json.dump({"message": "No rows returned", "requests": request_rows}, f, indent=2)
        print("No data returned; inspect raw JSON and request summary.")
        return

    print("\nAll returned columns:")
    for col in long_df.columns:
        print(f"- {col}")

    print("\nExample rows:")
    sample_sps = (
        long_df[["settlementDate", "settlementPeriod"]]
        .drop_duplicates()
        .sort_values(["settlementDate", "settlementPeriod"])
        .head(3)
    )
    for _, sample_row in sample_sps.iterrows():
        sub = long_df[
            (long_df["settlementDate"] == sample_row["settlementDate"])
            & (long_df["settlementPeriod"] == sample_row["settlementPeriod"])
        ].head(3)
        print(f"\nSample {sample_row['settlementDate']} SP {int(sample_row['settlementPeriod'])}:")
        print(sub.to_string())

    schema_summary = build_schema_summary(long_df)
    schema_summary.to_csv(SCHEMA_CSV, index=False)
    print("\nMissingness summary by column:")
    print(schema_summary[["column", "missingCount", "missingShare"]].to_string(index=False))

    candidates = identify_candidate_columns(long_df)
    print_candidate_samples(long_df, candidates)

    identifier_col = infer_identifier_column(long_df)
    side_col = infer_side_column(long_df)
    price_col = choose_price_column(long_df)
    volume_col = choose_volume_column(long_df)

    if side_col is not None:
        long_df["inferred_action_side"] = infer_side(long_df[side_col])
    else:
        long_df["inferred_action_side"] = pd.NA

    quality_summary, top_identifiers = build_identifier_quality_summary(long_df, identifier_col, price_col)
    quality_summary.to_csv(IDENTIFIER_QUALITY_CSV, index=False)

    marginal_test, bid_convention = build_marginal_action_test(long_df, identifier_col, price_col, volume_col)
    marginal_test.to_csv(MARGINAL_TEST_CSV, index=False)

    request_summary = {
        "startDate": START_DATE.isoformat(),
        "endDate": END_DATE.isoformat(),
        "requests_total": int(len(request_df)),
        "requests_ok": int((request_df["status"] == "ok").sum()),
        "requests_not_found": int((request_df["status"] == "not_found").sum()),
        "requests_error": int((request_df["status"] == "error").sum()),
        "rows_total": int(len(long_df)),
        "column_count": int(len(long_df.columns)),
        "identifier_column": identifier_col,
        "side_column": side_col,
        "price_column": price_col,
        "volume_column": volume_col,
        "bid_convention": bid_convention,
    }
    with REQUEST_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(request_summary, f, indent=2)

    print("\nIdentifier quality summary:")
    print(quality_summary.to_string(index=False))
    if not top_identifiers.empty:
        print("\nTop identifier frequency sample:")
        print(top_identifiers.head(20).to_string(index=False))

    print("\nFinal summary:")
    usable_identifier = identifier_col is not None and not long_df[identifier_col].isna().all()
    numeric_row_share = None
    if not quality_summary.empty and "numeric_only_row_share" in quality_summary["metric"].values:
        numeric_row_share = float(
            quality_summary.loc[quality_summary["metric"] == "numeric_only_row_share", "value"].iloc[0]
        )

    if usable_identifier:
        print(f"- Usable BMU/generator identifier column appears to be: {identifier_col}")
    else:
        print("- No clearly usable BMU/generator identifier column was found.")

    if numeric_row_share is not None:
        print(f"- Numeric-only identifier row share: {numeric_row_share:.4f}")
        print(
            "- Compared with the earlier ISPSTACK-derived issue, this probe is better only if that share is materially lower "
            "and the highest-priced accepted offers stop clustering on numeric-only identifiers."
        )

    enough_for_stack = side_col is not None and price_col is not None and volume_col is not None
    print(f"- Side column detected: {side_col}")
    print(f"- Price column detected: {price_col}")
    print(f"- Volume column detected: {volume_col}")
    print(f"- Enough information to reconstruct offer/bid stacks: {'yes' if enough_for_stack else 'no'}")
    print(f"- Recommendation on Q1 expansion: {'yes, if identifiers look cleaner and side/price/volume are coherent' if enough_for_stack else 'not yet'}")


if __name__ == "__main__":
    main()
