from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

START_DATE = date(2023, 1, 1)
END_DATE   = date(2025, 12, 31)

BASE_URL        = "https://data.elexon.co.uk/bmrs/api/v1/datasets/DISBSAD"
SLEEP_SECONDS   = 0.05
TIMEOUT_SECONDS = 60
MAX_RETRIES     = 3

OUT_DIR         = PROJECT_ROOT / "data" / "raw" / "ispstack" / "disbsad_2023_2025"
RAW_JSON_DIR    = OUT_DIR / "raw_json"
OUT_PARQUET     = OUT_DIR / "disbsad_2023_2025.parquet"
OUT_CSV         = OUT_DIR / "disbsad_2023_2025.csv"
REQUEST_SUMMARY = OUT_DIR / "disbsad_2023_2025_request_summary.json"

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
            r = SESSION.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
            if r.status_code == 429:
                last_error = f"429 rate limited on attempt {attempt}"
                time.sleep(attempt * 2)
                continue
            r.raise_for_status()
            payload = r.json()
            return {
                "status": "ok",
                "status_code": r.status_code,
                "settlementDate": day.isoformat(),
                "url": str(r.url),
                "rows": extract_data(payload),
                "error": None,
            }
        except Exception as exc:
            last_error = str(exc)
            time.sleep(attempt * 2)
    return {
        "status": "error",
        "status_code": None,
        "settlementDate": day.isoformat(),
        "url": BASE_URL,
        "rows": [],
        "error": last_error,
    }


def json_path(day: date) -> Path:
    return RAW_JSON_DIR / f"{day.isoformat()}.json"


def main() -> None:
    if OUT_PARQUET.exists():
        df = pd.read_parquet(OUT_PARQUET)
        print(f"Parquet already exists: {OUT_PARQUET}  ({len(df):,} rows). Delete to re-fetch.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)

    days = daterange(START_DATE, END_DATE)
    print(f"Fetching DISBSAD {START_DATE} to {END_DATE}  ({len(days)} days) ...")

    request_log: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for i, day in enumerate(days, 1):
        cached = json_path(day)
        if cached.exists():
            result = json.loads(cached.read_text(encoding="utf-8"))
        else:
            result = fetch_day(day)
            cached.write_text(json.dumps(result, indent=2), encoding="utf-8")
            time.sleep(SLEEP_SECONDS)

        request_log.append({
            "settlementDate": result["settlementDate"],
            "status": result["status"],
            "status_code": result["status_code"],
            "row_count": len(result["rows"]),
            "error": result["error"],
        })

        for row in result["rows"]:
            record = dict(row)
            record.setdefault("settlementDate", day.isoformat())
            all_rows.append(record)

        if i == 1 or i % 50 == 0 or i == len(days):
            errors = sum(1 for r in request_log if r["status"] != "ok")
            print(f"  {i:>4}/{len(days)}  rows so far: {len(all_rows):,}  errors: {errors}")

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    REQUEST_SUMMARY.write_text(json.dumps(request_log, indent=2), encoding="utf-8")

    errors_total = sum(1 for r in request_log if r["status"] != "ok")
    print(f"\nDone. {len(df):,} rows, {errors_total} failed days.")
    print(f"Saved -> {OUT_PARQUET}")
    if errors_total:
        print("Failed days:")
        for r in request_log:
            if r["status"] != "ok":
                print(f"  {r['settlementDate']}  {r['error']}")


if __name__ == "__main__":
    main()
