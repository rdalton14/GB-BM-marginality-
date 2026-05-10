from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

RUN_LABEL = "2026_01_05_to_2026_01_11"
OUT_DIR = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / f"ispstack_one_week_probe_{RUN_LABEL}"
DAILY_DIR = OUT_DIR / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = PROJECT_ROOT / "outputs" / "logs" / f"errors_ispstack_one_week_probe_{RUN_LABEL}.log"
ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(ERROR_LOG),
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/stack/all"
START_DATE = date(2026, 1, 5)
END_DATE = date(2026, 1, 11)
SPS = list(range(1, 49))
DIRECTIONS = ("offer", "bid")
SLEEP = 0.05

ALL_DATES: list[date] = []
d = START_DATE
while d <= END_DATE:
    ALL_DATES.append(d)
    d += timedelta(days=1)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def fetch(url: str) -> list[dict] | None:
    try:
        resp = SESSION.get(url, params={"format": "json"}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        return data if isinstance(data, list) else []
    except Exception as exc:
        logging.error("GET %s error=%s", url, exc)
        return None


def fetch_date(day: date) -> tuple[pd.DataFrame | None, list[str]]:
    date_str = day.isoformat()
    rows: list[dict] = []
    failed: list[str] = []

    for sp in SPS:
        for direction in DIRECTIONS:
            url = f"{BASE_URL}/{direction}/{date_str}/{sp}"
            print(f"  {direction:5s} {date_str} SP {sp:>2}/48", end="\r", flush=True)
            data = fetch(url)
            if data is None:
                failed.append(f"{direction} SP{sp}")
            else:
                for row in data:
                    row["settlementDate"] = row.get("settlementDate", date_str)
                    row["settlementPeriod"] = row.get("settlementPeriod", sp)
                    row["direction"] = direction
                    rows.append(row)
            time.sleep(SLEEP)

    if not rows:
        return None, failed

    return pd.DataFrame(rows), failed


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    total_rows = len(df)
    for column in sorted(df.columns):
        series = df[column]
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        record = {
            "column": column,
            "dtype": str(series.dtype),
            "non_null_count": non_null,
            "null_count": null_count,
            "null_share": null_count / total_rows if total_rows else None,
            "n_unique_non_null": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            record["min"] = float(numeric.min()) if numeric.notna().any() else None
            record["max"] = float(numeric.max()) if numeric.notna().any() else None
        else:
            uniques = series.dropna().astype(str).unique().tolist()
            record["sample_values"] = " | ".join(uniques[:5])
        records.append(record)
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, failed_dates: list[str]) -> dict:
    summary: dict[str, object] = {
        "run_label": RUN_LABEL,
        "date_from": START_DATE.isoformat(),
        "date_to": END_DATE.isoformat(),
        "n_dates": len(ALL_DATES),
        "n_requests_expected": len(ALL_DATES) * len(SPS) * len(DIRECTIONS),
        "total_rows": int(len(df)),
        "columns": sorted(df.columns.tolist()),
        "rows_by_direction": df["direction"].value_counts(dropna=False).sort_index().to_dict() if "direction" in df.columns else {},
        "rows_by_date": df.groupby("settlementDate").size().to_dict() if "settlementDate" in df.columns else {},
        "rows_by_side_and_date": (
            df.groupby(["settlementDate", "direction"]).size().reset_index(name="row_count").to_dict(orient="records")
            if {"settlementDate", "direction"}.issubset(df.columns) else []
        ),
        "repriced_indicator_share": (
            float(df["repricedIndicator"].fillna(False).astype(bool).mean()) if "repricedIndicator" in df.columns else None
        ),
        "so_flag_share": (
            float(df["soFlag"].fillna(False).astype(bool).mean()) if "soFlag" in df.columns else None
        ),
        "cadl_flag_share": (
            float(df["cadlFlag"].fillna(False).astype(bool).mean()) if "cadlFlag" in df.columns else None
        ),
        "stor_provider_flag_share": (
            float(df["storProviderFlag"].fillna(False).astype(bool).mean()) if "storProviderFlag" in df.columns else None
        ),
        "failed_dates": failed_dates,
    }

    if "sequenceNumber" in df.columns:
        seq_dupes = (
            df.groupby(["settlementDate", "settlementPeriod", "direction", "sequenceNumber"])
            .size()
            .gt(1)
            .sum()
        )
        summary["duplicate_sequence_number_keys"] = int(seq_dupes)

    for col in ["originalPrice", "finalPrice", "volume", "dmatAdjustedVolume", "arbitrageAdjustedVolume", "nivAdjustedVolume", "parAdjustedVolume"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            summary[f"{col}_min"] = float(numeric.min()) if numeric.notna().any() else None
            summary[f"{col}_max"] = float(numeric.max()) if numeric.notna().any() else None

    return summary


def run() -> None:
    print("=" * 60)
    print("ISPSTACK one-week probe collection")
    print(f"  {START_DATE.isoformat()} to {END_DATE.isoformat()}")
    print(f"  {len(ALL_DATES)} dates x 48 SPs x 2 directions = {len(ALL_DATES) * 48 * 2:,} requests")
    print(f"  Output dir: {OUT_DIR}")
    print("=" * 60)

    daily_frames: list[pd.DataFrame] = []
    failed_dates: list[str] = []

    for day in ALL_DATES:
        date_str = day.isoformat()
        out_path = DAILY_DIR / f"{date_str}.parquet"
        df, failures = fetch_date(day)
        print()

        if df is None or df.empty:
            failed_dates.append(date_str)
            logging.error("No data returned date=%s failures=%s", date_str, failures)
            print(f"[FAIL] {date_str} - no data")
            continue

        df.to_parquet(out_path, index=False)
        daily_frames.append(df)
        status = f"{len(df):,} rows saved"
        if failures:
            failed_dates.append(f"{date_str}: {failures}")
            logging.error("Partial failure date=%s failures=%s", date_str, failures)
            status += f" (partial failures: {failures})"
        print(f"[DONE] {date_str}: {status}")

    if not daily_frames:
        raise RuntimeError("No ISPSTACK data collected for the one-week probe.")

    full_df = pd.concat(daily_frames, ignore_index=True)
    offer_df = full_df[full_df["direction"] == "offer"].copy()
    bid_df = full_df[full_df["direction"] == "bid"].copy()

    offer_df.to_parquet(OUT_DIR / "ispstack_one_week_offer_raw.parquet", index=False)
    bid_df.to_parquet(OUT_DIR / "ispstack_one_week_bid_raw.parquet", index=False)
    full_df.to_parquet(OUT_DIR / "ispstack_one_week_long.parquet", index=False)

    offer_df.to_csv(OUT_DIR / "ispstack_one_week_offer_raw.csv", index=False)
    bid_df.to_csv(OUT_DIR / "ispstack_one_week_bid_raw.csv", index=False)
    full_df.to_csv(OUT_DIR / "ispstack_one_week_long.csv", index=False)

    column_profile = build_column_profile(full_df)
    column_profile.to_csv(OUT_DIR / "ispstack_one_week_column_profile.csv", index=False)

    summary = summarize(full_df, failed_dates)
    with (OUT_DIR / "ispstack_one_week_probe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("ONE-WEEK ISPSTACK PROBE COMPLETE")
    print("=" * 60)
    print(f"  Total rows         : {len(full_df):,}")
    print(f"  Offer rows         : {len(offer_df):,}")
    print(f"  Bid rows           : {len(bid_df):,}")
    print(f"  Failed dates       : {len(failed_dates)}")
    print(f"  Summary JSON       : {OUT_DIR / 'ispstack_one_week_probe_summary.json'}")
    print(f"  Column profile CSV : {OUT_DIR / 'ispstack_one_week_column_profile.csv'}")
    print(f"  Error log          : {ERROR_LOG}")


if __name__ == "__main__":
    run()
