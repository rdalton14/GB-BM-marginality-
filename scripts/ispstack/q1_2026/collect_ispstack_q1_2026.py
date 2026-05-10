"""
collect_ispstack_q1_2026.py
---------------------------
Pulls ISPStack bid + offer data from the Elexon Insights Solution API for
Q1 2026 (2026-01-01 to 2026-03-31), all 48 settlement periods per day.

Endpoint
--------
  GET /balancing/settlement/stack/all/{offer|bid}/{date}/{sp}

Output
------
  data/raw/ispstack/q1_2026/YYYY-MM-DD.parquet   (one file per date, bid+offer combined)
  outputs/logs/errors_q1_2026.log

Checkpointing
-------------
  Dates already saved to disk are skipped automatically.
  Re-run after any interruption to resume from where it stopped.

Total requests: 90 days x 48 SPs x 2 directions = 8,640
Est. runtime  : ~15 minutes at 0.1 s between requests
"""

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
OUT_DIR  = PROJECT_ROOT / "data" / "raw" / "ispstack" / "q1_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = PROJECT_ROOT / "outputs" / "logs" / "errors_q1_2026.log"
ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(ERROR_LOG),
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BASE_URL   = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/stack/all"
START_DATE = date(2026, 1, 1)
END_DATE   = date(2026, 3, 31)
SPS        = list(range(1, 49))
DIRECTIONS = ("offer", "bid")
SLEEP      = 0.1

ALL_DATES: list[date] = []
d = START_DATE
while d <= END_DATE:
    ALL_DATES.append(d)
    d += timedelta(days=1)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch(url: str) -> list[dict] | None:
    try:
        resp = SESSION.get(url, params={"format": "json"}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        return data if isinstance(data, list) else []
    except Exception as exc:
        logging.error("GET %s  error=%s", url, exc)
        return None


def fetch_date(day: date) -> tuple[pd.DataFrame | None, list[str]]:
    """Pull all 48 SPs x 2 directions for one date. Returns (df, failed_urls)."""
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
                    row["settlementDate"]   = row.get("settlementDate",   date_str)
                    row["settlementPeriod"] = row.get("settlementPeriod", sp)
                    row["direction"]        = direction
                    rows.append(row)
            time.sleep(SLEEP)

    if not rows:
        return None, failed

    return pd.DataFrame(rows), failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("=" * 60)
    print("ISPStack Q1 2026 collection")
    print(f"  {len(ALL_DATES)} dates  x  48 SPs  x  2 directions  =  "
          f"{len(ALL_DATES)*48*2:,} requests")
    print("=" * 60)

    total_rows   = 0
    failed_dates = []

    for day in ALL_DATES:
        date_str  = day.isoformat()
        out_path  = OUT_DIR / f"{date_str}.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            total_rows += len(existing)
            print(f"[SKIP] {date_str} — already on disk ({len(existing):,} rows)")
            continue

        df, failures = fetch_date(day)
        print()   # newline after \r progress

        if df is not None and not df.empty:
            df.to_parquet(out_path, index=False)
            total_rows += len(df)
            status = f"{len(df):,} rows saved"
            if failures:
                status += f"  (partial failures: {failures})"
                failed_dates.append(f"{date_str}: {failures}")
                logging.error("Partial failure  date=%s  failed=%s", date_str, failures)
        else:
            failed_dates.append(date_str)
            logging.error("No data returned  date=%s  failures=%s", date_str, failures)
            print(f"[FAIL] {date_str} — no data (see errors.log)")
            continue

        print(f"[DONE] {date_str}: {status}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    files = list(OUT_DIR.glob("*.parquet"))
    print(f"  Parquet files saved : {len(files)}")
    print(f"  Total rows          : {total_rows:,}")
    if failed_dates:
        print(f"  Failed dates ({len(failed_dates)}):")
        for f in failed_dates:
            print(f"    {f}")
    else:
        print("  No failures.")
    print(f"  Error log           : {ERROR_LOG}")
    print()
    print("Next step: update INPUT_PATH in process_q1_2026.py to:")
    print(f"  DATA_DIR / 'raw' / 'ispstack'")
    print("  (the script reads an entire folder of parquet files)")


if __name__ == "__main__":
    run()
