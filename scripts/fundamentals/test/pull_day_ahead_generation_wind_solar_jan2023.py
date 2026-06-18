from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "day_ahead_generation_wind_solar"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind-and-solar/day-ahead"
START = "2023-01-01"
END = "2023-01-31"


def fetch_day(day: pd.Timestamp) -> list[dict]:
    date_text = day.strftime("%Y-%m-%d")
    params = {
        "from": date_text,
        "to": date_text,
        "settlementPeriodFrom": 1,
        "settlementPeriodTo": 48,
        "processType": "all",
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return list(payload.get("data", []))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    day = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while day <= end:
        date_text = day.strftime("%Y-%m-%d")
        data = fetch_day(day)
        df = pd.DataFrame(data)
        out_path = RAW_DIR / f"{date_text}.csv"
        if df.empty:
            out_path.write_text("", encoding="utf-8")
        else:
            keep = [
                "publishTime",
                "processType",
                "businessType",
                "psrType",
                "startTime",
                "settlementDate",
                "settlementPeriod",
                "quantity",
            ]
            df = df[[col for col in keep if col in df.columns]]
            df = df.sort_values(["settlementDate", "settlementPeriod", "processType", "psrType", "publishTime"])
            df.to_csv(out_path, index=False)
        rows.append({"settlementDate": date_text, "rawRows": len(df), "rawPath": str(out_path)})
        print(f"{date_text}: rows={len(df)}")
        day += pd.Timedelta(days=1)
        time.sleep(0.1)

    pd.DataFrame(rows).to_csv(DIAG_DIR / "day_ahead_generation_wind_solar_jan2023_pull_summary.csv", index=False)


if __name__ == "__main__":
    main()
