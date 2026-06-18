from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "system_price_niv"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"
START = "2023-01-01"
END = "2023-01-31"


def fetch_day(day: pd.Timestamp) -> list[dict]:
    date_text = day.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{urllib.parse.quote(date_text)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return list(payload.get("data", []))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

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
            preferred = [
                "settlementDate",
                "settlementPeriod",
                "startTime",
                "createdDateTime",
                "systemSellPrice",
                "systemBuyPrice",
                "bsadDefaulted",
                "priceDerivationCode",
                "reserveScarcityPrice",
                "netImbalanceVolume",
                "sellPriceAdjustment",
                "buyPriceAdjustment",
                "replacementPrice",
                "replacementPriceReferenceVolume",
                "totalAcceptedOfferVolume",
                "totalAcceptedBidVolume",
                "totalAdjustmentSellVolume",
                "totalAdjustmentBuyVolume",
                "totalSystemTaggedAcceptedOfferVolume",
                "totalSystemTaggedAcceptedBidVolume",
                "totalSystemTaggedAdjustmentSellVolume",
                "totalSystemTaggedAdjustmentBuyVolume",
            ]
            df = df[[col for col in preferred if col in df.columns]]
            df = df.sort_values(["settlementDate", "settlementPeriod"])
            df.to_csv(out_path, index=False)
        summary.append({"settlementDate": date_text, "rawRows": len(df), "rawPath": str(out_path)})
        print(f"{date_text}: rows={len(df)}")
        day += pd.Timedelta(days=1)
        time.sleep(0.1)

    pd.DataFrame(summary).to_csv(DIAG_DIR / "system_price_niv_jan2023_pull_summary.csv", index=False)


if __name__ == "__main__":
    main()
