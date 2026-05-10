from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / "acceptances_all_probe_2026_01_06"
SUMMARY_JSON = OUT_DIR / "probe_summary.json"

SETTLEMENT_DATE = "2026-01-06"
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/acceptances/all"


def extract_data(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def fetch_settlement_period(settlement_period: int) -> dict[str, Any]:
    params = {
        "settlementDate": SETTLEMENT_DATE,
        "settlementPeriod": settlement_period,
        "format": "json",
    }
    response = requests.get(BASE_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    out_file = OUT_DIR / f"sp_{settlement_period:02d}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    rows = extract_data(payload)
    keys = sorted({key for row in rows for key in row.keys()})

    return {
        "settlementPeriod": settlement_period,
        "url": response.url,
        "row_count": len(rows),
        "keys": keys,
        "rows": rows,
        "output_file": str(out_file),
    }


def summarise_period(period_result: dict[str, Any]) -> dict[str, Any]:
    rows = period_result["rows"]
    action_counter = Counter()
    dataset_counter = Counter()
    bm_units = set()
    prices = []
    volumes = []

    for row in rows:
        action_counter.update([str(row.get("acceptanceType", "UNKNOWN"))])
        datasets = row.get("dataset")
        if datasets is not None:
            dataset_counter.update([str(datasets)])
        bm_unit = row.get("bmUnit")
        if bm_unit:
            bm_units.add(str(bm_unit))
        price = row.get("bidOfferPrice")
        volume = row.get("acceptanceVolume")
        if isinstance(price, (int, float)):
            prices.append(float(price))
        if isinstance(volume, (int, float)):
            volumes.append(float(volume))

    return {
        "settlementPeriod": period_result["settlementPeriod"],
        "row_count": len(rows),
        "n_unique_bm_units": len(bm_units),
        "acceptance_type_counts": dict(sorted(action_counter.items())),
        "dataset_counts": dict(sorted(dataset_counter.items())),
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
        "volume_min": min(volumes) if volumes else None,
        "volume_max": max(volumes) if volumes else None,
        "keys": period_result["keys"],
        "output_file": period_result["output_file"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    period_summaries = []
    all_rows: list[dict[str, Any]] = []

    for sp in range(1, 49):
        print(f"Fetching settlement period {sp:02d}...")
        result = fetch_settlement_period(sp)
        all_rows.extend(result["rows"])
        period_summary = summarise_period(result)
        period_summaries.append(period_summary)
        print(json.dumps(period_summary, indent=2))

    aggregate_action_counter = Counter(str(row.get("acceptanceType", "UNKNOWN")) for row in all_rows)
    aggregate_bmu_counter = Counter(str(row.get("bmUnit")) for row in all_rows if row.get("bmUnit"))
    aggregate_keys = sorted({key for row in all_rows for key in row.keys()})

    summary = {
        "settlementDate": SETTLEMENT_DATE,
        "baseUrl": BASE_URL,
        "n_settlement_periods_requested": 48,
        "n_settlement_periods_with_rows": sum(1 for x in period_summaries if x["row_count"] > 0),
        "total_rows": len(all_rows),
        "all_keys": aggregate_keys,
        "acceptance_type_counts": dict(sorted(aggregate_action_counter.items())),
        "top_bm_units": aggregate_bmu_counter.most_common(20),
        "period_summaries": period_summaries,
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary -> {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
