from __future__ import annotations

import json
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / "boal_bod_probe_2026_01_06"
SUMMARY_JSON = OUT_DIR / "probe_summary.json"

DATE_FROM = "2026-01-06T00:00:00"
DATE_TO = "2026-01-07T00:00:00"

DATASETS = {
    "BOAL": "https://bmrs.elexon.co.uk/bmrs/api/v1/datasets/BOAL",
    "BOD": "https://bmrs.elexon.co.uk/bmrs/api/v1/datasets/BOD",
}


def fetch_dataset(name: str, base_url: str) -> dict[str, object]:
    params = {
        "publishDateTimeFrom": DATE_FROM,
        "publishDateTimeTo": DATE_TO,
        "format": "json",
    }
    response = requests.get(base_url, params=params, timeout=120)
    body_path = OUT_DIR / f"{name.lower()}_response.json"

    payload: object
    try:
        payload = response.json()
    except Exception:
        payload = {"raw_text": response.text}

    with body_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if isinstance(payload, list):
        row_count = len(payload)
        sample_keys = sorted({k for row in payload[:10] if isinstance(row, dict) for k in row.keys()})
    elif isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            row_count = len(data)
            sample_keys = sorted({k for row in data[:10] if isinstance(row, dict) for k in row.keys()})
        else:
            row_count = None
            sample_keys = sorted(payload.keys())
    else:
        row_count = None
        sample_keys = []

    return {
        "dataset": name,
        "url": response.url,
        "status_code": response.status_code,
        "row_count_guess": row_count,
        "sample_keys": sample_keys,
        "output_file": str(body_path),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, url in DATASETS.items():
        print(f"Fetching {name} ...")
        result = fetch_dataset(name, url)
        summary.append(result)
        print(json.dumps(result, indent=2))

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary -> {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
