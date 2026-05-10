from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "fundamentals"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "fundamentals"
COVERAGE_DIR = PROJECT_ROOT / "data" / "diagnostics" / "coverage"
MISSINGNESS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "missingness"
START = date(2023, 1, 1)
END = date(2025, 12, 31)
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
MISSINGNESS_DIR.mkdir(parents=True, exist_ok=True)


def all_dates() -> list[date]:
    out: list[date] = []
    d = START
    while d <= END:
        out.append(d)
        d += timedelta(days=1)
    return out


DATES = all_dates()


def missing_keys(df: pd.DataFrame) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product(
        [[d.isoformat() for d in DATES], list(range(1, 49))],
        names=["settlementDate", "settlementPeriod"],
    )
    actual = pd.MultiIndex.from_frame(df[["settlementDate", "settlementPeriod"]].drop_duplicates())
    missing = expected.difference(actual)
    if len(missing) == 0:
        return pd.DataFrame(columns=["settlementDate", "settlementPeriod"])
    return missing.to_frame(index=False)


def save_diagnostics(df: pd.DataFrame, stem: str) -> None:
    (
        df.groupby("settlementDate")
        .size()
        .reset_index(name="rows")
        .sort_values("settlementDate")
        .to_csv(COVERAGE_DIR / f"{stem}_daily_completeness.csv", index=False)
    )
    missing_keys(df).to_csv(MISSINGNESS_DIR / f"{stem}_missing_sps.csv", index=False)


def fetch_json(url: str) -> list[dict]:
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    return data if isinstance(data, list) else []


def build_interconnector_flows() -> None:
    raw_dir = RAW_ROOT / "interconnector_flows"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    base_url = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/interconnectors"

    for d in DATES:
        date_str = d.isoformat()
        raw_path = raw_dir / f"{date_str}.csv"
        if raw_path.exists() and raw_path.stat().st_size > 0:
            day = pd.read_csv(raw_path)
        else:
            day = pd.DataFrame(fetch_json(f"{base_url}?settlementDateFrom={date_str}&settlementDateTo={date_str}"))
            day.to_csv(raw_path, index=False)
        rows.append(day)

    raw = pd.concat(rows, ignore_index=True)
    raw["settlementPeriod"] = raw["settlementPeriod"].astype(int)
    raw["generation"] = raw["generation"].astype(float)
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    latest = (
        raw.sort_values("publishTime", ascending=False)
        .drop_duplicates(["settlementDate", "settlementPeriod", "interconnectorName"])
    )
    processed = (
        latest.groupby(["settlementDate", "settlementPeriod"], as_index=False)
        .agg(netInterconnectorFlow=("generation", "sum"), interconnectorCount=("interconnectorName", "size"))
        .sort_values(["settlementDate", "settlementPeriod"])
    )
    processed.to_csv(PROCESSED_DIR / "interconnector_flows_2023_2025.csv", index=False)
    save_diagnostics(processed, "interconnector_flows_2023_2025")


def build_lolpdrm() -> None:
    raw_dir = RAW_ROOT / "lolpdrm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    base_url = "https://data.elexon.co.uk/bmrs/api/v1/forecast/system/loss-of-load"

    for d in DATES:
        date_str = d.isoformat()
        raw_path = raw_dir / f"{date_str}.csv"
        if raw_path.exists() and raw_path.stat().st_size > 0:
            day = pd.read_csv(raw_path)
        else:
            day = pd.DataFrame(fetch_json(f"{base_url}?from={date_str}&to={date_str}&settlementPeriodFrom=1&settlementPeriodTo=48"))
            day.to_csv(raw_path, index=False)
        rows.append(day)

    raw = pd.concat(rows, ignore_index=True)
    raw["settlementPeriod"] = raw["settlementPeriod"].astype(int)
    raw["forecastHorizon"] = raw["forecastHorizon"].astype(int)
    raw["lossOfLoadProbability"] = raw["lossOfLoadProbability"].astype(float)
    raw["deratedMargin"] = raw["deratedMargin"].astype(float)
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    processed = (
        raw.loc[raw["forecastHorizon"] == 2]
        .sort_values("publishTime", ascending=False)
        .drop_duplicates(["settlementDate", "settlementPeriod"])
        .sort_values(["settlementDate", "settlementPeriod"])[
            [
                "settlementDate",
                "settlementPeriod",
                "lossOfLoadProbability",
                "deratedMargin",
                "forecastHorizon",
                "publishTime",
                "publishingPeriodCommencingTime",
                "startTime",
            ]
        ]
    )
    processed.to_csv(PROCESSED_DIR / "lolpdrm_2023_2025.csv", index=False)
    save_diagnostics(processed, "lolpdrm_2023_2025")


def build_apx_mid() -> None:
    raw_dir = RAW_ROOT / "apx_mid"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    base_url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID"

    for d in DATES:
        date_str = d.isoformat()
        raw_path = raw_dir / f"{date_str}.csv"
        if raw_path.exists() and raw_path.stat().st_size > 0:
            day = pd.read_csv(raw_path)
        else:
            day = pd.DataFrame(fetch_json(f"{base_url}?from={date_str}&to={date_str}&settlementPeriodFrom=1&settlementPeriodTo=48&dataProviders=APXMIDP"))
            day.to_csv(raw_path, index=False)
        rows.append(day)

    raw = pd.concat(rows, ignore_index=True)
    raw["settlementPeriod"] = raw["settlementPeriod"].astype(int)
    raw["price"] = raw["price"].astype(float)
    raw["volume"] = raw["volume"].astype(float)
    processed = (
        raw.sort_values(["settlementDate", "settlementPeriod"])[
            ["settlementDate", "settlementPeriod", "price", "volume", "dataProvider"]
        ]
        .rename(columns={"price": "marketIndexPrice", "volume": "marketIndexVolume"})
    )
    processed.to_csv(PROCESSED_DIR / "apx_mid_2023_2025.csv", index=False)
    save_diagnostics(processed, "apx_mid_2023_2025")


if __name__ == "__main__":
    print("Building remaining full-history fundamentals: interconnector_flows, lolpdrm, apx_mid")
    build_interconnector_flows()
    build_lolpdrm()
    build_apx_mid()
    print("Remaining full-history fundamentals complete.")
