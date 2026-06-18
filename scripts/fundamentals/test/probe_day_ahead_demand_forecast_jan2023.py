from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "test" / "fundamentals" / "day_ahead_demand_forecast_jan2023_probe"
EVOLUTION_DIR = OUT_DIR / "evolution"
DIRECT_DIR = OUT_DIR / "direct_day_ahead"
TOTAL_LOAD_DIR = OUT_DIR / "total_day_ahead_load"
REPORT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "baseline_forecasting_fundamentals" / "endpoint_probes"

EVOLUTION_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/evolution"
DIRECT_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead"
TOTAL_LOAD_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/total/day-ahead"
START = date(2023, 1, 1)
END = date(2023, 1, 7)


def days() -> list[date]:
    out: list[date] = []
    d = START
    while d <= END:
        out.append(d)
        d += timedelta(days=1)
    return out


def payload_data(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        data = payload.get("data", payload)
    else:
        data = payload
    return data if isinstance(data, list) else []


def fetch_json(session: requests.Session, url: str) -> tuple[int, object]:
    resp = session.get(url, timeout=60)
    status = resp.status_code
    try:
        payload: object = resp.json()
    except ValueError:
        payload = {"text": resp.text[:1000]}
    resp.raise_for_status()
    return status, payload


def flatten_rows(rows: list[dict], source: str, url: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    df.insert(0, "source", source)
    df.insert(1, "requestUrl", url)
    return df


def fetch_evolution(session: requests.Session) -> pd.DataFrame:
    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    for d in days():
        day_frames: list[pd.DataFrame] = []
        date_text = d.isoformat()
        for sp in range(1, 49):
            url = f"{EVOLUTION_URL}?{urlencode({'settlementDate': date_text, 'settlementPeriod': sp})}"
            status, payload = fetch_json(session, url)
            rows = payload_data(payload)
            frame = flatten_rows(rows, "evolution", url)
            if not frame.empty:
                frame["httpStatus"] = status
                day_frames.append(frame)

        if day_frames:
            day = pd.concat(day_frames, ignore_index=True)
        else:
            day = pd.DataFrame()
        day.to_csv(EVOLUTION_DIR / f"{date_text}.csv", index=False)
        all_frames.append(day)
        print(f"[evolution] {date_text} rows={len(day)}")

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def direct_probe_urls() -> list[tuple[str, str]]:
    start = START.isoformat()
    end = END.isoformat()
    probes = [
        ("no_params", DIRECT_URL),
        ("boundary_national", f"{DIRECT_URL}?{urlencode({'boundary': 'national'})}"),
        ("from_to", f"{DIRECT_URL}?{urlencode({'from': start, 'to': end})}"),
        (
            "from_to_sp",
            f"{DIRECT_URL}?{urlencode({'from': start, 'to': end, 'settlementPeriodFrom': 1, 'settlementPeriodTo': 48})}",
        ),
        ("settlement_date_first_day", f"{DIRECT_URL}?{urlencode({'settlementDate': start})}"),
    ]
    for d in days():
        probes.append((f"from_to_one_day_{d.isoformat()}", f"{DIRECT_URL}?{urlencode({'from': d.isoformat(), 'to': d.isoformat()})}"))
    return probes


def fetch_direct(session: requests.Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    DIRECT_DIR.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict] = []
    frames: list[pd.DataFrame] = []

    for name, url in direct_probe_urls():
        out_json = DIRECT_DIR / f"{name}.json"
        out_csv = DIRECT_DIR / f"{name}.csv"
        try:
            status, payload = fetch_json(session, url)
            rows = payload_data(payload)
            out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            frame = flatten_rows(rows, name, url)
            frame.to_csv(out_csv, index=False)
            if not frame.empty:
                frame["httpStatus"] = status
                frames.append(frame)
            cols = ",".join(frame.columns)
            status_rows.append({"probe": name, "url": url, "status": status, "rows": len(frame), "columns": cols})
            print(f"[direct] {name} status={status} rows={len(frame)}")
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else ""
            text = exc.response.text[:500] if exc.response is not None else str(exc)
            status_rows.append({"probe": name, "url": url, "status": status, "rows": 0, "columns": "", "error": text})
            print(f"[direct] {name} status={status} error")

    direct = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return direct, pd.DataFrame(status_rows)


def total_load_probe_urls() -> list[tuple[str, str]]:
    start = START.isoformat()
    end = END.isoformat()
    probes = [
        ("from_to_week", f"{TOTAL_LOAD_URL}?{urlencode({'from': start, 'to': end})}"),
        (
            "from_to_week_sp",
            f"{TOTAL_LOAD_URL}?{urlencode({'from': start, 'to': end, 'settlementPeriodFrom': 1, 'settlementPeriodTo': 48})}",
        ),
    ]
    for d in days():
        probes.append(
            (
                f"from_to_one_day_{d.isoformat()}",
                f"{TOTAL_LOAD_URL}?{urlencode({'from': d.isoformat(), 'to': d.isoformat(), 'settlementPeriodFrom': 1, 'settlementPeriodTo': 48})}",
            )
        )
    return probes


def fetch_total_day_ahead_load(session: requests.Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    TOTAL_LOAD_DIR.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict] = []
    frames: list[pd.DataFrame] = []

    for name, url in total_load_probe_urls():
        out_json = TOTAL_LOAD_DIR / f"{name}.json"
        out_csv = TOTAL_LOAD_DIR / f"{name}.csv"
        try:
            status, payload = fetch_json(session, url)
            rows = payload_data(payload)
            out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            frame = flatten_rows(rows, name, url)
            frame.to_csv(out_csv, index=False)
            if not frame.empty:
                frame["httpStatus"] = status
                frames.append(frame)
            cols = ",".join(frame.columns)
            status_rows.append({"probe": name, "url": url, "status": status, "rows": len(frame), "columns": cols})
            print(f"[total-load] {name} status={status} rows={len(frame)}")
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else ""
            text = exc.response.text[:500] if exc.response is not None else str(exc)
            status_rows.append({"probe": name, "url": url, "status": status, "rows": 0, "columns": "", "error": text})
            print(f"[total-load] {name} status={status} error")

    total = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return total, pd.DataFrame(status_rows)


def select_latest_day_ahead_from_evolution(evolution: pd.DataFrame) -> pd.DataFrame:
    if evolution.empty:
        return pd.DataFrame()
    df = evolution.copy()
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True, errors="coerce")
    df["nationalDemand"] = pd.to_numeric(df["nationalDemand"], errors="coerce")
    df["transmissionSystemDemand"] = pd.to_numeric(df["transmissionSystemDemand"], errors="coerce")
    df["dayAheadDate"] = pd.to_datetime(df["settlementDate"], errors="coerce").dt.tz_localize(None) - pd.Timedelta(days=1)
    df["publishDate"] = df["publishTime"].dt.floor("D").dt.tz_localize(None)
    candidates = df[(df["publishDate"] == df["dayAheadDate"]) & (df["nationalDemand"].fillna(0) != 0)].copy()
    selected = (
        candidates.sort_values(["settlementDate", "settlementPeriod", "publishTime"])
        .drop_duplicates(["settlementDate", "settlementPeriod"], keep="last")
        .rename(
            columns={
                "nationalDemand": "evolution_latest_dminus1_nationalDemand",
                "transmissionSystemDemand": "evolution_latest_dminus1_transmissionSystemDemand",
                "publishTime": "evolution_latest_dminus1_publishTime",
            }
        )
    )
    return selected[
        [
            "settlementDate",
            "settlementPeriod",
            "startTime",
            "evolution_latest_dminus1_nationalDemand",
            "evolution_latest_dminus1_transmissionSystemDemand",
            "evolution_latest_dminus1_publishTime",
        ]
    ]


def summarize_evolution_revisions(evolution: pd.DataFrame) -> pd.DataFrame:
    if evolution.empty:
        return pd.DataFrame()
    df = evolution.copy()
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True, errors="coerce")
    return (
        df.groupby(["settlementDate", "settlementPeriod"], as_index=False)
        .agg(
            revisionCount=("publishTime", "count"),
            firstPublishTime=("publishTime", "min"),
            lastPublishTime=("publishTime", "max"),
            firstNationalDemand=("nationalDemand", "first"),
            lastNationalDemand=("nationalDemand", "last"),
        )
        .sort_values(["settlementDate", "settlementPeriod"])
    )


def compare_direct_to_evolution(direct: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if direct.empty or selected.empty:
        return pd.DataFrame()

    df = direct.copy()
    if "settlementDate" not in df.columns or "settlementPeriod" not in df.columns:
        return pd.DataFrame()
    df = df[df["settlementDate"].astype(str).between(START.isoformat(), END.isoformat())].copy()
    if df.empty:
        return pd.DataFrame()

    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    numeric_cols = [c for c in ["nationalDemand", "transmissionSystemDemand"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    direct_one = df.sort_values(["settlementDate", "settlementPeriod"]).drop_duplicates(["settlementDate", "settlementPeriod"])
    compare = selected.merge(direct_one, on=["settlementDate", "settlementPeriod"], how="outer", suffixes=("_evolution", "_direct"))
    if "nationalDemand" in compare.columns:
        compare["nationalDemand_diff_direct_minus_evolution"] = (
            compare["nationalDemand"] - compare["evolution_latest_dminus1_nationalDemand"]
        )
    if "transmissionSystemDemand" in compare.columns:
        compare["transmissionDemand_diff_direct_minus_evolution"] = (
            compare["transmissionSystemDemand"] - compare["evolution_latest_dminus1_transmissionSystemDemand"]
        )
    return compare


def compare_total_load_to_evolution(total: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if total.empty or selected.empty:
        return pd.DataFrame()
    if "settlementDate" not in total.columns or "settlementPeriod" not in total.columns:
        return pd.DataFrame()

    df = total.copy()
    df = df[df["settlementDate"].astype(str).between(START.isoformat(), END.isoformat())].copy()
    if df.empty:
        return pd.DataFrame()
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    if "publishTime" in df.columns:
        df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True, errors="coerce")

    # DATL may return duplicate rows because we probe both week and day query shapes.
    # Keep one observed value per SP after sorting by source/probe and publish time.
    sort_cols = ["settlementDate", "settlementPeriod"]
    if "publishTime" in df.columns:
        sort_cols.append("publishTime")
    total_one = df.sort_values(sort_cols).drop_duplicates(["settlementDate", "settlementPeriod"], keep="last")
    keep_cols = [
        c
        for c in [
            "settlementDate",
            "settlementPeriod",
            "startTime",
            "publishTime",
            "quantity",
            "dataset",
            "source",
            "requestUrl",
        ]
        if c in total_one.columns
    ]
    total_one = total_one[keep_cols].rename(
        columns={
            "startTime": "totalLoad_startTime",
            "publishTime": "totalLoad_publishTime",
            "quantity": "totalLoad_quantity",
            "dataset": "totalLoad_dataset",
            "source": "totalLoad_probe",
            "requestUrl": "totalLoad_requestUrl",
        }
    )
    compare = selected.merge(total_one, on=["settlementDate", "settlementPeriod"], how="outer")
    if "totalLoad_quantity" in compare.columns:
        compare["totalLoad_minus_evolution_nationalDemand"] = (
            compare["totalLoad_quantity"] - compare["evolution_latest_dminus1_nationalDemand"]
        )
        compare["totalLoad_minus_evolution_transmissionDemand"] = (
            compare["totalLoad_quantity"] - compare["evolution_latest_dminus1_transmissionSystemDemand"]
        )
    return compare


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    evolution = fetch_evolution(session)
    direct, direct_status = fetch_direct(session)
    total_load, total_load_status = fetch_total_day_ahead_load(session)
    selected = select_latest_day_ahead_from_evolution(evolution)
    revisions = summarize_evolution_revisions(evolution)
    comparison = compare_direct_to_evolution(direct, selected)
    total_load_comparison = compare_total_load_to_evolution(total_load, selected)

    evolution.to_csv(OUT_DIR / "evolution_first_week_jan2023_all_revisions.csv", index=False)
    selected.to_csv(OUT_DIR / "evolution_first_week_jan2023_latest_dminus1.csv", index=False)
    revisions.to_csv(OUT_DIR / "evolution_first_week_jan2023_revision_summary.csv", index=False)
    direct_status.to_csv(OUT_DIR / "direct_day_ahead_probe_status.csv", index=False)
    direct.to_csv(OUT_DIR / "direct_day_ahead_probe_rows.csv", index=False)
    comparison.to_csv(OUT_DIR / "direct_vs_evolution_first_week_jan2023_comparison.csv", index=False)
    total_load_status.to_csv(OUT_DIR / "total_day_ahead_load_probe_status.csv", index=False)
    total_load.to_csv(OUT_DIR / "total_day_ahead_load_probe_rows.csv", index=False)
    total_load_comparison.to_csv(OUT_DIR / "total_load_vs_evolution_first_week_jan2023_comparison.csv", index=False)

    summary_rows = [
        {"metric": "evolution_raw_rows", "value": len(evolution)},
        {"metric": "evolution_latest_dminus1_rows", "value": len(selected)},
        {"metric": "direct_probe_rows_in_any_response", "value": len(direct)},
        {"metric": "direct_vs_evolution_comparison_rows", "value": len(comparison)},
        {"metric": "total_day_ahead_load_probe_rows_in_any_response", "value": len(total_load)},
        {"metric": "total_load_vs_evolution_comparison_rows", "value": len(total_load_comparison)},
        {
            "metric": "direct_probe_successes",
            "value": int((pd.to_numeric(direct_status["status"], errors="coerce") == 200).sum()) if not direct_status.empty else 0,
        },
        {
            "metric": "total_day_ahead_load_probe_successes",
            "value": int((pd.to_numeric(total_load_status["status"], errors="coerce") == 200).sum()) if not total_load_status.empty else 0,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(REPORT_DIR / "day_ahead_demand_first_week_jan2023_probe_summary.csv", index=False)
    direct_status.to_csv(REPORT_DIR / "day_ahead_demand_first_week_jan2023_direct_endpoint_status.csv", index=False)
    total_load_status.to_csv(REPORT_DIR / "day_ahead_total_load_first_week_jan2023_endpoint_status.csv", index=False)

    print("Probe complete")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
