from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "fundamentals"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "forecast_aligned_baseline_jan2023"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_aligned_baseline_jan2023"

START = "2023-01-01"
END = "2023-01-07"
HORIZONS = [1, 2, 4, 8, 12]


def files_for_week(raw_name: str) -> list[Path]:
    start = pd.Timestamp(START)
    end = pd.Timestamp(END)
    files: list[Path] = []
    d = start
    while d <= end:
        path = RAW_ROOT / raw_name / f"{d.strftime('%Y-%m-%d')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Empty raw file: {path}")
        files.append(path)
        d += pd.Timedelta(days=1)
    return files


def read_week(raw_name: str, usecols: list[str]) -> pd.DataFrame:
    return pd.concat((pd.read_csv(path, usecols=usecols) for path in files_for_week(raw_name)), ignore_index=True)


def build_target_spine() -> pd.DataFrame:
    raw = read_week(
        "system_price_niv",
        ["settlementDate", "settlementPeriod", "startTime", "createdDateTime", "systemSellPrice", "systemBuyPrice"],
    )
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["targetStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["systemSellPrice"] = pd.to_numeric(raw["systemSellPrice"], errors="coerce")
    raw["systemBuyPrice"] = pd.to_numeric(raw["systemBuyPrice"], errors="coerce")
    raw["systemPrice"] = np.where(
        np.isclose(raw["systemSellPrice"], raw["systemBuyPrice"], equal_nan=False),
        raw["systemBuyPrice"],
        np.nan,
    )
    out = raw[["settlementDate", "settlementPeriod", "targetStartTime", "systemPrice"]].sort_values(
        ["settlementDate", "settlementPeriod"]
    )
    out["hour"] = ((out["settlementPeriod"].astype(int) - 1) // 2).astype("int8")
    dt = pd.to_datetime(out["settlementDate"])
    out["dayOfWeek"] = dt.dt.dayofweek.astype("int8")
    out["isWeekend"] = out["dayOfWeek"].isin([5, 6]).astype("int8")
    radians = 2 * np.pi * ((out["settlementPeriod"].astype(int) - 1) / 48.0)
    out["settlementPeriod_sin"] = np.sin(radians)
    out["settlementPeriod_cos"] = np.cos(radians)
    return out


def demand_raw(spine: pd.DataFrame) -> pd.DataFrame:
    raw = read_week(
        "day_ahead_demand_forecast_evolution",
        ["settlementDate", "settlementPeriod", "startTime", "publishTime", "nationalDemand", "transmissionSystemDemand"],
    )
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["nationalDemand"] = pd.to_numeric(raw["nationalDemand"], errors="coerce")
    raw["transmissionSystemDemand"] = pd.to_numeric(raw["transmissionSystemDemand"], errors="coerce")
    raw = raw[raw["nationalDemand"].fillna(0) != 0]
    return raw.merge(spine[["settlementDate", "settlementPeriod", "targetStartTime"]], on=["settlementDate", "settlementPeriod"])


def wind_solar_raw(spine: pd.DataFrame) -> pd.DataFrame:
    raw = read_week(
        "day_ahead_generation_wind_solar",
        ["settlementDate", "settlementPeriod", "startTime", "publishTime", "processType", "psrType", "quantity"],
    )
    raw["settlementPeriod"] = raw["settlementPeriod"].astype("int16")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")
    raw = raw[raw["psrType"].isin(["Wind Onshore", "Wind Offshore", "Solar"])]
    return raw.merge(spine[["settlementDate", "settlementPeriod", "targetStartTime"]], on=["settlementDate", "settlementPeriod"])


def select_latest_demand(raw: pd.DataFrame, spine: pd.DataFrame, horizon: int) -> pd.DataFrame:
    candidates = raw.copy()
    candidates["asOfTime"] = candidates["targetStartTime"] - pd.to_timedelta(horizon, unit="h")
    candidates = candidates[candidates["publishTime"] <= candidates["asOfTime"]]
    selected = (
        candidates.sort_values(["settlementDate", "settlementPeriod", "publishTime"])
        .drop_duplicates(["settlementDate", "settlementPeriod"], keep="last")
        .rename(
            columns={
                "nationalDemand": "demandForecast",
                "transmissionSystemDemand": "transmissionDemandForecast",
                "publishTime": "demandForecast_publishTime",
            }
        )
    )
    return spine.merge(
        selected[
            [
                "settlementDate",
                "settlementPeriod",
                "demandForecast",
                "transmissionDemandForecast",
                "demandForecast_publishTime",
            ]
        ],
        on=["settlementDate", "settlementPeriod"],
        how="left",
        validate="one_to_one",
    )


def select_latest_wind_solar(raw: pd.DataFrame, spine: pd.DataFrame, horizon: int) -> pd.DataFrame:
    candidates = raw.copy()
    candidates["asOfTime"] = candidates["targetStartTime"] - pd.to_timedelta(horizon, unit="h")
    candidates = candidates[candidates["publishTime"] <= candidates["asOfTime"]]
    selected = (
        candidates.sort_values(["settlementDate", "settlementPeriod", "psrType", "publishTime"])
        .drop_duplicates(["settlementDate", "settlementPeriod", "psrType"], keep="last")
    )

    values = selected.pivot_table(
        index=["settlementDate", "settlementPeriod"],
        columns="psrType",
        values="quantity",
        aggfunc="first",
    ).reset_index()
    values = values.rename(
        columns={
            "Wind Onshore": "windOnshoreForecast",
            "Wind Offshore": "windOffshoreForecast",
            "Solar": "solarForecast",
        }
    )
    for col in ["windOnshoreForecast", "windOffshoreForecast", "solarForecast"]:
        if col not in values.columns:
            values[col] = np.nan
    values["windForecast"] = values["windOnshoreForecast"] + values["windOffshoreForecast"]

    publish = selected.pivot_table(
        index=["settlementDate", "settlementPeriod"],
        columns="psrType",
        values="publishTime",
        aggfunc="first",
    ).reset_index()
    publish = publish.rename(
        columns={
            "Wind Onshore": "windOnshoreForecast_publishTime",
            "Wind Offshore": "windOffshoreForecast_publishTime",
            "Solar": "solarForecast_publishTime",
        }
    )
    process = selected.pivot_table(
        index=["settlementDate", "settlementPeriod"],
        columns="psrType",
        values="processType",
        aggfunc="first",
    ).reset_index()
    process = process.rename(
        columns={
            "Wind Onshore": "windOnshoreForecast_processType",
            "Wind Offshore": "windOffshoreForecast_processType",
            "Solar": "solarForecast_processType",
        }
    )

    out = spine.merge(values, on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
    out = out.merge(publish, on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
    out = out.merge(process, on=["settlementDate", "settlementPeriod"], how="left", validate="one_to_one")
    return out


def audit_variable(panel: pd.DataFrame, horizon: int, variable: str, value_cols: list[str], publish_cols: list[str]) -> dict[str, object]:
    missing = int(panel[value_cols].isna().any(axis=1).sum())
    leakage = 0
    min_lead = None
    median_lead = None
    max_lead = None
    lead_values: list[pd.Series] = []
    for col in publish_cols:
        if col in panel.columns:
            publish = pd.to_datetime(panel[col], utc=True, errors="coerce")
            leakage += int((publish.notna() & (publish > panel["asOfTime"])).sum())
            lead = (panel["asOfTime"] - publish).dt.total_seconds() / 3600.0
            lead_values.append(lead[publish.notna()])
    if lead_values:
        leads = pd.concat(lead_values, ignore_index=True)
        if not leads.empty:
            min_lead = float(leads.min())
            median_lead = float(leads.median())
            max_lead = float(leads.max())
    return {
        "horizonHours": horizon,
        "variable": variable,
        "rows": len(panel),
        "missingRows": missing,
        "publishAfterAsOfRows": leakage,
        "minLeadHours": min_lead,
        "medianLeadHours": median_lead,
        "maxLeadHours": max_lead,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    spine = build_target_spine()
    d_raw = demand_raw(spine)
    ws_raw = wind_solar_raw(spine)
    audits: list[dict[str, object]] = []
    process_counts: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        base = spine.copy()
        base["forecastHorizonHours"] = horizon
        base["asOfTime"] = base["targetStartTime"] - pd.to_timedelta(horizon, unit="h")

        demand = select_latest_demand(d_raw, base, horizon)
        wind_solar = select_latest_wind_solar(ws_raw, base, horizon)
        panel = base.merge(
            demand[
                [
                    "settlementDate",
                    "settlementPeriod",
                    "demandForecast",
                    "transmissionDemandForecast",
                    "demandForecast_publishTime",
                ]
            ],
            on=["settlementDate", "settlementPeriod"],
            how="left",
            validate="one_to_one",
        )
        panel = panel.merge(
            wind_solar[
                [
                    "settlementDate",
                    "settlementPeriod",
                    "windForecast",
                    "windOnshoreForecast",
                    "windOffshoreForecast",
                    "solarForecast",
                    "windOnshoreForecast_publishTime",
                    "windOffshoreForecast_publishTime",
                    "solarForecast_publishTime",
                    "windOnshoreForecast_processType",
                    "windOffshoreForecast_processType",
                    "solarForecast_processType",
                ]
            ],
            on=["settlementDate", "settlementPeriod"],
            how="left",
            validate="one_to_one",
        )

        panel.to_csv(OUT_DIR / f"forecast_aligned_baseline_{horizon}h_first_week_jan2023.csv", index=False)
        panel.to_parquet(OUT_DIR / f"forecast_aligned_baseline_{horizon}h_first_week_jan2023.parquet", index=False)
        audits.append(audit_variable(panel, horizon, "demandForecast", ["demandForecast"], ["demandForecast_publishTime"]))
        audits.append(
            audit_variable(
                panel,
                horizon,
                "windForecast",
                ["windForecast"],
                ["windOnshoreForecast_publishTime", "windOffshoreForecast_publishTime"],
            )
        )
        audits.append(audit_variable(panel, horizon, "solarForecast", ["solarForecast"], ["solarForecast_publishTime"]))

        proc_cols = [
            "windOnshoreForecast_processType",
            "windOffshoreForecast_processType",
            "solarForecast_processType",
        ]
        proc = (
            panel[proc_cols]
            .melt(var_name="component", value_name="processType")
            .dropna()
            .groupby(["component", "processType"], as_index=False)
            .size()
        )
        proc["horizonHours"] = horizon
        process_counts.append(proc)

    pd.DataFrame(audits).to_csv(DIAG_DIR / "forecast_alignment_coverage_first_week_jan2023.csv", index=False)
    if process_counts:
        pd.concat(process_counts, ignore_index=True).to_csv(
            DIAG_DIR / "forecast_alignment_wind_solar_process_types_first_week_jan2023.csv", index=False
        )
    print(f"Wrote first-week Jan 2023 forecast-aligned panels to {OUT_DIR}")
    print(pd.DataFrame(audits).to_string(index=False))


if __name__ == "__main__":
    main()
