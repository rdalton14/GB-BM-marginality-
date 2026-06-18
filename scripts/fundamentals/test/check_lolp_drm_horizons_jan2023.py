from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "lolpdrm"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "forecast_base_joins_jan2023"

START = "2023-01-01"
END = "2023-01-31"
EXPECTED_HORIZONS = [1, 2, 4, 8, 12]


def files_for_window() -> list[Path]:
    files: list[Path] = []
    date = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while date <= end:
        path = RAW_DIR / f"{date.strftime('%Y-%m-%d')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Empty raw file: {path}")
        files.append(path)
        date += pd.Timedelta(days=1)
    return files


def read_raw() -> pd.DataFrame:
    raw = pd.concat((pd.read_csv(path) for path in files_for_window()), ignore_index=True)
    raw["settlementPeriod"] = pd.to_numeric(raw["settlementPeriod"], errors="raise").astype("int16")
    raw["forecastHorizon"] = pd.to_numeric(raw["forecastHorizon"], errors="raise").astype("int16")
    raw["targetStartTime"] = pd.to_datetime(raw["startTime"], utc=True, errors="coerce")
    raw["publishTime"] = pd.to_datetime(raw["publishTime"], utc=True, errors="coerce")
    raw["publishingPeriodCommencingTime"] = pd.to_datetime(
        raw["publishingPeriodCommencingTime"], utc=True, errors="coerce"
    )
    raw["lossOfLoadProbability"] = pd.to_numeric(raw["lossOfLoadProbability"], errors="coerce")
    raw["deratedMargin"] = pd.to_numeric(raw["deratedMargin"], errors="coerce")
    raw["publishLagMinutes"] = (
        raw["publishTime"] - raw["publishingPeriodCommencingTime"]
    ).dt.total_seconds() / 60.0
    raw["leadHoursToDelivery"] = (raw["targetStartTime"] - raw["publishTime"]).dt.total_seconds() / 3600.0
    raw["nominalLeadHoursToDelivery"] = (
        raw["targetStartTime"] - raw["publishingPeriodCommencingTime"]
    ).dt.total_seconds() / 3600.0
    return raw


def build_horizon_summary(raw: pd.DataFrame) -> pd.DataFrame:
    expected_sp_rows = int(pd.date_range(START, END, freq="D").size * 48)
    rows = []
    for horizon in sorted(raw["forecastHorizon"].dropna().unique()):
        subset = raw[raw["forecastHorizon"].eq(horizon)]
        key_dupes = int(
            subset.duplicated(["settlementDate", "settlementPeriod", "forecastHorizon"], keep=False).sum()
        )
        rows.append(
            {
                "forecastHorizon": int(horizon),
                "rows": len(subset),
                "expectedSpRows": expected_sp_rows,
                "missingSpRows": expected_sp_rows - len(subset),
                "duplicateKeyRows": key_dupes,
                "missingLolpRows": int(subset["lossOfLoadProbability"].isna().sum()),
                "missingDrmRows": int(subset["deratedMargin"].isna().sum()),
                "negativeLolpRows": int((subset["lossOfLoadProbability"] < 0).sum()),
                "lolpAboveOneRows": int((subset["lossOfLoadProbability"] > 1).sum()),
                "negativeDrmRows": int((subset["deratedMargin"] < 0).sum()),
                "minPublishLagMinutes": float(subset["publishLagMinutes"].min()),
                "medianPublishLagMinutes": float(subset["publishLagMinutes"].median()),
                "maxPublishLagMinutes": float(subset["publishLagMinutes"].max()),
                "minLeadHoursToDelivery": float(subset["leadHoursToDelivery"].min()),
                "medianLeadHoursToDelivery": float(subset["leadHoursToDelivery"].median()),
                "maxLeadHoursToDelivery": float(subset["leadHoursToDelivery"].max()),
                "nominalLeadHoursToDelivery": float(subset["nominalLeadHoursToDelivery"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_sp_horizon_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["settlementDate", "settlementPeriod", "targetStartTime"]
    matrix = (
        raw.assign(hasHorizon=1)
        .pivot_table(index=keys, columns="forecastHorizon", values="hasHorizon", aggfunc="max", fill_value=0)
        .reset_index()
    )
    for horizon in EXPECTED_HORIZONS:
        if horizon not in matrix.columns:
            matrix[horizon] = 0
    matrix = matrix.rename(columns={h: f"has_lolp_drm_{h}h" for h in EXPECTED_HORIZONS})
    expected_cols = [f"has_lolp_drm_{h}h" for h in EXPECTED_HORIZONS]
    matrix["availableHorizonCount"] = matrix[expected_cols].sum(axis=1)
    matrix["hasAllExpectedHorizons"] = matrix["availableHorizonCount"].eq(len(EXPECTED_HORIZONS))
    return matrix.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_raw()
    summary = build_horizon_summary(raw)
    matrix = build_sp_horizon_matrix(raw)

    raw.to_csv(OUT_DIR / "lolp_drm_raw_jan2023.csv", index=False)
    summary.to_csv(OUT_DIR / "lolp_drm_horizon_summary_jan2023.csv", index=False)
    matrix.to_csv(OUT_DIR / "lolp_drm_sp_horizon_matrix_jan2023.csv", index=False)

    print(f"Observed horizons: {sorted(raw['forecastHorizon'].unique().tolist())}")
    print(summary.to_string(index=False))
    print()
    print(
        matrix["hasAllExpectedHorizons"]
        .value_counts(dropna=False)
        .rename_axis("hasAllExpectedHorizons")
        .reset_index(name="spRows")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
