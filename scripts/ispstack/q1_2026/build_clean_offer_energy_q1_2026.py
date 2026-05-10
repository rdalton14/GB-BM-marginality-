from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_DIR = Path("data/raw/ispstack/q1_2026")
OUTPUT_DIR = Path("data/processed/clean_offer_energy_q1_2026")

CLEAN_STACK_PATH = OUTPUT_DIR / "clean_offer_energy_stack_q1_2026.parquet"
MARGINAL_PATH = OUTPUT_DIR / "marginal_offer_energy_q1_2026.parquet"
DAILY_SUMMARY_PATH = OUTPUT_DIR / "clean_offer_daily_summary_q1_2026.parquet"
SP_SUMMARY_PATH = OUTPUT_DIR / "clean_offer_sp_summary_q1_2026.parquet"
BMU_SUMMARY_PATH = OUTPUT_DIR / "clean_offer_bmu_summary_q1_2026.parquet"
QUALITY_SUMMARY_PATH = OUTPUT_DIR / "clean_offer_quality_summary_q1_2026.csv"
MULTIHOT_PATH = OUTPUT_DIR / "marginal_offer_multihot_matrix_q1_2026.parquet"
MULTIHOT_CSV_PATH = OUTPUT_DIR / "marginal_offer_multihot_matrix_q1_2026.csv"

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-03-31")
NUMERIC_ONLY_RE = re.compile(r"^\d+(\.\d+)?$")


def is_numeric_only(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(NUMERIC_ONLY_RE.fullmatch(str(value).strip()))


def weighted_average(values: pd.Series, weights: pd.Series) -> float | pd.NA:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return pd.NA
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def join_as_strings(series: pd.Series) -> str:
    return ";".join("" if pd.isna(x) else str(x) for x in series.tolist())


def load_q1_files() -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    file_count = 0

    for path in sorted(INPUT_DIR.glob("*.parquet")):
        try:
            file_date = pd.Timestamp(path.stem)
        except ValueError:
            continue
        if not (START_DATE <= file_date <= END_DATE):
            continue
        frames.append(pd.read_parquet(path))
        file_count += 1

    if not frames:
        raise FileNotFoundError(f"No daily parquet files found in {INPUT_DIR}")

    return pd.concat(frames, ignore_index=True), file_count


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, file_count = load_q1_files()

    df["settlementDate"] = pd.to_datetime(df["settlementDate"], errors="coerce")
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce")
    df["direction"] = df["direction"].astype(str).str.upper()
    df["finalPrice"] = pd.to_numeric(df["finalPrice"], errors="coerce")
    df["originalPrice"] = pd.to_numeric(df["originalPrice"], errors="coerce")
    df["parAdjustedVolume"] = pd.to_numeric(df["parAdjustedVolume"], errors="coerce")
    df["sequenceNumber"] = pd.to_numeric(df["sequenceNumber"], errors="coerce")

    total_raw_rows = len(df)
    raw_offer_rows = int((df["direction"] == "OFFER").sum())

    clean_offer = df.loc[
        (df["direction"] == "OFFER")
        & (df["finalPrice"].notna())
        & (df["soFlag"] != True)
        & (df["cadlFlag"] != True)
        & (df["storProviderFlag"] != True)
    ].copy()

    clean_offer["abs_par_volume"] = clean_offer["parAdjustedVolume"].abs()
    clean_offer["price_changed"] = clean_offer["originalPrice"] != clean_offer["finalPrice"]
    clean_offer["is_numeric_id"] = clean_offer["id"].map(is_numeric_only)
    clean_offer["is_missing_id"] = clean_offer["id"].isna()
    clean_offer["raw_cost_proxy"] = clean_offer["originalPrice"] * clean_offer["parAdjustedVolume"]
    clean_offer["final_cost_proxy"] = clean_offer["finalPrice"] * clean_offer["parAdjustedVolume"]
    clean_offer["abs_final_cost_proxy"] = clean_offer["final_cost_proxy"].abs()

    clean_offer = clean_offer.sort_values(
        ["settlementDate", "settlementPeriod", "finalPrice", "sequenceNumber"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    max_price_by_sp = clean_offer.groupby(["settlementDate", "settlementPeriod"], dropna=False)["finalPrice"].transform("max")
    marginal_offer = (
        clean_offer.loc[clean_offer["finalPrice"] == max_price_by_sp]
        .copy()
        .sort_values(["settlementDate", "settlementPeriod", "finalPrice", "sequenceNumber"], ascending=[True, True, False, False])
        .reset_index(drop=True)
    )

    unique_dates = int(df["settlementDate"].dt.normalize().nunique())
    expected_sps = unique_dates * 48
    sp_covered = int(
        clean_offer.loc[
            clean_offer["settlementPeriod"].between(1, 48, inclusive="both"),
            ["settlementDate", "settlementPeriod"],
        ]
        .dropna()
        .drop_duplicates()
        .shape[0]
    )

    clean_row_count = len(clean_offer)
    marginal_row_count = len(marginal_offer)
    marginal_sp_count = int(marginal_offer[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0])

    clean_missing_id_share = float(clean_offer["is_missing_id"].mean()) if clean_row_count else 0.0
    clean_numeric_id_share = float(clean_offer["is_numeric_id"].mean()) if clean_row_count else 0.0
    clean_missing_final_price_share = float(clean_offer["finalPrice"].isna().mean()) if clean_row_count else 0.0
    clean_missing_par_share = float(clean_offer["parAdjustedVolume"].isna().mean()) if clean_row_count else 0.0
    clean_repriced_share = float((clean_offer["repricedIndicator"] == True).mean()) if clean_row_count else 0.0
    total_clean_abs_par_volume = float(clean_offer["abs_par_volume"].sum())
    mean_clean_final_price = float(clean_offer["finalPrice"].mean()) if clean_row_count else pd.NA
    vwap_clean_final_price = weighted_average(clean_offer["finalPrice"], clean_offer["abs_par_volume"])

    daily_raw = (
        df.assign(date=df["settlementDate"].dt.normalize())
        .groupby("date", dropna=False)
        .agg(
            raw_action_count=("settlementDate", "size"),
            raw_offer_count=("direction", lambda s: int((s == "OFFER").sum())),
        )
        .reset_index()
        .rename(columns={"date": "settlementDate"})
    )

    daily_clean = (
        clean_offer.assign(date=clean_offer["settlementDate"].dt.normalize())
        .groupby("date", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "clean_offer_count": int(len(g)),
                    "clean_offer_abs_par_volume": float(g["abs_par_volume"].sum()),
                    "clean_offer_mean_finalPrice": float(g["finalPrice"].mean()) if len(g) else pd.NA,
                    "clean_offer_vwap_finalPrice": weighted_average(g["finalPrice"], g["abs_par_volume"]),
                    "unique_clean_offer_bmus": int(g["id"].nunique(dropna=True)),
                    "numeric_id_share": float(g["is_numeric_id"].mean()) if len(g) else 0.0,
                    "repriced_share": float((g["repricedIndicator"] == True).mean()) if len(g) else 0.0,
                }
            )
        )
        .reset_index()
        .rename(columns={"date": "settlementDate"})
    )

    daily_marginal = (
        marginal_offer.assign(date=marginal_offer["settlementDate"].dt.normalize())
        .groupby("date", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "marginal_offer_count": int(len(g)),
                    "marginal_offer_sp_count": int(g[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]),
                    "marginal_offer_mean_finalPrice": float(g["finalPrice"].mean()) if len(g) else pd.NA,
                    "marginal_offer_vwap_finalPrice": weighted_average(g["finalPrice"], g["abs_par_volume"]),
                }
            )
        )
        .reset_index()
        .rename(columns={"date": "settlementDate"})
    )

    daily_summary = daily_raw.merge(daily_clean, on="settlementDate", how="left").merge(
        daily_marginal, on="settlementDate", how="left"
    )
    daily_summary["clean_offer_count"] = daily_summary["clean_offer_count"].fillna(0).astype(int)
    daily_summary["marginal_offer_count"] = daily_summary["marginal_offer_count"].fillna(0).astype(int)
    daily_summary["marginal_offer_sp_count"] = daily_summary["marginal_offer_sp_count"].fillna(0).astype(int)
    daily_summary["clean_offer_coverage_share"] = daily_summary["clean_offer_count"] / daily_summary["raw_offer_count"].replace(0, pd.NA)
    daily_summary = daily_summary.sort_values("settlementDate").reset_index(drop=True)

    sp_clean = (
        clean_offer.groupby(["settlementDate", "settlementPeriod"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "clean_offer_count": int(len(g)),
                    "clean_offer_abs_par_volume": float(g["abs_par_volume"].sum()),
                    "clean_offer_mean_finalPrice": float(g["finalPrice"].mean()) if len(g) else pd.NA,
                    "clean_offer_vwap_finalPrice": weighted_average(g["finalPrice"], g["abs_par_volume"]),
                }
            )
        )
        .reset_index()
    )

    marginal_cols = (
        marginal_offer.groupby(["settlementDate", "settlementPeriod"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "marginal_offer_count": int(len(g)),
                    "marginal_bmu_ids": join_as_strings(g["id"]),
                    "marginal_acceptanceIds": join_as_strings(g["acceptanceId"]),
                    "marginal_bidOfferPairIds": join_as_strings(g["bidOfferPairId"]),
                    "marginal_sequenceNumbers": join_as_strings(g["sequenceNumber"]),
                    "marginal_originalPrices": join_as_strings(g["originalPrice"]),
                    "marginal_finalPrice": float(g["finalPrice"].iloc[0]) if len(g) else pd.NA,
                    "marginal_parAdjustedVolumes": join_as_strings(g["parAdjustedVolume"]),
                    "marginal_abs_par_volume_sum": float(g["abs_par_volume"].sum()),
                    "marginal_any_repricedIndicator": bool((g["repricedIndicator"] == True).any()),
                    "marginal_any_price_changed": bool(g["price_changed"].any()),
                }
            )
        )
        .reset_index()
    )

    sp_summary = (
        sp_clean.merge(marginal_cols, on=["settlementDate", "settlementPeriod"], how="left")
        .sort_values(["settlementDate", "settlementPeriod"])
        .reset_index(drop=True)
    )

    marginal_counts = (
        marginal_offer.groupby("id", dropna=False)
        .size()
        .rename("marginal_count")
        .reset_index()
    )

    bmu_summary = (
        clean_offer.groupby("id", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "clean_offer_action_count": int(len(g)),
                    "number_of_days_present": int(g["settlementDate"].dt.normalize().nunique()),
                    "number_of_sps_present": int(g[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]),
                    "total_abs_par_volume": float(g["abs_par_volume"].sum()),
                    "mean_finalPrice": float(g["finalPrice"].mean()) if len(g) else pd.NA,
                    "vwap_finalPrice": weighted_average(g["finalPrice"], g["abs_par_volume"]),
                    "min_finalPrice": float(g["finalPrice"].min()) if len(g) else pd.NA,
                    "max_finalPrice": float(g["finalPrice"].max()) if len(g) else pd.NA,
                    "repriced_share": float((g["repricedIndicator"] == True).mean()) if len(g) else 0.0,
                }
            )
        )
        .reset_index()
    ).merge(marginal_counts, on="id", how="left")
    bmu_summary["marginal_count"] = bmu_summary["marginal_count"].fillna(0).astype(int)
    bmu_summary["marginal_share_of_sps"] = bmu_summary["marginal_count"] / marginal_sp_count if marginal_sp_count else 0.0

    multihot_source = marginal_offer[["settlementDate", "settlementPeriod", "id"]].dropna(subset=["id"]).drop_duplicates().copy()
    multihot_source["value"] = 1
    multihot_matrix = (
        multihot_source.pivot(index=["settlementDate", "settlementPeriod"], columns="id", values="value")
        .fillna(0)
        .astype("int8")
        .reset_index()
        .sort_values(["settlementDate", "settlementPeriod"])
        .reset_index(drop=True)
    )

    quality_summary = pd.DataFrame(
        [
            {"metric": "files_loaded", "value": file_count},
            {"metric": "total_raw_rows", "value": total_raw_rows},
            {"metric": "raw_offer_rows", "value": raw_offer_rows},
            {"metric": "clean_offer_side_energy_rows", "value": clean_row_count},
            {"metric": "share_of_raw_offer_rows_retained", "value": (clean_row_count / raw_offer_rows) if raw_offer_rows else 0.0},
            {"metric": "unique_settlement_dates", "value": unique_dates},
            {"metric": "expected_settlement_periods", "value": expected_sps},
            {"metric": "settlement_periods_covered_by_clean_stack", "value": sp_covered},
            {"metric": "share_of_settlement_periods_covered", "value": (sp_covered / expected_sps) if expected_sps else 0.0},
            {"metric": "number_of_joint_marginal_offer_rows", "value": marginal_row_count},
            {"metric": "number_of_settlement_periods_with_joint_marginal_offers", "value": marginal_sp_count},
            {"metric": "share_of_clean_rows_with_missing_id", "value": clean_missing_id_share},
            {"metric": "share_of_clean_rows_with_numeric_only_id", "value": clean_numeric_id_share},
            {"metric": "share_of_clean_rows_with_missing_finalPrice", "value": clean_missing_final_price_share},
            {"metric": "share_of_clean_rows_with_missing_parAdjustedVolume", "value": clean_missing_par_share},
            {"metric": "share_of_clean_rows_where_repricedIndicator_true", "value": clean_repriced_share},
            {"metric": "total_clean_offer_abs_par_volume", "value": total_clean_abs_par_volume},
            {"metric": "average_clean_offer_finalPrice", "value": mean_clean_final_price},
            {"metric": "vwap_clean_offer_finalPrice", "value": vwap_clean_final_price},
        ]
    )

    print(f"Files loaded: {file_count:,}")
    print(f"Total raw rows: {total_raw_rows:,}")
    print(f"Raw offer rows: {raw_offer_rows:,}")
    print(f"Clean offer-side energy rows: {clean_row_count:,}")
    print(f"Share of raw offer rows retained: {(clean_row_count / raw_offer_rows) if raw_offer_rows else 0.0:.4f}")
    print(f"Unique settlement dates: {unique_dates:,}")
    print(f"Expected settlement periods: {expected_sps:,}")
    print(f"Settlement periods covered by clean offer stack: {sp_covered:,}")
    print(f"Share of settlement periods covered: {(sp_covered / expected_sps) if expected_sps else 0.0:.4f}")
    print(f"Number of joint marginal offer rows: {marginal_row_count:,}")
    print(f"Settlement periods with joint marginal offers: {marginal_sp_count:,}")
    print(f"Share of clean rows with missing id: {clean_missing_id_share:.4f}")
    print(f"Share of clean rows with numeric-only id: {clean_numeric_id_share:.4f}")
    print(f"Share of clean rows with missing finalPrice: {clean_missing_final_price_share:.4f}")
    print(f"Share of clean rows with missing parAdjustedVolume: {clean_missing_par_share:.4f}")
    print(f"Share of clean rows where repricedIndicator == True: {clean_repriced_share:.4f}")
    print(f"Total clean offer abs par volume: {total_clean_abs_par_volume:,.4f}")
    print(f"Average clean offer finalPrice: {mean_clean_final_price}")
    print(f"VWAP clean offer finalPrice: {vwap_clean_final_price}")

    print("\nTop 25 BMUs by clean offer abs par volume")
    print(bmu_summary.sort_values("total_abs_par_volume", ascending=False).head(25).to_string(index=False))

    print("\nTop 25 BMUs by clean offer action count")
    print(bmu_summary.sort_values("clean_offer_action_count", ascending=False).head(25).to_string(index=False))

    print("\nTop 25 BMUs by marginal_count")
    print(bmu_summary.sort_values("marginal_count", ascending=False).head(25).to_string(index=False))

    print("\nTop 25 BMUs by average marginal finalPrice (min 10 marginal appearances)")
    avg_marginal_price = (
        marginal_offer.groupby("id", dropna=False)
        .agg(marginal_count=("id", "size"), average_marginal_finalPrice=("finalPrice", "mean"))
        .reset_index()
    )
    print(
        avg_marginal_price.loc[avg_marginal_price["marginal_count"] >= 10]
        .sort_values("average_marginal_finalPrice", ascending=False)
        .head(25)
        .to_string(index=False)
    )

    clean_offer.to_parquet(CLEAN_STACK_PATH, index=False)
    marginal_offer.to_parquet(MARGINAL_PATH, index=False)
    daily_summary.to_parquet(DAILY_SUMMARY_PATH, index=False)
    sp_summary.to_parquet(SP_SUMMARY_PATH, index=False)
    bmu_summary.to_parquet(BMU_SUMMARY_PATH, index=False)
    multihot_matrix.to_parquet(MULTIHOT_PATH, index=False)
    multihot_matrix.to_csv(MULTIHOT_CSV_PATH, index=False)
    quality_summary.to_csv(QUALITY_SUMMARY_PATH, index=False)

    print(f"\nSaved: {CLEAN_STACK_PATH}")
    print(f"Saved: {MARGINAL_PATH}")
    print(f"Saved: {DAILY_SUMMARY_PATH}")
    print(f"Saved: {SP_SUMMARY_PATH}")
    print(f"Saved: {BMU_SUMMARY_PATH}")
    print(f"Saved: {MULTIHOT_PATH}")
    print(f"Saved: {MULTIHOT_CSV_PATH}")
    print(f"Saved: {QUALITY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
