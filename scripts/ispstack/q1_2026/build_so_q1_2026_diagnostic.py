from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_DIR = Path("data/raw/ispstack/q1_2026")
OUTPUT_DIR = Path("data/processed/so_q1_2026")

SO_ACTIONS_PATH = OUTPUT_DIR / "so_actions_q1_2026.parquet"
DAILY_SUMMARY_PATH = OUTPUT_DIR / "so_daily_summary_q1_2026.parquet"
SP_SUMMARY_PATH = OUTPUT_DIR / "so_sp_summary_q1_2026.parquet"
BMU_SUMMARY_PATH = OUTPUT_DIR / "so_bmu_summary_q1_2026.parquet"
QUALITY_SUMMARY_PATH = OUTPUT_DIR / "so_q1_quality_summary.csv"

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-03-31")

NUMERIC_ONLY_RE = re.compile(r"^\d+(\.\d+)?$")


def is_numeric_only(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(NUMERIC_ONLY_RE.fullmatch(str(value).strip()))


def load_q1_files() -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    file_count = 0

    for path in sorted(INPUT_DIR.glob("*.parquet")):
        try:
            date = pd.Timestamp(path.stem)
        except ValueError:
            continue
        if not (START_DATE <= date <= END_DATE):
            continue
        frames.append(pd.read_parquet(path))
        file_count += 1

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR} for the requested Q1 2026 window.")

    return pd.concat(frames, ignore_index=True), file_count


def weighted_average(price: pd.Series, weights: pd.Series) -> float | pd.NA:
    valid = price.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return pd.NA
    return float((price[valid] * weights[valid]).sum() / weights[valid].sum())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, file_count = load_q1_files()

    df["settlementDate"] = pd.to_datetime(df["settlementDate"], errors="coerce")
    df["date"] = df["settlementDate"].dt.normalize()
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce")
    df["direction"] = df["direction"].astype(str).str.upper()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["originalPrice"] = pd.to_numeric(df["originalPrice"], errors="coerce")

    df["abs_volume"] = df["volume"].abs()
    df["raw_cost_proxy"] = df["originalPrice"] * df["volume"]
    df["abs_raw_cost_proxy"] = df["raw_cost_proxy"].abs()
    df["is_numeric_id"] = df["id"].map(is_numeric_only)
    df["is_missing_id"] = df["id"].isna()

    so_actions = df.loc[df["soFlag"] == True].copy()

    total_raw_rows = len(df)
    total_so_rows = len(so_actions)
    unique_dates = int(df["date"].nunique())
    sp_covered = int(df.loc[df["settlementPeriod"].between(1, 48, inclusive="both"), ["date", "settlementPeriod"]].dropna().drop_duplicates().shape[0])
    expected_sp = unique_dates * 48

    so_missing_id_share = float(so_actions["is_missing_id"].mean()) if total_so_rows else 0.0
    so_numeric_id_share = float(so_actions["is_numeric_id"].mean()) if total_so_rows else 0.0
    so_missing_original_price_share = float(so_actions["originalPrice"].isna().mean()) if total_so_rows else 0.0
    so_missing_volume_share = float(so_actions["volume"].isna().mean()) if total_so_rows else 0.0
    so_zero_volume_share = float((so_actions["volume"] == 0).mean()) if total_so_rows else 0.0

    total_so_abs_volume = float(so_actions["abs_volume"].sum())
    total_so_abs_raw_cost_proxy = float(so_actions["abs_raw_cost_proxy"].sum())
    so_vwap_original_price = weighted_average(so_actions["originalPrice"], so_actions["abs_volume"])

    so_bid = so_actions.loc[so_actions["direction"] == "BID"].copy()
    so_offer = so_actions.loc[so_actions["direction"] == "OFFER"].copy()

    so_bid_abs_volume = float(so_bid["abs_volume"].sum())
    so_offer_abs_volume = float(so_offer["abs_volume"].sum())
    so_bid_abs_raw_cost_proxy = float(so_bid["abs_raw_cost_proxy"].sum())
    so_offer_abs_raw_cost_proxy = float(so_offer["abs_raw_cost_proxy"].sum())

    daily_summary = (
        df.groupby("date", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "total_action_count": int(len(g)),
                    "so_action_count": int((g["soFlag"] == True).sum()),
                    "so_action_share": float((g["soFlag"] == True).mean()) if len(g) else 0.0,
                    "so_bid_count": int(((g["soFlag"] == True) & (g["direction"] == "BID")).sum()),
                    "so_offer_count": int(((g["soFlag"] == True) & (g["direction"] == "OFFER")).sum()),
                    "so_abs_volume": float(g.loc[g["soFlag"] == True, "abs_volume"].sum()),
                    "so_bid_abs_volume": float(g.loc[(g["soFlag"] == True) & (g["direction"] == "BID"), "abs_volume"].sum()),
                    "so_offer_abs_volume": float(g.loc[(g["soFlag"] == True) & (g["direction"] == "OFFER"), "abs_volume"].sum()),
                    "so_abs_raw_cost_proxy": float(g.loc[g["soFlag"] == True, "abs_raw_cost_proxy"].sum()),
                    "so_vwap_original_price": weighted_average(
                        g.loc[g["soFlag"] == True, "originalPrice"],
                        g.loc[g["soFlag"] == True, "abs_volume"],
                    ),
                    "unique_so_bmus": int(g.loc[g["soFlag"] == True, "id"].nunique(dropna=True)),
                    "numeric_id_share": float(g.loc[g["soFlag"] == True, "is_numeric_id"].mean())
                    if (g["soFlag"] == True).any()
                    else 0.0,
                    "missing_id_share": float(g.loc[g["soFlag"] == True, "is_missing_id"].mean())
                    if (g["soFlag"] == True).any()
                    else 0.0,
                }
            )
        )
        .reset_index()
    )
    daily_summary["so_bid_volume_share"] = daily_summary["so_bid_abs_volume"] / daily_summary["so_abs_volume"].replace(0, pd.NA)
    daily_summary["so_offer_volume_share"] = daily_summary["so_offer_abs_volume"] / daily_summary["so_abs_volume"].replace(0, pd.NA)

    sp_summary = (
        df.groupby(["date", "settlementPeriod"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "total_action_count": int(len(g)),
                    "so_action_count": int((g["soFlag"] == True).sum()),
                    "so_action_share": float((g["soFlag"] == True).mean()) if len(g) else 0.0,
                    "so_bid_count": int(((g["soFlag"] == True) & (g["direction"] == "BID")).sum()),
                    "so_offer_count": int(((g["soFlag"] == True) & (g["direction"] == "OFFER")).sum()),
                    "so_abs_volume": float(g.loc[g["soFlag"] == True, "abs_volume"].sum()),
                    "so_bid_abs_volume": float(g.loc[(g["soFlag"] == True) & (g["direction"] == "BID"), "abs_volume"].sum()),
                    "so_offer_abs_volume": float(g.loc[(g["soFlag"] == True) & (g["direction"] == "OFFER"), "abs_volume"].sum()),
                    "so_abs_raw_cost_proxy": float(g.loc[g["soFlag"] == True, "abs_raw_cost_proxy"].sum()),
                    "so_vwap_original_price": weighted_average(
                        g.loc[g["soFlag"] == True, "originalPrice"],
                        g.loc[g["soFlag"] == True, "abs_volume"],
                    ),
                    "unique_so_bmus": int(g.loc[g["soFlag"] == True, "id"].nunique(dropna=True)),
                }
            )
        )
        .reset_index()
        .sort_values(["date", "settlementPeriod"])
        .reset_index(drop=True)
    )
    sp_summary["so_bid_volume_share"] = sp_summary["so_bid_abs_volume"] / sp_summary["so_abs_volume"].replace(0, pd.NA)

    so_actions["weighted_price_volume"] = so_actions["originalPrice"] * so_actions["abs_volume"]
    bmu_summary = (
        so_actions.groupby(["id", "direction"], dropna=False)
        .agg(
            action_count=("id", "size"),
            number_of_days_present=("date", "nunique"),
            number_of_sps_present=("settlementPeriod", "nunique"),
            abs_volume=("abs_volume", "sum"),
            abs_raw_cost_proxy=("abs_raw_cost_proxy", "sum"),
            weighted_price_volume=("weighted_price_volume", "sum"),
            mean_original_price=("originalPrice", "mean"),
            min_original_price=("originalPrice", "min"),
            max_original_price=("originalPrice", "max"),
        )
        .reset_index()
    )
    bmu_summary["vwap_original_price"] = bmu_summary["weighted_price_volume"] / bmu_summary["abs_volume"].replace(0, pd.NA)
    bmu_summary = bmu_summary.drop(columns=["weighted_price_volume"])

    quality_summary = pd.DataFrame(
        [
            {"metric": "files_loaded", "value": file_count},
            {"metric": "total_raw_rows", "value": total_raw_rows},
            {"metric": "total_so_rows", "value": total_so_rows},
            {"metric": "so_action_share", "value": (total_so_rows / total_raw_rows) if total_raw_rows else 0.0},
            {"metric": "unique_settlement_dates", "value": unique_dates},
            {"metric": "settlement_period_combinations_covered", "value": sp_covered},
            {"metric": "expected_sp_combinations_48_per_day", "value": expected_sp},
            {"metric": "sp_coverage_share", "value": (sp_covered / expected_sp) if expected_sp else 0.0},
            {"metric": "so_missing_id_share", "value": so_missing_id_share},
            {"metric": "so_numeric_id_share", "value": so_numeric_id_share},
            {"metric": "so_missing_originalPrice_share", "value": so_missing_original_price_share},
            {"metric": "so_missing_volume_share", "value": so_missing_volume_share},
            {"metric": "so_zero_volume_share", "value": so_zero_volume_share},
            {"metric": "total_so_abs_accepted_volume", "value": total_so_abs_volume},
            {"metric": "total_so_abs_raw_cost_proxy", "value": total_so_abs_raw_cost_proxy},
            {"metric": "so_vwap_original_price", "value": so_vwap_original_price},
            {"metric": "so_bid_abs_volume", "value": so_bid_abs_volume},
            {"metric": "so_offer_abs_volume", "value": so_offer_abs_volume},
            {"metric": "so_bid_abs_raw_cost_proxy", "value": so_bid_abs_raw_cost_proxy},
            {"metric": "so_offer_abs_raw_cost_proxy", "value": so_offer_abs_raw_cost_proxy},
        ]
    )

    print(f"Files loaded: {file_count:,}")
    print(f"Total raw rows: {total_raw_rows:,}")
    print(f"Total SO-flagged rows: {total_so_rows:,}")
    print(f"SO action share: {(total_so_rows / total_raw_rows) if total_raw_rows else 0.0:.4f}")
    print(f"Unique settlement dates: {unique_dates:,}")
    print(f"Settlement periods covered: {sp_covered:,} / {expected_sp:,}")
    print(f"Share of SO rows with missing id: {so_missing_id_share:.4f}")
    print(f"Share of SO rows with numeric-only id: {so_numeric_id_share:.4f}")
    print(f"Share of SO rows with missing originalPrice: {so_missing_original_price_share:.4f}")
    print(f"Share of SO rows with missing volume: {so_missing_volume_share:.4f}")
    print(f"Share of SO rows where volume == 0: {so_zero_volume_share:.4f}")
    print(f"Total SO abs accepted volume: {total_so_abs_volume:,.4f}")
    print(f"Total SO abs raw cost proxy: {total_so_abs_raw_cost_proxy:,.4f}")
    print(f"SO VWAP original price: {so_vwap_original_price}")
    print(
        "SO abs volume split: "
        f"BID={so_bid_abs_volume:,.4f}, OFFER={so_offer_abs_volume:,.4f}"
    )
    print(
        "SO abs raw cost proxy split: "
        f"BID={so_bid_abs_raw_cost_proxy:,.4f}, OFFER={so_offer_abs_raw_cost_proxy:,.4f}"
    )

    print("\nTop 25 SO BID BMUs by abs_volume")
    print(
        bmu_summary.loc[bmu_summary["direction"] == "BID"]
        .sort_values("abs_volume", ascending=False)
        .head(25)
        .to_string(index=False)
    )

    print("\nTop 25 SO OFFER BMUs by abs_volume")
    print(
        bmu_summary.loc[bmu_summary["direction"] == "OFFER"]
        .sort_values("abs_volume", ascending=False)
        .head(25)
        .to_string(index=False)
    )

    print("\nTop 25 SO BID BMUs by abs_raw_cost_proxy")
    print(
        bmu_summary.loc[bmu_summary["direction"] == "BID"]
        .sort_values("abs_raw_cost_proxy", ascending=False)
        .head(25)
        .to_string(index=False)
    )

    print("\nTop 25 SO OFFER BMUs by abs_raw_cost_proxy")
    print(
        bmu_summary.loc[bmu_summary["direction"] == "OFFER"]
        .sort_values("abs_raw_cost_proxy", ascending=False)
        .head(25)
        .to_string(index=False)
    )

    print("\nTop 25 BMUs by number_of_days_present")
    print(
        bmu_summary.sort_values(["number_of_days_present", "action_count"], ascending=[False, False])
        .head(25)
        .to_string(index=False)
    )

    so_actions = so_actions.drop(columns=["weighted_price_volume"])

    so_actions.to_parquet(SO_ACTIONS_PATH, index=False)
    daily_summary.to_parquet(DAILY_SUMMARY_PATH, index=False)
    sp_summary.to_parquet(SP_SUMMARY_PATH, index=False)
    bmu_summary.to_parquet(BMU_SUMMARY_PATH, index=False)
    quality_summary.to_csv(QUALITY_SUMMARY_PATH, index=False)

    print(f"\nSaved: {SO_ACTIONS_PATH}")
    print(f"Saved: {DAILY_SUMMARY_PATH}")
    print(f"Saved: {SP_SUMMARY_PATH}")
    print(f"Saved: {BMU_SUMMARY_PATH}")
    print(f"Saved: {QUALITY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
