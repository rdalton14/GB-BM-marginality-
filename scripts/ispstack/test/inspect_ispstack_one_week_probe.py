from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
INPUT_PATH = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / "ispstack_one_week_probe_2026_01_05_to_2026_01_11" / "ispstack_one_week_long.parquet"
OUT_DIR = INPUT_PATH.parent


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def value_counts_dict(series: pd.Series, top_n: int = 20) -> dict:
    counts = series.value_counts(dropna=False).head(top_n)
    out = {}
    for key, value in counts.items():
        out[str(key)] = int(value)
    return out


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    print_section("Basic Shape")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Columns list: {sorted(df.columns.tolist())}")

    print_section("Head")
    print(df.head(10).to_string())

    print_section("Direction x Sign Checks")
    sign_summary = (
        df.assign(
            volume_sign=pd.Series(pd.NA, index=df.index, dtype="object")
        )
    )
    sign_summary.loc[df["volume"] > 0, "volume_sign"] = "positive"
    sign_summary.loc[df["volume"] < 0, "volume_sign"] = "negative"
    sign_summary.loc[df["volume"] == 0, "volume_sign"] = "zero"
    print(
        sign_summary.groupby(["direction", "volume_sign"]).size().reset_index(name="row_count").to_string(index=False)
    )

    print_section("Price Field Diagnostics")
    for col in ["originalPrice", "finalPrice", "reserveScarcityPrice"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"{col}: min={s.min()} max={s.max()} mean={s.mean()}")
            sentinel_counts = {
                "-99999": int((s == -99999).sum()),
                "9999": int((s == 9999).sum()),
                "0": int((s == 0).sum()),
            }
            print(f"  sentinel counts: {sentinel_counts}")

    print_section("Adjusted Volume Diagnostics")
    for col in ["volume", "dmatAdjustedVolume", "arbitrageAdjustedVolume", "nivAdjustedVolume", "parAdjustedVolume", "tlmAdjustedVolume"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"{col}: non_null={int(s.notna().sum()):,} min={s.min()} max={s.max()} mean={s.mean()}")

    print_section("ID Quality")
    id_series = df["id"].astype(str)
    id_quality = {
        "n_unique_ids": int(id_series.nunique()),
        "ids_starting_with_letter_prefix": int(id_series.str.match(r"^[A-Z0-9]_[A-Z_0-9-]+$").sum()),
        "ids_pure_digits": int(id_series.str.match(r"^\\d+$").sum()),
        "ids_with_double_underscore": int(id_series.str.contains(r"__").sum()),
        "ids_with_dash": int(id_series.str.contains(r"-").sum()),
    }
    print(json.dumps(id_quality, indent=2))
    print("Top 30 ids:")
    print(pd.Series(value_counts_dict(id_series, top_n=30)).to_string())

    print_section("Sequence Number Uniqueness")
    dupes = (
        df.groupby(["settlementDate", "settlementPeriod", "direction", "sequenceNumber"])
        .size()
        .reset_index(name="n")
    )
    print(f"Duplicate sequence-number keys: {int((dupes['n'] > 1).sum())}")
    seq_by_side = df.groupby("direction")["sequenceNumber"].agg(["min", "max", "nunique"])
    print(seq_by_side.to_string())

    print_section("Flags")
    for col in ["soFlag", "repricedIndicator", "cadlFlag", "storProviderFlag"]:
        if col in df.columns:
            print(f"{col}:")
            print(df.groupby(["direction", col]).size().reset_index(name="row_count").to_string(index=False))

    print_section("Bid/Offer Pair IDs")
    if "bidOfferPairId" in df.columns:
        pair = pd.to_numeric(df["bidOfferPairId"], errors="coerce")
        print(f"min={pair.min()} max={pair.max()} unique={pair.nunique(dropna=True)}")
        print(pd.Series(value_counts_dict(pair, top_n=20)).to_string())

    print_section("Plant Key Cross-Check")
    if {"id", "acceptanceId"}.issubset(df.columns):
        print("acceptanceId null count:", int(df["acceptanceId"].isna().sum()))
        print("Rows with numeric-only id:")
        print(df.loc[df["id"].astype(str).str.match(r"^\\d+$"), ["settlementDate", "settlementPeriod", "direction", "id", "acceptanceId", "bidOfferPairId", "originalPrice", "finalPrice", "volume"]].head(20).to_string(index=False))

    print_section("Stack Ordering Hints")
    order_probe = (
        df.groupby(["settlementDate", "settlementPeriod", "direction"])
        .agg(
            n_rows=("sequenceNumber", "size"),
            seq_min=("sequenceNumber", "min"),
            seq_max=("sequenceNumber", "max"),
            price_min=("finalPrice", "min"),
            price_max=("finalPrice", "max"),
            vol_min=("volume", "min"),
            vol_max=("volume", "max"),
        )
        .reset_index()
    )
    print(order_probe.head(20).to_string(index=False))

    print_section("Potential Sentinel Price Rows")
    sentinel = df[df["finalPrice"].isin([-99999, 9999]) | df["originalPrice"].isin([-99999, 9999])]
    print(f"Rows with sentinel price values: {len(sentinel):,}")
    print(
        sentinel[["settlementDate", "settlementPeriod", "direction", "id", "soFlag", "repricedIndicator", "originalPrice", "finalPrice", "volume"]]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
