from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_DIR = Path("data/raw/ispstack/q1_2026")
DISBSAD_PATH = Path("data/raw/ispstack/disbsad_q1_2026/disbsad_q1_2026.parquet")
BMU_REFERENCE_PATH = Path("data/raw/reference/bmu_reference.csv")
OUTPUT_DIR = Path("data/processed/q1_2026/bid_offer_stack_first_week_energy")

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-01-07")
NUMERIC_ID_RE = re.compile(r"^\d+(\.0)?$")

LONG_COLUMNS = [
    "settlementDate",
    "settlementPeriod",
    "startTime",
    "side",
    "stack_rank",
    "bid_rank_price_desc",
    "id",
    "resolved_bmu_id",
    "identity_source",
    "is_numeric_id",
    "disbsad_service",
    "disbsad_partyId",
    "disbsad_assetId",
    "disbsad_cost",
    "disbsad_volume",
    "disbsad_implied_price",
    "acceptanceId",
    "bidOfferPairId",
    "sequenceNumber",
    "originalPrice",
    "finalPrice",
    "volume",
    "dmatAdjustedVolume",
    "arbitrageAdjustedVolume",
    "nivAdjustedVolume",
    "parAdjustedVolume",
    "tlmAdjustedVolume",
    "tlmAdjustedCost",
    "cadlFlag",
    "soFlag",
    "storProviderFlag",
    "repricedIndicator",
]


def is_numeric_id(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(NUMERIC_ID_RE.fullmatch(str(value).strip()))


def normalise_id(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if NUMERIC_ID_RE.fullmatch(text):
        return str(int(float(text)))
    return text


def load_first_week() -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        path = INPUT_DIR / f"{day.date()}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True), len(frames)


def load_disbsad_energy_lookup() -> pd.DataFrame:
    if not DISBSAD_PATH.exists():
        return pd.DataFrame(
            columns=[
                "settlementDate",
                "settlementPeriod",
                "id_norm",
                "disbsad_service",
                "disbsad_partyId",
                "disbsad_assetId",
                "disbsad_cost",
                "disbsad_volume",
                "disbsad_implied_price",
                "resolved_bmu_id",
            ]
        )

    disbsad = pd.read_parquet(DISBSAD_PATH)
    disbsad["settlementDate"] = pd.to_datetime(disbsad["settlementDate"], errors="coerce").dt.normalize()
    disbsad["settlementPeriod"] = pd.to_numeric(disbsad["settlementPeriod"], errors="coerce").astype("Int64")
    disbsad["id_norm"] = disbsad["id"].map(normalise_id)
    disbsad = disbsad.loc[(disbsad["settlementDate"] >= START_DATE) & (disbsad["settlementDate"] <= END_DATE)].copy()

    disbsad = disbsad.rename(
        columns={
            "service": "disbsad_service",
            "partyId": "disbsad_partyId",
            "assetId": "disbsad_assetId",
            "cost": "disbsad_cost",
            "volume": "disbsad_volume",
        }
    )
    disbsad["disbsad_implied_price"] = pd.to_numeric(disbsad["disbsad_cost"], errors="coerce") / pd.to_numeric(
        disbsad["disbsad_volume"], errors="coerce"
    )

    if BMU_REFERENCE_PATH.exists():
        bmu = pd.read_csv(BMU_REFERENCE_PATH, dtype=str)
        bmu = bmu[["nationalGridBmUnit", "elexonBmUnit", "leadPartyName", "fuelType"]].drop_duplicates("nationalGridBmUnit")
        disbsad = disbsad.merge(
            bmu,
            left_on="disbsad_assetId",
            right_on="nationalGridBmUnit",
            how="left",
        )
        disbsad["resolved_bmu_id"] = disbsad["elexonBmUnit"].fillna(disbsad["nationalGridBmUnit"])
        disbsad = disbsad.drop(columns=["nationalGridBmUnit", "elexonBmUnit", "leadPartyName", "fuelType"], errors="ignore")
    else:
        disbsad["resolved_bmu_id"] = pd.NA

    keep = [
        "settlementDate",
        "settlementPeriod",
        "id_norm",
        "disbsad_service",
        "disbsad_partyId",
        "disbsad_assetId",
        "disbsad_cost",
        "disbsad_volume",
        "disbsad_implied_price",
        "resolved_bmu_id",
    ]
    return disbsad[keep].drop_duplicates(["settlementDate", "settlementPeriod", "id_norm"])


def build_long_stack(raw: pd.DataFrame, disbsad_lookup: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["settlementDate"] = pd.to_datetime(df["settlementDate"], errors="coerce").dt.normalize()
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    df["side"] = df["direction"].astype(str).str.lower()
    df["id_norm"] = df["id"].map(normalise_id)
    df["is_numeric_id"] = df["id_norm"].map(is_numeric_id)

    numeric_cols = [
        "sequenceNumber",
        "originalPrice",
        "finalPrice",
        "volume",
        "dmatAdjustedVolume",
        "arbitrageAdjustedVolume",
        "nivAdjustedVolume",
        "parAdjustedVolume",
        "tlmAdjustedVolume",
        "tlmAdjustedCost",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    stack = df.loc[
        df["side"].isin(["bid", "offer"])
        & df["finalPrice"].notna()
        & df["volume"].notna()
        & (df["volume"] != 0)
        & (df["cadlFlag"] != True)
        & (df["soFlag"] != True)
        & (df["storProviderFlag"] != True)
    ].copy()

    stack = stack.merge(disbsad_lookup, on=["settlementDate", "settlementPeriod", "id_norm"], how="left")
    stack["is_disbsad_energy"] = stack["disbsad_service"].astype("string").str.casefold().eq("energy")
    stack = stack.loc[(~stack["is_numeric_id"]) | stack["is_disbsad_energy"]].copy()

    stack["identity_source"] = "named_ispstack_id"
    stack.loc[stack["is_numeric_id"] & stack["is_disbsad_energy"], "identity_source"] = "disbsad_energy_asset"
    stack["resolved_bmu_id"] = stack["resolved_bmu_id"].where(stack["is_numeric_id"], stack["id_norm"])

    sort_cols = ["settlementDate", "settlementPeriod", "side", "finalPrice", "sequenceNumber", "id_norm"]
    stack = stack.sort_values(sort_cols, ascending=[True, True, True, True, True, True]).reset_index(drop=True)

    # Economic stack rank: offers cheap-to-expensive; bids low-to-high as decremental merit order.
    stack["stack_rank"] = (
        stack.sort_values(["settlementDate", "settlementPeriod", "side", "finalPrice", "sequenceNumber", "id_norm"])
        .groupby(["settlementDate", "settlementPeriod", "side"], dropna=False)
        .cumcount()
        + 1
    )

    bid_desc = stack.loc[stack["side"] == "bid"].copy()
    bid_desc = bid_desc.sort_values(
        ["settlementDate", "settlementPeriod", "finalPrice", "sequenceNumber", "id_norm"],
        ascending=[True, True, False, True, True],
    )
    stack["bid_rank_price_desc"] = pd.NA
    stack.loc[bid_desc.index, "bid_rank_price_desc"] = (
        bid_desc.groupby(["settlementDate", "settlementPeriod"], dropna=False).cumcount() + 1
    ).astype("Int64")

    return stack[LONG_COLUMNS].sort_values(["settlementDate", "settlementPeriod", "side", "stack_rank"]).reset_index(drop=True)


def build_wide_stack(long_stack: pd.DataFrame) -> pd.DataFrame:
    bid_cols = {
        "id": "bid_id",
        "resolved_bmu_id": "bid_resolved_bmu_id",
        "identity_source": "bid_identity_source",
        "finalPrice": "bid_price",
        "volume": "bid_volume",
        "originalPrice": "bid_originalPrice",
        "acceptanceId": "bid_acceptanceId",
        "bidOfferPairId": "bid_bidOfferPairId",
        "sequenceNumber": "bid_sequenceNumber",
        "disbsad_service": "bid_disbsad_service",
        "disbsad_partyId": "bid_disbsad_partyId",
        "disbsad_assetId": "bid_disbsad_assetId",
    }
    offer_cols = {
        "id": "offer_id",
        "resolved_bmu_id": "offer_resolved_bmu_id",
        "identity_source": "offer_identity_source",
        "finalPrice": "offer_price",
        "volume": "offer_volume",
        "originalPrice": "offer_originalPrice",
        "acceptanceId": "offer_acceptanceId",
        "bidOfferPairId": "offer_bidOfferPairId",
        "sequenceNumber": "offer_sequenceNumber",
        "disbsad_service": "offer_disbsad_service",
        "disbsad_partyId": "offer_disbsad_partyId",
        "disbsad_assetId": "offer_disbsad_assetId",
    }

    bids = long_stack.loc[long_stack["side"] == "bid", ["settlementDate", "settlementPeriod", "stack_rank", *bid_cols.keys()]].rename(
        columns=bid_cols
    )
    offers = long_stack.loc[
        long_stack["side"] == "offer", ["settlementDate", "settlementPeriod", "stack_rank", *offer_cols.keys()]
    ].rename(columns=offer_cols)

    wide = bids.merge(offers, on=["settlementDate", "settlementPeriod", "stack_rank"], how="outer")
    return wide.sort_values(["settlementDate", "settlementPeriod", "stack_rank"]).reset_index(drop=True)


def build_quality_summary(raw: pd.DataFrame, long_stack: pd.DataFrame, file_count: int) -> pd.DataFrame:
    sp_count = long_stack[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    expected_sp_count = len(pd.date_range(START_DATE, END_DATE, freq="D")) * 48
    numeric_rows = long_stack.loc[long_stack["is_numeric_id"]]
    metrics = [
        ("files_loaded", file_count),
        ("raw_rows_first_week", len(raw)),
        ("energy_stack_rows", len(long_stack)),
        ("energy_bid_rows", int((long_stack["side"] == "bid").sum())),
        ("energy_offer_rows", int((long_stack["side"] == "offer").sum())),
        ("settlement_periods_covered", sp_count),
        ("expected_settlement_periods", expected_sp_count),
        ("settlement_period_coverage_share", sp_count / expected_sp_count if expected_sp_count else 0),
        ("numeric_disbsad_energy_rows_retained", len(numeric_rows)),
        ("numeric_disbsad_energy_rows_with_resolved_bmu_id", int(numeric_rows["resolved_bmu_id"].notna().sum())),
        ("unique_bid_ids", int(long_stack.loc[long_stack["side"] == "bid", "resolved_bmu_id"].nunique(dropna=True))),
        ("unique_offer_ids", int(long_stack.loc[long_stack["side"] == "offer", "resolved_bmu_id"].nunique(dropna=True))),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def build_core_long_stack(long_stack: pd.DataFrame) -> pd.DataFrame:
    core_cols = [
        "settlementDate",
        "settlementPeriod",
        "startTime",
        "side",
        "stack_rank",
        "bid_rank_price_desc",
        "id",
        "resolved_bmu_id",
        "identity_source",
        "is_numeric_id",
        "acceptanceId",
        "bidOfferPairId",
        "sequenceNumber",
        "originalPrice",
        "finalPrice",
        "volume",
        "repricedIndicator",
    ]
    return long_stack[core_cols].copy()


def build_core_wide_stack(core_long: pd.DataFrame) -> pd.DataFrame:
    bid_cols = {
        "id": "bid_id",
        "resolved_bmu_id": "bid_resolved_bmu_id",
        "identity_source": "bid_identity_source",
        "finalPrice": "bid_price",
        "volume": "bid_volume",
        "acceptanceId": "bid_acceptanceId",
        "bidOfferPairId": "bid_bidOfferPairId",
        "sequenceNumber": "bid_sequenceNumber",
    }
    offer_cols = {
        "id": "offer_id",
        "resolved_bmu_id": "offer_resolved_bmu_id",
        "identity_source": "offer_identity_source",
        "finalPrice": "offer_price",
        "volume": "offer_volume",
        "acceptanceId": "offer_acceptanceId",
        "bidOfferPairId": "offer_bidOfferPairId",
        "sequenceNumber": "offer_sequenceNumber",
    }
    bids = core_long.loc[core_long["side"] == "bid", ["settlementDate", "settlementPeriod", "stack_rank", *bid_cols]].rename(
        columns=bid_cols
    )
    offers = core_long.loc[
        core_long["side"] == "offer", ["settlementDate", "settlementPeriod", "stack_rank", *offer_cols]
    ].rename(columns=offer_cols)
    return bids.merge(offers, on=["settlementDate", "settlementPeriod", "stack_rank"], how="outer").sort_values(
        ["settlementDate", "settlementPeriod", "stack_rank"]
    )


def format_stack_item(row: pd.Series) -> str:
    bmu = row["resolved_bmu_id"] if pd.notna(row["resolved_bmu_id"]) else row["id"]
    price = "" if pd.isna(row["finalPrice"]) else f"{row['finalPrice']:.6g}"
    volume = "" if pd.isna(row["volume"]) else f"{row['volume']:.6g}"
    return f"{int(row['stack_rank'])}:{bmu}@{price}({volume})"


def build_stack_lists_by_sp(core_long: pd.DataFrame) -> pd.DataFrame:
    source = core_long.sort_values(["settlementDate", "settlementPeriod", "side", "stack_rank"]).copy()
    source["stack_item"] = source.apply(format_stack_item, axis=1)
    lists = (
        source.groupby(["settlementDate", "settlementPeriod", "side"], dropna=False)["stack_item"]
        .apply(lambda s: " | ".join(s.astype(str)))
        .reset_index()
    )
    counts = (
        source.groupby(["settlementDate", "settlementPeriod", "side"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    bid_lists = lists.loc[lists["side"] == "bid"].drop(columns="side").rename(columns={"stack_item": "bid_stack"})
    offer_lists = lists.loc[lists["side"] == "offer"].drop(columns="side").rename(columns={"stack_item": "offer_stack"})
    bid_counts = counts.loc[counts["side"] == "bid"].drop(columns="side").rename(columns={"count": "bid_count"})
    offer_counts = counts.loc[counts["side"] == "offer"].drop(columns="side").rename(columns={"count": "offer_count"})

    all_sps = core_long[["settlementDate", "settlementPeriod"]].drop_duplicates()
    by_sp = (
        all_sps.merge(bid_counts, on=["settlementDate", "settlementPeriod"], how="left")
        .merge(offer_counts, on=["settlementDate", "settlementPeriod"], how="left")
        .merge(bid_lists, on=["settlementDate", "settlementPeriod"], how="left")
        .merge(offer_lists, on=["settlementDate", "settlementPeriod"], how="left")
        .sort_values(["settlementDate", "settlementPeriod"])
        .reset_index(drop=True)
    )
    by_sp["bid_count"] = by_sp["bid_count"].fillna(0).astype(int)
    by_sp["offer_count"] = by_sp["offer_count"].fillna(0).astype(int)
    by_sp["bid_stack"] = by_sp["bid_stack"].fillna("")
    by_sp["offer_stack"] = by_sp["offer_stack"].fillna("")
    return by_sp


def build_disbsad_resolution_audit(long_stack: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "settlementDate",
        "settlementPeriod",
        "side",
        "stack_rank",
        "id",
        "resolved_bmu_id",
        "disbsad_service",
        "disbsad_partyId",
        "disbsad_assetId",
        "disbsad_cost",
        "disbsad_volume",
        "disbsad_implied_price",
        "finalPrice",
        "volume",
    ]
    return long_stack.loc[long_stack["is_numeric_id"], cols].copy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw, file_count = load_first_week()
    disbsad_lookup = load_disbsad_energy_lookup()
    long_stack = build_long_stack(raw, disbsad_lookup)
    wide_stack = build_wide_stack(long_stack)
    core_long_stack = build_core_long_stack(long_stack)
    core_wide_stack = build_core_wide_stack(core_long_stack)
    stack_lists_by_sp = build_stack_lists_by_sp(core_long_stack)
    disbsad_resolution_audit = build_disbsad_resolution_audit(long_stack)
    quality = build_quality_summary(raw, long_stack, file_count)

    long_csv = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_long.csv"
    long_parquet = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_long.parquet"
    wide_csv = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_wide_by_rank.csv"
    wide_parquet = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_wide_by_rank.parquet"
    quality_csv = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_quality_summary.csv"
    core_long_csv = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_core_long.csv"
    core_wide_csv = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_core_wide_by_rank.csv"
    stack_lists_csv = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_stack_lists_by_sp.csv"
    disbsad_audit_csv = OUTPUT_DIR / "bid_offer_stack_energy_first_week_q1_2026_disbsad_numeric_audit.csv"

    long_stack.to_csv(long_csv, index=False)
    long_stack.to_parquet(long_parquet, index=False)
    wide_stack.to_csv(wide_csv, index=False)
    wide_stack.to_parquet(wide_parquet, index=False)
    core_long_stack.to_csv(core_long_csv, index=False)
    core_wide_stack.to_csv(core_wide_csv, index=False)
    stack_lists_by_sp.to_csv(stack_lists_csv, index=False)
    disbsad_resolution_audit.to_csv(disbsad_audit_csv, index=False)
    quality.to_csv(quality_csv, index=False)

    print(f"Files loaded: {file_count}")
    print(f"Raw rows first week: {len(raw):,}")
    print(f"Energy stack rows: {len(long_stack):,}")
    print(f"Bid rows: {(long_stack['side'] == 'bid').sum():,}")
    print(f"Offer rows: {(long_stack['side'] == 'offer').sum():,}")
    print(f"Wide rows: {len(wide_stack):,}")
    print("\nQuality summary")
    print(quality.to_string(index=False))
    print("\nFirst 20 wide rows")
    print(wide_stack.head(20).to_string(index=False))
    print(f"\nSaved: {long_csv}")
    print(f"Saved: {long_parquet}")
    print(f"Saved: {wide_csv}")
    print(f"Saved: {wide_parquet}")
    print(f"Saved: {core_long_csv}")
    print(f"Saved: {core_wide_csv}")
    print(f"Saved: {stack_lists_csv}")
    print(f"Saved: {disbsad_audit_csv}")
    print(f"Saved: {quality_csv}")


if __name__ == "__main__":
    main()
