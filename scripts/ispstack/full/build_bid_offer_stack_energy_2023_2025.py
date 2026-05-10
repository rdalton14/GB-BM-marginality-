from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

INPUT_DIR   = PROJECT_ROOT / "data" / "raw" / "ispstack" / "full_2023_2025"
DISBSAD_PATH = PROJECT_ROOT / "data" / "raw" / "ispstack" / "disbsad_2023_2025" / "disbsad_2023_2025.parquet"
BMU_REFERENCE_PATH = PROJECT_ROOT / "data" / "raw" / "reference" / "bmu_register_elexon_current.csv"
OUTPUT_DIR  = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "bid_offer_stack_2023_2025"

START_DATE = pd.Timestamp("2023-01-01")
END_DATE   = pd.Timestamp("2025-12-31")
NUMERIC_ID_RE = re.compile(r"^\d+(\.0)?$")

LONG_COLUMNS = [
    "settlementDate", "settlementPeriod", "startTime",
    "side", "stack_rank", "bid_rank_price_desc",
    "id", "resolved_bmu_id", "identity_source", "is_numeric_id",
    "disbsad_service", "disbsad_partyId", "disbsad_assetId",
    "disbsad_cost", "disbsad_volume", "disbsad_implied_price",
    "acceptanceId", "bidOfferPairId", "sequenceNumber",
    "originalPrice", "finalPrice", "volume",
    "dmatAdjustedVolume", "arbitrageAdjustedVolume", "nivAdjustedVolume",
    "parAdjustedVolume", "tlmAdjustedVolume", "tlmAdjustedCost",
    "cadlFlag", "soFlag", "storProviderFlag", "repricedIndicator",
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


def load_full() -> tuple[pd.DataFrame, int]:
    files = sorted(INPUT_DIR.glob("*.parquet"))
    files = [f for f in files if START_DATE <= pd.Timestamp(f.stem) <= END_DATE]
    if not files:
        raise RuntimeError(f"No parquet files found in {INPUT_DIR}")
    frames = []
    for i, f in enumerate(files, 1):
        if i == 1 or i % 100 == 0 or i == len(files):
            print(f"  Loading {i:,}/{len(files):,}: {f.name}")
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True), len(files)


def load_disbsad_energy_lookup() -> pd.DataFrame:
    empty = pd.DataFrame(columns=[
        "settlementDate", "settlementPeriod", "id_norm",
        "disbsad_service", "disbsad_partyId", "disbsad_assetId",
        "disbsad_cost", "disbsad_volume", "disbsad_implied_price", "resolved_bmu_id",
    ])
    if not DISBSAD_PATH.exists():
        return empty

    disbsad = pd.read_parquet(DISBSAD_PATH)
    disbsad["settlementDate"] = pd.to_datetime(disbsad["settlementDate"], errors="coerce").dt.normalize()
    disbsad["settlementPeriod"] = pd.to_numeric(disbsad["settlementPeriod"], errors="coerce").astype("Int64")
    disbsad["id_norm"] = disbsad["id"].map(normalise_id)
    disbsad = disbsad.loc[
        (disbsad["settlementDate"] >= START_DATE) & (disbsad["settlementDate"] <= END_DATE)
    ].copy()
    if disbsad.empty:
        return empty

    disbsad = disbsad.rename(columns={
        "service": "disbsad_service", "partyId": "disbsad_partyId",
        "assetId": "disbsad_assetId", "cost": "disbsad_cost", "volume": "disbsad_volume",
    })
    disbsad["disbsad_implied_price"] = (
        pd.to_numeric(disbsad["disbsad_cost"], errors="coerce")
        / pd.to_numeric(disbsad["disbsad_volume"], errors="coerce")
    )
    if BMU_REFERENCE_PATH.exists():
        bmu = pd.read_csv(BMU_REFERENCE_PATH, dtype=str)
        bmu = bmu[["nationalGridBmUnit", "elexonBmUnit"]].drop_duplicates("nationalGridBmUnit")
        disbsad = disbsad.merge(bmu, left_on="disbsad_assetId", right_on="nationalGridBmUnit", how="left")
        disbsad["resolved_bmu_id"] = disbsad["elexonBmUnit"].fillna(disbsad["nationalGridBmUnit"])
        disbsad = disbsad.drop(columns=["nationalGridBmUnit", "elexonBmUnit"], errors="ignore")
    else:
        disbsad["resolved_bmu_id"] = pd.NA

    keep = [
        "settlementDate", "settlementPeriod", "id_norm",
        "disbsad_service", "disbsad_partyId", "disbsad_assetId",
        "disbsad_cost", "disbsad_volume", "disbsad_implied_price", "resolved_bmu_id",
    ]
    return disbsad[keep].drop_duplicates(["settlementDate", "settlementPeriod", "id_norm"])


def build_long_stack(raw: pd.DataFrame, disbsad_lookup: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["settlementDate"] = pd.to_datetime(df["settlementDate"], errors="coerce").dt.normalize()
    df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce").astype("Int64")
    df["side"] = df["direction"].astype(str).str.lower()
    df["id_norm"] = df["id"].map(normalise_id)
    df["is_numeric_id"] = df["id_norm"].map(is_numeric_id)

    for col in ["sequenceNumber", "originalPrice", "finalPrice", "volume",
                "dmatAdjustedVolume", "arbitrageAdjustedVolume", "nivAdjustedVolume",
                "parAdjustedVolume", "tlmAdjustedVolume", "tlmAdjustedCost"]:
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

    SP_KEYS = ["settlementDate", "settlementPeriod"]
    sort_cols = SP_KEYS + ["side", "finalPrice", "sequenceNumber", "id_norm"]

    stack["stack_rank"] = (
        stack.sort_values(sort_cols)
        .groupby(SP_KEYS + ["side"], dropna=False)
        .cumcount() + 1
    )

    bid_desc = stack.loc[stack["side"] == "bid"].copy()
    bid_desc = bid_desc.sort_values(SP_KEYS + ["finalPrice", "sequenceNumber", "id_norm"],
                                    ascending=[True, True, False, True, True])
    stack["bid_rank_price_desc"] = pd.NA
    stack.loc[bid_desc.index, "bid_rank_price_desc"] = (
        bid_desc.groupby(SP_KEYS, dropna=False).cumcount() + 1
    ).astype("Int64")

    stack = stack.sort_values(sort_cols).reset_index(drop=True)
    available = [c for c in LONG_COLUMNS if c in stack.columns]
    return stack[available]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 2023-2025 ISPSTACK data ...")
    raw, n_files = load_full()
    print(f"  Files loaded : {n_files:,}")
    print(f"  Raw rows     : {len(raw):,}")

    print("Loading DISBSAD energy lookup ...")
    disbsad_lookup = load_disbsad_energy_lookup()
    print(f"  DISBSAD rows in date window: {len(disbsad_lookup):,}")

    print("Building energy-only stack ...")
    stack = build_long_stack(raw, disbsad_lookup)

    n_sps = stack[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    n_numeric = int(stack["is_numeric_id"].sum())
    print(f"  Stack rows   : {len(stack):,}")
    print(f"  SPs covered  : {n_sps:,} / {n_files * 48} expected")
    print(f"  Numeric IDs  : {n_numeric:,}")
    print(f"  Offer rows   : {(stack['side'] == 'offer').sum():,}")
    print(f"  Bid rows     : {(stack['side'] == 'bid').sum():,}")

    out = OUTPUT_DIR / "bid_offer_stack_energy_2023_2025_long.parquet"
    stack.to_parquet(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
