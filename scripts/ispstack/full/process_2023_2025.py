"""
process_2023_2025.py
--------------------
Reads raw ISPStack parquet files for 2023-01-01 to 2025-12-31 and constructs
a master settlement-period panel for analysis.

Input
-----
  data/raw/ispstack/full_2023_2025/       (folder of per-date parquet files)
  data/raw/reference/bmu_register_q1_2026.csv      (547 BMUs, manually verified)
  data/raw/reference/bmu_register_supplement_2023_2025.csv   (69 additional BMUs)
  Elexon BMRS API  /balancing/settlement/system-prices/{date}  (NIV)

Output
------
  data/processed/full_2023_2025/master_panel_2023_2025.parquet
  data/processed/full_2023_2025/master_panel_2023_2025.csv

Schema note
-----------
  Some columns (e.g. reserveScarcityPrice) are null-typed in early 2023
  parquet files and float64 in later files. To avoid a pyarrow schema
  conflict, only the 9 columns actually used by the pipeline are loaded.
"""

import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
DATA_DIR  = PROJECT_ROOT / "data"
PROC_DIR  = DATA_DIR / "processed" / "full_2023_2025"
PROC_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH      = DATA_DIR / "raw" / "ispstack" / "full_2023_2025"
FUEL_TYPE_PATH  = DATA_DIR / "raw" / "reference" / "BMUFuelType (1).xlsx"
BMU_REGISTER    = DATA_DIR / "raw" / "reference" / "bmu_register_q1_2026.csv"
BMU_SUPPLEMENT  = DATA_DIR / "raw" / "reference" / "bmu_register_supplement_2023_2025.csv"
PANEL_OUT       = PROC_DIR / "master_panel_2023_2025.parquet"
CSV_OUT         = PROC_DIR / "master_panel_2023_2025.csv"

SYSTEM_PRICES_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"

# Only the columns the pipeline uses — avoids schema conflicts across parquet files
ISP_COLUMNS = [
    "settlementDate", "settlementPeriod",
    "id", "direction",
    "soFlag", "repricedIndicator", "bidOfferPairId",
    "originalPrice", "volume",
]

SP_KEY = ["settlementDate", "settlementPeriod"]

# ---------------------------------------------------------------------------
# Step 1 — Load and inspect
# ---------------------------------------------------------------------------
print("=" * 55)
print("STEP 1 — LOAD AND INSPECT")
print("=" * 55)

isp = pd.read_parquet(INPUT_PATH, columns=ISP_COLUMNS)
print(f"\nISPStack shape        : {isp.shape}")
print(f"Date range            : {isp['settlementDate'].min()} to {isp['settlementDate'].max()}")
print(f"Unique dates          : {isp['settlementDate'].nunique():,}")
print(f"Unique SPs per day    : {isp['settlementPeriod'].nunique()}")
print(f"Unique BMU IDs        : {isp['id'].nunique():,}")
print(f"Null id rows          : {isp['id'].isna().sum():,}")

# ---------------------------------------------------------------------------
# Step 2 — BMU technology classification
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("STEP 2 — BMU TECHNOLOGY CLASSIFICATION")
print("=" * 55)

# Load primary register (547 Q1 2026 verified BMUs)
register_q1 = (
    pd.read_csv(BMU_REGISTER, usecols=["elexon_bmu_id", "reg_fuel_type"])
    .drop_duplicates(subset=["elexon_bmu_id"])
)
print(f"\nPrimary register (Q1 2026)     : {len(register_q1):,} BMUs")
print(f"  BATTERY                      : {register_q1['reg_fuel_type'].value_counts().get('BATTERY', 0)}")
print(f"  Null reg_fuel_type           : {register_q1['reg_fuel_type'].isna().sum()}")

# Load supplementary register (69 additional BMUs from 2023-25 gap analysis)
register_supp = (
    pd.read_csv(BMU_SUPPLEMENT, usecols=["elexon_bmu_id", "reg_fuel_type"])
    .drop_duplicates(subset=["elexon_bmu_id"])
)
print(f"Supplementary register (23-25) : {len(register_supp):,} BMUs")
print(f"  BATTERY                      : {register_supp['reg_fuel_type'].value_counts().get('BATTERY', 0)}")

# Combine: primary takes precedence over supplement
combined_register = pd.concat([register_q1, register_supp], ignore_index=True)
combined_register = combined_register.drop_duplicates(subset=["elexon_bmu_id"], keep="first")
print(f"Combined register              : {len(combined_register):,} BMUs")

isp = isp.merge(
    combined_register.rename(columns={"elexon_bmu_id": "id"}),
    on="id",
    how="left",
)

# Also try Excel register three-join for any remaining unmatched BMUs
# Join 1: direct SETT UNIT ID match
# Join 2: strip prefix (T_, E_, 2__, V__) and match NESO BMU ID
xl = pd.read_excel(FUEL_TYPE_PATH, dtype=str)
xl.columns = [c.strip() for c in xl.columns]
sett_map = (
    xl.dropna(subset=["SETT UNIT ID"])
    .set_index("SETT UNIT ID")["REG FUEL TYPE"]
    .to_dict()
)
neso_map = (
    xl.dropna(subset=["NESO BMU ID"])
    .set_index("NESO BMU ID")["REG FUEL TYPE"]
    .to_dict()
)

def xl_lookup(bmu_id):
    if pd.isna(bmu_id):
        return None
    s = str(bmu_id)
    if s in sett_map:
        return sett_map[s]
    stripped = s.replace("2__", "").replace("V__", "").replace("T_", "").replace("E_", "")
    return neso_map.get(stripped) or neso_map.get(s)

# Fill unmatched rows from Excel
unmatched_mask = isp["reg_fuel_type"].isna() & isp["id"].notna()
xl_fills = isp.loc[unmatched_mask, "id"].map(xl_lookup)
isp.loc[unmatched_mask, "reg_fuel_type"] = xl_fills.values

n_matched = isp["reg_fuel_type"].notna().sum()
n_unclass = isp["reg_fuel_type"].isna().sum()
print(f"\nRows matched to a reg_fuel_type : {n_matched:,} / {len(isp):,} ({100*n_matched/len(isp):.1f}%)")
if n_unclass:
    unmatched_ids = isp[isp["reg_fuel_type"].isna()]["id"].dropna().unique()
    numeric_ids   = [b for b in unmatched_ids if str(b).isdigit()]
    named_ids     = [b for b in unmatched_ids if not str(b).isdigit()]
    print(f"Unmatched rows               : {n_unclass:,}")
    print(f"  Numeric virtual IDs (ok)   : {len(numeric_ids)}")
    print(f"  Unclassified named BMUs    : {len(named_ids)}")
    if named_ids:
        print(f"  Named unclassified         : {named_ids[:20]}")

isp["tech"] = isp["reg_fuel_type"].fillna("UNCLASSIFIED").str.strip().str.upper()

print(f"\nFull technology classification counts:")
print(isp["tech"].value_counts(dropna=False).to_string())

# ---------------------------------------------------------------------------
# Step 3 — Separate bids and offers
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("STEP 3 — SEPARATE BIDS AND OFFERS")
print("=" * 55)

offers = isp[isp["direction"] == "offer"].copy()
bids   = isp[isp["direction"] == "bid"].copy()
print(f"  Offer rows : {len(offers):,}")
print(f"  Bid rows   : {len(bids):,}")

off_energy = offers["soFlag"] == False
off_system = offers["soFlag"] == True
bid_energy = bids["soFlag"]   == False

# Marginal candidate filter (offer side): bidOfferPairId > 0 for genuine price-setting offers.
# Exclude originalPrice >= 9999 — the Elexon technical cap (and above-cap submissions
# e.g. 99999) are not real market prices. Preserves legitimate extreme prices (e.g. 5750).
marginal_mask = (
    (offers["soFlag"] == False) &
    (offers["bidOfferPairId"] > 0) &
    (offers["repricedIndicator"] == False) &
    (offers["originalPrice"] < 9999)
)

# ---------------------------------------------------------------------------
# Step 4 — Per-SP aggregations
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("STEP 4 — PER-SP AGGREGATIONS")
print("=" * 55)

# --- Offer aggregations ---
off_all = (
    offers.groupby(SP_KEY)
    .agg(
        offer_volume_total  =("volume", "sum"),
        offer_n_bmus        =("id", "nunique"),
        offer_n_energy      =("soFlag", lambda x: (x == False).sum()),
        offer_n_system      =("soFlag", lambda x: (x == True).sum()),
    )
    .reset_index()
)
off_energy_vol = (
    offers[off_energy].groupby(SP_KEY)["volume"].sum()
    .reset_index().rename(columns={"volume": "offer_volume_energy"})
)
off_system_vol = (
    offers[off_system].groupby(SP_KEY)["volume"].sum()
    .reset_index().rename(columns={"volume": "offer_volume_system"})
)

# Exclude 9999 sentinel (Elexon API price cap) — SPs with real prices below cap.
off_price_mask = (
    (offers["soFlag"] == False) &
    (offers["repricedIndicator"] == False) &
    (offers["originalPrice"] != 9999)
)
off_max_price = (
    offers[off_price_mask].groupby(SP_KEY)["originalPrice"].max()
    .reset_index().rename(columns={"originalPrice": "offer_max_price"})
)
n_sentinel_sps = (offers[offers["originalPrice"] == 9999]["soFlag"] == False).sum()
print(f"  9999 sentinel rows (energy offers) filtered: {n_sentinel_sps:,}")

# --- Bid aggregations ---
bid_all = (
    bids.groupby(SP_KEY)
    .agg(
        bid_volume_total =("volume", "sum"),
        bid_n_bmus       =("id", "nunique"),
        bid_n_energy     =("soFlag", lambda x: (x == False).sum()),
        bid_n_system     =("soFlag", lambda x: (x == True).sum()),
    )
    .reset_index()
)
bid_energy_vol = (
    bids[bid_energy].groupby(SP_KEY)["volume"].sum()
    .reset_index().rename(columns={"volume": "bid_volume_energy"})
)
bid_system_vol = (
    bids[~bid_energy].groupby(SP_KEY)["volume"].sum()
    .reset_index().rename(columns={"volume": "bid_volume_system"})
)
bid_min_price = (
    bids[bid_energy].groupby(SP_KEY)["originalPrice"].min()
    .reset_index().rename(columns={"originalPrice": "bid_min_price"})
)

# BESS bid volume — total accepted BESS bid volume per SP (absolute value)
bess_bid_vol = (
    bids[bid_energy & (bids["tech"] == "BATTERY")]
    .groupby(SP_KEY)["volume"]
    .apply(lambda x: x.abs().sum())
    .reset_index().rename(columns={"volume": "bess_bid_volume"})
)

# --- Tech volumes (energy offers only) ---
energy_by_tech = (
    offers[off_energy]
    .groupby(SP_KEY + ["tech"])["volume"]
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)
energy_by_tech.columns.name = None
energy_by_tech = energy_by_tech.rename(
    columns={c: f"vol_{c.lower().replace(' ', '_')}"
             for c in energy_by_tech.columns if c not in SP_KEY}
)
# vol_battery stays in the vol_* family; sum(vol_*) == offer_volume_energy (verified below)
if "vol_battery" not in energy_by_tech.columns:
    energy_by_tech["vol_battery"] = 0.0

# --- Marginal unit per SP (two-sided) ---
# Short system: highest-priced non-repriced energy offer with valid pair ID
marginal_candidates = offers[marginal_mask].copy()
marginal_idx = marginal_candidates.groupby(SP_KEY)["originalPrice"].idxmax()
marginal_offer = (
    marginal_candidates.loc[marginal_idx, SP_KEY + ["originalPrice", "id", "tech"]]
    .rename(columns={"originalPrice": "marginal_price",
                     "id":            "marginal_bmu",
                     "tech":          "marginal_tech"})
    .reset_index(drop=True)
)

# Long system: highest (least negative) non-repriced energy bid
# bidOfferPairId is negative for bids (-1, -2, -3 ...)
bid_marginal_mask = (
    (bids["soFlag"] == False) &
    (bids["bidOfferPairId"] < 0) &
    (bids["repricedIndicator"] == False)
)
bid_marginal_candidates = bids[bid_marginal_mask].copy()
bid_marginal_idx = bid_marginal_candidates.groupby(SP_KEY)["originalPrice"].idxmax()
marginal_bid = (
    bid_marginal_candidates.loc[bid_marginal_idx, SP_KEY + ["originalPrice", "id", "tech"]]
    .rename(columns={"originalPrice": "marginal_price",
                     "id":            "marginal_bmu",
                     "tech":          "marginal_tech"})
    .reset_index(drop=True)
)

# Combine: prefer offer-side; fill remaining SPs with bid-side
marginal = marginal_offer.merge(
    marginal_bid, on=SP_KEY, how="outer", suffixes=("", "_bid")
)
for col in ["marginal_price", "marginal_bmu", "marginal_tech"]:
    marginal[col] = marginal[col].fillna(marginal[f"{col}_bid"])
marginal = marginal.drop(columns=[c for c in marginal.columns if c.endswith("_bid")])

n_offer_side = marginal_offer[SP_KEY].apply(tuple, axis=1).nunique()
n_bid_side   = marginal["marginal_price"].notna().sum() - n_offer_side
print(f"\n  Marginal from offer-side (short system): {n_offer_side:,} SPs")
print(f"  Marginal from bid-side  (long system) : {n_bid_side:,} SPs")

# ---------------------------------------------------------------------------
# Step 4b — Fetch Net Imbalance Volume from Elexon BMRS
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("STEP 4b — FETCH NET IMBALANCE VOLUME (BMRS)")
print("=" * 55)

_session = requests.Session()
_session.headers["Accept"] = "application/json"

unique_dates = sorted(isp["settlementDate"].unique())
niv_records: list[dict] = []
n_failed = 0

print(f"  Fetching system prices for {len(unique_dates):,} dates ...")
for i, date_str in enumerate(unique_dates):
    if (i + 1) % 100 == 0:
        print(f"  ... {i+1}/{len(unique_dates)} dates fetched")
    try:
        resp = _session.get(
            f"{SYSTEM_PRICES_URL}/{date_str}",
            params={"format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        for row in data:
            niv_records.append({
                "settlementDate":   row["settlementDate"],
                "settlementPeriod": row["settlementPeriod"],
                "niv":              row.get("netImbalanceVolume"),
            })
        time.sleep(0.05)
    except Exception as exc:
        print(f"  WARNING: system-prices fetch failed for {date_str}: {exc}")
        n_failed += 1

niv_df = pd.DataFrame(niv_records)
if not niv_df.empty:
    niv_df["settlementDate"]   = niv_df["settlementDate"].astype(str).str[:10]
    niv_df["settlementPeriod"] = niv_df["settlementPeriod"].astype(int)
    niv_df = niv_df.drop_duplicates(subset=SP_KEY)
    n_niv = niv_df["niv"].notna().sum()
    print(f"  NIV records fetched : {len(niv_df):,}  ({n_niv:,} non-null)")
    print(f"  Failed date fetches : {n_failed}")
    print(f"  NIV stats — mean: {niv_df['niv'].mean():.1f}  "
          f"min: {niv_df['niv'].min():.1f}  max: {niv_df['niv'].max():.1f}")
else:
    print("  WARNING: no NIV data retrieved — niv column will be all NaN")

# --- Assemble panel ---
panel = off_all.copy()
for df_part in [
    off_energy_vol, off_system_vol, off_max_price,
    bid_all, bid_energy_vol, bid_system_vol, bid_min_price,
    bess_bid_vol,
    energy_by_tech, marginal,
]:
    panel = panel.merge(df_part, on=SP_KEY, how="left")

if not niv_df.empty:
    panel = panel.merge(niv_df, on=SP_KEY, how="left")
else:
    panel["niv"] = float("nan")

# Derived columns
panel["offer_system_share"] = (
    panel["offer_n_system"] /
    (panel["offer_n_energy"] + panel["offer_n_system"]).replace(0, float("nan"))
).round(4)

# gross_bm_volume: energy-only net dispatch (offer minus bid), NaN only when both absent
_offer_e = panel["offer_volume_energy"].fillna(0)
_bid_e   = panel["bid_volume_energy"].fillna(0)
_either  = panel["offer_volume_energy"].notna() | panel["bid_volume_energy"].notna()
panel["gross_bm_volume"] = (_offer_e - _bid_e).where(_either)

# bess_offer_volume: alias for vol_battery (convenience column)
panel["bess_offer_volume"] = panel["vol_battery"]

panel["bess_offer_share"] = (
    panel["bess_offer_volume"] /
    panel["offer_volume_energy"].replace(0, float("nan"))
).round(4)

# bess_bid_volume: NaN where no bid data at all, 0 where BESS simply didn't bid
panel["bess_bid_volume"] = (
    panel["bess_bid_volume"]
    .fillna(0)
    .where(panel["bid_volume_total"].notna())
)

panel["bess_bid_share"] = (
    panel["bess_bid_volume"] /
    panel["bid_volume_energy"].abs().replace(0, float("nan"))
).round(4)

print(f"\nmarginal_tech full breakdown:")
print(panel["marginal_tech"].value_counts(dropna=False).to_string())

panel = panel.sort_values(SP_KEY).reset_index(drop=True)
print(f"\n  Panel shape: {panel.shape}")

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
# A: offer_max_price NaN only when no non-repriced energy offers exist
unpriced_energy_count = (
    offers[off_price_mask].groupby(SP_KEY).size()
    .reset_index(name="_n_unpriced_energy")
)
panel = panel.merge(unpriced_energy_count, on=SP_KEY, how="left")
panel["_n_unpriced_energy"] = panel["_n_unpriced_energy"].fillna(0)
price_nan      = panel["offer_max_price"].isna()
has_unpriced   = panel["_n_unpriced_energy"] > 0
residual_fail  = panel[price_nan & has_unpriced]
assert len(residual_fail) == 0, (
    f"{len(residual_fail)} SPs have non-repriced energy offers but offer_max_price is NaN\n"
    + residual_fail[SP_KEY + ["_n_unpriced_energy"]].to_string()
)
repriced_only = panel[price_nan & (panel["offer_n_energy"] > 0) & ~has_unpriced]
print(f"\n  offer_max_price NaN audit:")
print(f"    Zero-energy-offer SPs    : {(price_nan & (panel['offer_n_energy'] == 0)).sum():,}")
print(f"    Repriced-only SPs        : {len(repriced_only):,}")
print(f"    Extraction failures      : {len(residual_fail):,}")
panel = panel.drop(columns=["_n_unpriced_energy"])

# B: vol_* sum approximately equals offer_volume_energy
vol_cols = [c for c in panel.columns if c.startswith("vol_")]
panel["_vol_sum"] = panel[vol_cols].sum(axis=1)
gap = (panel["offer_volume_energy"] - panel["_vol_sum"]).dropna()
print(f"\n  vol_* coverage audit:")
print(f"    vol_* columns : {vol_cols}")
print(f"    Mean gap vs offer_volume_energy : {gap.mean():.2f} MWh/SP")
print(f"    Max  gap                        : {gap.max():.2f} MWh/SP")
panel = panel.drop(columns=["_vol_sum"])

# ---------------------------------------------------------------------------
# Step 4c — Within-day imputation
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("STEP 4c — WITHIN-DAY IMPUTATION")
print("=" * 55)

IMPUTE_COLS = ["offer_max_price", "marginal_price", "marginal_tech", "marginal_bmu"]
originally_nan = panel[IMPUTE_COLS].isna().any(axis=1)
n_to_impute = originally_nan.sum()
panel["price_imputed"] = originally_nan.astype(int)
print(f"\n  SPs requiring imputation (pre-fill) : {n_to_impute:,}")

panel = panel.sort_values(SP_KEY).reset_index(drop=True)
for col in IMPUTE_COLS:
    panel[col] = (
        panel.groupby("settlementDate")[col]
        .transform(lambda s: s.ffill().bfill())
    )

# Verify imputation completeness
for col in IMPUTE_COLS:
    n_remaining = panel[col].isna().sum()
    assert n_remaining == 0, (
        f"Imputation incomplete: {col} still has {n_remaining:,} NaN values after fill."
    )
actual_flagged = panel["price_imputed"].sum()
assert actual_flagged == n_to_impute, (
    f"price_imputed flag mismatch: expected {n_to_impute:,}, got {actual_flagged:,}"
)
print(f"  SPs imputed (post-fill zero NaN)    : {n_to_impute:,}")

# ---------------------------------------------------------------------------
# Part 2 — Feature engineering
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("PART 2 — FEATURE ENGINEERING")
print("=" * 55)

_GAS_TECHS  = {"GAS", "CCGT", "OCGT"}
_BESS_TECHS = {"BATTERY"}
panel["marginal_tech_unified"] = panel["marginal_tech"].map(
    lambda t: "Gas" if t in _GAS_TECHS else ("BESS" if t in _BESS_TECHS else "Other")
)
panel["gas_marginal"] = (panel["marginal_tech_unified"] == "Gas").astype(int)

print(f"\n  marginal_tech_unified value counts:")
print(panel["marginal_tech_unified"].value_counts(dropna=False).to_string())

# Enforce ISO8601 date strings throughout
panel["settlementDate"] = pd.to_datetime(panel["settlementDate"]).dt.strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Step 5 — Save and summarise
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("STEP 5 — SAVE AND SUMMARY")
print("=" * 55)

panel.to_parquet(PANEL_OUT, index=False)
panel.to_csv(CSV_OUT, index=False)
print(f"\nSaved parquet -> {PANEL_OUT}")
print(f"Saved CSV     -> {CSV_OUT}")

print(f"\nPanel shape: {panel.shape}  ({panel['settlementDate'].nunique():,} dates x 48 SPs)")

print(f"\nmarginal_tech value counts:")
print(panel["marginal_tech"].value_counts(dropna=False).to_string())

print(f"\nmarginal_price stats:")
print(f"  mean : {panel['marginal_price'].mean():.2f}")
print(f"  std  : {panel['marginal_price'].std():.2f}")
print(f"  min  : {panel['marginal_price'].min():.2f}")
print(f"  max  : {panel['marginal_price'].max():.2f}")

vol_cols = [c for c in panel.columns if c.startswith("vol_")]
panel["_vol_sum"] = panel[vol_cols].sum(axis=1)
vol_gap = (panel["offer_volume_energy"] - panel["_vol_sum"]).dropna()
panel = panel.drop(columns=["_vol_sum"])

print()
print("=" * 55)
print("PANEL QUALITY SUMMARY")
print("=" * 55)
print(f"\n  Total SPs in panel                          : {len(panel):,}")
print(f"  offer_max_price  NaN                        : {panel['offer_max_price'].isna().sum():,}")
print(f"  gross_bm_volume  NaN                        : {panel['gross_bm_volume'].isna().sum():,}")
print(f"  bess_bid_volume  NaN                        : {panel['bess_bid_volume'].isna().sum():,}")
print(f"  marginal_price   NaN                        : {panel['marginal_price'].isna().sum():,}")
print(f"  price_imputed    total                      : {panel['price_imputed'].sum():,}")
print(f"  niv              NaN                        : {panel['niv'].isna().sum():,}")
print(f"  niv mean / min / max                        : "
      f"{panel['niv'].mean():.1f} / {panel['niv'].min():.1f} / {panel['niv'].max():.1f}")
print(f"  mean bess_bid_volume (MWh/SP)               : {panel['bess_bid_volume'].mean():.2f}")
print(f"  mean bess_offer_volume (MWh/SP)             : {panel['bess_offer_volume'].mean():.2f}")
print(f"  mean vol_* gap vs offer_volume_energy       : {vol_gap.mean():.2f} MWh/SP")
print(f"  marginal_tech_unified distribution:")
for tech, n in panel["marginal_tech_unified"].value_counts(dropna=False).items():
    pct = 100 * n / len(panel)
    print(f"    {str(tech):<10} {n:>7,}  ({pct:.1f}%)")
print(f"  Total columns                               : {len(panel.columns):,}")
