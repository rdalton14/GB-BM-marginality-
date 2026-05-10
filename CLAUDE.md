# Dissertation: BESS Participation in the GB Balancing Mechanism

## Project overview
MSc Data Science dissertation examining marginal price dynamics in the GB Balancing
Mechanism (BM), with particular focus on how Battery Energy Storage Systems (BESS)
compete at the margin alongside CCGT, hydro, pumped storage, and interconnectors.
The central question is which technologies set the clearing price and what that implies
for system cost and price predictability.

## Study window
- Primary: 2023-01-01 to 2025-12-31
- Subsample / exploration: Q1 2026 (2026-01-01 to 2026-03-31)

## Key terminology
- SP: Settlement Period (48 per day, 30 mins each)
- BMU: Balancing Mechanism Unit (the unit of participation)
- ISPSTACK: In-Settlement Price Stack — accepted BM actions with finalPrice and soFlag
- DISBSAD: Balancing Services Adjustment Data — numeric-ID actions (interconnectors etc.)
- soFlag: system vs energy acceptance reason (True = system, False = energy)
- NIV: Net Imbalance Volume — positive = short system (offer side active), negative = long
- Marginal technology: the BMU(s) whose accepted price equals the top price on the active NIV side
- Co-marginal: multiple BMUs tied at the top price in the same SP (all flagged, not one winner)
- family_id: plant-family prefix stripped of trailing digits; mixed-fuel families get a fuel-type suffix
- Sided label: category label suffixed with _OFFER or _BID to encode NIV direction (e.g. BATTERY_OFFER)

## Marginal identification logic
- Energy-only filter: soFlag=False, cadlFlag=False, storProviderFlag=False, non-zero volume
- Numeric IDs kept only if matched to DISBSAD with service="energy"
- NIV>0 (short): marginal = all BMUs at the maximum finalPrice on the offer side
- NIV<0 (long): marginal = all BMUs at the maximum finalPrice on the bid side (by bid_rank_price_desc)
- Ties preserved — each co-marginal BMU appears as a separate row in the SP summary
- n_tied_marginal_candidates: count of distinct BMUs at the top price per SP

## Fuel type taxonomy
Full label set used in outputs (uppercase unless a named interconnector):
BATTERY, CCGT, OCGT, GAS, PS, NPSHYD, WIND, SOLAR, BIOMASS, COAL, DIESEL,
LOAD RESPONSE, OTHER — plus named interconnectors:
IFA, IFA2, BritNed, NEMO, Eleclink, Viking, Moyle, EastWest, NSL, Greenlink

## Pipeline stages (2023-25; Q1 2026 mirrors with q1_2026 path substitutions)

| Stage | Script | Output |
|-------|--------|--------|
| 1. Fetch DISBSAD | `scripts/ispstack/full/collect_disbsad_2023_2025.py` | `data/raw/ispstack/disbsad_2023_2025/disbsad_2023_2025.parquet` |
| 2. Build stack | `scripts/ispstack/full/build_bid_offer_stack_energy_2023_2025.py` | `...bid_offer_stack_energy_2023_2025_long.parquet` |
| 3. BMU map | `scripts/diagnostics/build_bmu_register_and_map_2023_2025.py` | `...bmu_map.csv` + `bmu_register_elexon_current.csv` |
| 4. BMU register | `scripts/diagnostics/build_bmu_register_2023_2025.py` | `data/raw/reference/bmu_register_2023_2025.csv` |
| 5. NIV marginal | `scripts/diagnostics/build_niv_marginal_stack_energy_2023_2025.py` | `..._niv_marginal.parquet` + `..._niv_marginal_sp_summary.parquet` |
| 6. EDA | `scripts/diagnostics/eda_2023_2025_sp_summary.py` | `reports/eda_2023_2025_sp_summary.html` |
| 7. RF importance | `scripts/diagnostics/rf_price_importance_2023_2025.py` | `reports/rf_price_importance_2023_2025.html` |
| 7b. RF sided | `scripts/diagnostics/rf_price_importance_2023_2025_sided.py` | `reports/rf_price_importance_2023_2025_sided.html` |

## Key analytical findings (exploratory)
- NIV side is the dominant source of price variance: adding _OFFER/_BID to labels lifts
  generation-type R² from ~0.06 to ~0.34 (2023-25) and ~0.40 (Q1 2026)
- Family ID explains ~5x more price variance than generation type alone (unsided)
- 2023-25 dominant marginals: CCGT_OFFER (upward), BATTERY_BID (downward), hydro/PS offer-side
- Q1 2026 dominant marginals: BATTERY_OFFER and BATTERY_BID -- BESS now sets price on both sides
- Interconnectors (IFA, BritNed, NEMO, Viking, Eleclink, IFA2) account for 2,356 marginal
  candidates in 2023-25, resolved via DISBSAD + BMU ID prefix convention

## Data conventions
- Raw ISPSTACK: one Parquet per day in `data/raw/ispstack/`
- Processed stacks: single Parquet per study window in `data/processed/`
- Settlement periods are integers 1-48
- Dates as pd.Timestamp or "YYYY-MM-DD" strings
- Never modify raw data files — all transformations in processing scripts
- BMU register is the authoritative fuel type source; interconnector names override BMU map labels

## Python environment
- Python 3.14
- Key libraries: pandas, requests, pyarrow, scikit-learn, statsmodels, matplotlib
