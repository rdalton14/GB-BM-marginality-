# Supervisor Review Guide

This repository is prepared as a reviewable snapshot of the dissertation analysis.
It includes the scripts, generated datasets, and HTML reports needed to inspect the
logic and outputs.

## Start Here

- `README.md` gives the repository structure.
- `CLAUDE.md` contains the project-specific analytical context and terminology.
- `requirements.txt` lists the Python dependencies used by the scripts.

## Main Analysis Window

- Study period: `2023-01-01` to `2025-12-31`
- Exploratory comparison window: `2026-01-01` to `2026-03-31`

## Key Scripts

- `scripts/ispstack/full/collect_ispstack_2023_2025.py`
- `scripts/ispstack/full/collect_disbsad_2023_2025.py`
- `scripts/ispstack/full/build_bid_offer_stack_energy_2023_2025.py`
- `scripts/ispstack/full/build_marginal_action_2023_2025.py`
- `scripts/ispstack/full/build_master_panel_2023_2025_energy_offer_stack.py`
- `scripts/diagnostics/eda_2023_2025_sp_summary.py`
- `scripts/diagnostics/rf_price_importance_2023_2025_sided.py`
- `scripts/diagnostics/bid_side_price_driver_cost_to_so_diagnostic.py`

## Main HTML Reports

- `reports/full_2023_2025/canonical/eda_2023_2025_sp_summary.html`
- `reports/full_2023_2025/canonical/rf_price_importance_2023_2025_sided.html`
- `reports/full_2023_2025/canonical/shap_price_importance_2023_2025_sided_updated.html`
- `reports/full_2023_2025/canonical/bid_side_price_driver_cost_to_so_diagnostic.html`

## Main Data Outputs

- `data/processed/full_2023_2025/bid_offer_stack_2023_2025/`
- `data/processed/q1_2026/bid_offer_stack_q1_2026/`
- `data/raw/reference/`

Large binary datasets are tracked using Git LFS. After cloning, run:

```powershell
git lfs pull
```

## Not Included

Local environments, machine-specific assistant settings, and secrets are intentionally
excluded from version control:

- `.venv/`
- `.claude/`
- `config/secret_key`

