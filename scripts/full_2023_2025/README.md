# Full 2023-2025 Script Index

This is the intended home for the 2023-2025 pipeline, but the active diagnostics have not yet been physically moved because several scripts import each other from `scripts.diagnostics`.

Current active scripts:

- Build/check:
  - `scripts/diagnostics/build_niv_marginal_stack_energy_2023_2025.py`
  - `scripts/diagnostics/check_bid_marginal_selection_logic_2023_2025.py`
- Reports:
  - `scripts/diagnostics/eda_2023_2025_sp_summary.py`
  - `scripts/diagnostics/bid_side_price_driver_cost_to_so_diagnostic.py`
  - `scripts/diagnostics/shap_price_importance_interpretation_summary_2023_2025.py`
- Modelling:
  - `scripts/diagnostics/shap_price_importance_2023_2025_sided_updated.py`
  - `scripts/diagnostics/shap_price_importance_year_stability_2023_2025.py`
  - `scripts/diagnostics/shap_price_importance_regime_stability_2023_2025.py`

Next migration stage: move these files here and update imports from `scripts.diagnostics` to `scripts.full_2023_2025`.

