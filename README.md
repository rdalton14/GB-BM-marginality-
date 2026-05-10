# GB Balancing Mechanism Dissertation

This project builds GB Balancing Mechanism datasets for dissertation analysis.

## Structure

- `config/` - local configuration.
- `data/raw/` - untouched source pulls and reference files.
- `data/raw_test/` - small-window test pulls.
- `data/interim/` - temporary working datasets.
- `data/processed/` - generated analysis-ready panels and processed fundamentals.
- `data/diagnostics/` - coverage, missingness, and audit outputs.
- `scripts/` - runnable entry points for pulling, processing, diagnostics, and maintenance.
- `src/` - reusable package code as the project is modularised.
- `outputs/` - figures, tables, reports, and logs.
- `archive/` - legacy or experimental files preserved from earlier work.

## Key Outputs

- `data/processed/q1_2026/master_panel_q1_2026.csv`
- `data/processed/q1_2026/master_panel_q1_2026.parquet`
- `data/processed/full_2023_2025/master_panel_2023_2025.csv`
- `data/processed/full_2023_2025/master_panel_2023_2025.parquet`
- `outputs/reports/eda_report_q1_2026.html`

## Current Runnable Scripts

- `scripts/fundamentals/test/`
- `scripts/fundamentals/q1_2026/`
- `scripts/ispstack/q1_2026/`
- `scripts/ispstack/full/`
- `scripts/diagnostics/`
- `scripts/maintenance/`
