from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

SP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "full_2023_2025" / "bid_offer_stack_2023_2025"
    / "bid_offer_stack_energy_2023_2025_niv_marginal_sp_summary.parquet"
)
OUT_HTML = (
    PROJECT_ROOT
    / "reports" / "full_2023_2025" / "archive" / "superseded"
    / "rf_price_importance_2023_2025.html"
)

N_RUNS   = 10
N_TREES  = 300
TOP_N    = 25   # max categories shown in chart for high-cardinality columns

PREDICTORS = [
    ("marginal_fuel_type", "Generation type"),
    ("marginal_family_id", "Family ID"),
    ("marginal_bmu_id",    "BMU ID"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def run_rf(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame, float]:
    """Run N_RUNS forests, return DataFrame of importances (rows=features, cols=runs) and mean R²."""
    imp_runs = np.zeros((len(feature_names), N_RUNS))
    r2_runs  = np.zeros(N_RUNS)
    for i in range(N_RUNS):
        rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=i, max_features="sqrt")
        rf.fit(X, y)
        imp_runs[:, i] = rf.feature_importances_
        r2_runs[i]     = rf.score(X, y)
    imp_df = pd.DataFrame(imp_runs, index=feature_names,
                          columns=[f"run_{i}" for i in range(N_RUNS)])
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df["std"]  = imp_df.std(axis=1)
    return imp_df, float(r2_runs.mean())


def plot_importance(imp_df: pd.DataFrame, label: str, n_top: int = TOP_N) -> str:
    top = imp_df.nlargest(n_top, "mean").sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(top) + 1)))
    ax.barh(top.index, top["mean"], xerr=top["std"],
            color="#3F51B5", edgecolor="white", error_kw=dict(ecolor="#B0BEC5", capsize=3))
    ax.set_xlabel("Mean MDI importance ± 1 SD across runs")
    ax.set_title(f"System price ~ {label}  (top {len(top)})", fontsize=11)
    ax.set_xlim(0, (top["mean"] + top["std"]).max() * 1.18)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_price_by_category(sp: pd.DataFrame, col: str, label: str, n_top: int = TOP_N) -> str:
    medians = sp.groupby(col)["systemPrice"].median().sort_values(ascending=False)
    top_cats = medians.head(n_top).index.tolist()
    data  = [sp.loc[sp[col] == c, "systemPrice"].dropna().values for c in top_cats]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(top_cats) + 2), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#3F51B5")
    ax.set_xticklabels(top_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("System price (£/MWh)")
    ax.set_title(f"System price by {label}  (top {len(top_cats)} by median, IQR)", fontsize=11)
    ax.axhline(sp["systemPrice"].median(), color="#EF5350", linestyle="--",
               linewidth=1, label=f"Overall median {sp['systemPrice'].median():.1f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def desc_table(sp: pd.DataFrame, col: str) -> str:
    tbl = (
        sp.groupby(col)["systemPrice"]
        .agg(n="count", mean="mean", median="median",
             p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9))
        .round(2)
        .sort_values("median", ascending=False)
        .head(TOP_N)
    )
    tbl.columns = ["N rows", "Mean £/MWh", "Median £/MWh", "P10 £/MWh", "P90 £/MWh"]
    return tbl.to_html(border=0, classes="tbl")


CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1050px; margin: 40px auto; padding: 0 20px; color: #212121; }
  h1   { border-bottom: 3px solid #3F51B5; padding-bottom: 8px; color: #3F51B5; }
  h2   { margin-top: 44px; color: #283593; border-left: 4px solid #7986CB; padding-left: 10px; }
  h3   { color: #37474F; margin-top: 24px; }
  section { margin-bottom: 48px; }
  .tbl { border-collapse: collapse; font-size: 13px; width: 100%; margin-bottom: 16px; }
  .tbl th { background: #3F51B5; color: white; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
  .tbl tr:nth-child(even) td { background: #f5f5f5; }
  img  { display: block; margin: 12px 0; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  p.note { font-size: 12px; color: #757575; font-style: italic; }
  .stat { display:inline-block; background:#E8EAF6; border-radius:8px;
          padding:8px 18px; margin:4px; font-size:14px; }
  .stat strong { display:block; font-size:22px; color:#3F51B5; }
</style>
"""


def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    y = sp["systemPrice"].values
    print(f"  {len(sp):,} rows  |  {N_RUNS} runs x {N_TREES} trees per model")

    setup_html = f"""
<section>
  <h2>Setup</h2>
  <table class="tbl">
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Rows</td><td>{len(sp):,} marginal candidate rows</td></tr>
    <tr><td>Unique SPs</td><td>{sp[['settlementDate','settlementPeriod']].drop_duplicates().shape[0]:,}</td></tr>
    <tr><td>Date range</td><td>{sp['settlementDate'].dt.date.min()} to {sp['settlementDate'].dt.date.max()}</td></tr>
    <tr><td>Target</td><td>systemPrice (£/MWh) — regression</td></tr>
    <tr><td>Predictors</td><td>One-hot encoded categories for each column tested separately</td></tr>
    <tr><td>Training split</td><td>100% — MDI feature importance only, no held-out evaluation</td></tr>
    <tr><td>Runs per model</td><td>{N_RUNS} (different random seeds, importances averaged)</td></tr>
    <tr><td>Trees per run</td><td>{N_TREES}</td></tr>
    <tr><td>Note on co-marginals</td><td>SPs with multiple tied marginals contribute one row per candidate,
        all sharing the same systemPrice. This inflates representation of high-tie SPs slightly.</td></tr>
  </table>
</section>"""

    sections_html = ""

    for col, label in PREDICTORS:
        print(f"\n{label} ({sp[col].nunique()} unique values) ...")
        dummies = pd.get_dummies(sp[col].fillna("MISSING"), prefix="", prefix_sep="")
        feature_names = dummies.columns.tolist()
        X = dummies.values.astype(float)

        imp_df, mean_r2 = run_rf(X, y, feature_names)
        print(f"  Mean train R² = {mean_r2:.4f}")
        print(f"  Top 3: {imp_df.nlargest(3,'mean').index.tolist()}")

        b64_imp  = plot_importance(imp_df, label)
        b64_box  = plot_price_by_category(sp, col, label)

        stat_html = (
            f'<div><span class="stat"><strong>{sp[col].nunique()}</strong>unique values</span>'
            f'<span class="stat"><strong>{len(feature_names)}</strong>one-hot features</span>'
            f'<span class="stat"><strong>{mean_r2:.3f}</strong>mean train R²</span></div>'
        )

        sections_html += f"""
<section>
  <h2>Predictor: {label}</h2>
  {stat_html}

  <h3>System price distribution by {label} (top {TOP_N} by median)</h3>
  <img src="data:image/png;base64,{b64_box}" style="width:100%;max-width:950px;">

  <h3>Descriptive stats (top {TOP_N} by median price)</h3>
  {desc_table(sp, col)}

  <h3>RF feature importance — which {label} values drive system price?</h3>
  <img src="data:image/png;base64,{b64_imp}" style="width:100%;max-width:850px;">
  <p class="note">Error bars = 1 SD across {N_RUNS} runs. Higher MDI = that category drives
     more variance in system price across the forest's splits.</p>
</section>
"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>RF Price Importance — 2023–2025</title>
{CSS}
</head><body>
<h1>RF Feature Importance: System Price ~ Marginal Identity — 2023–2025</h1>
<p>Which generation type, plant family, or BMU, when at the margin, is most associated
   with variation in system price? Each column is one-hot encoded and regressed against
   <em>systemPrice</em> in a separate Random Forest.
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>
{setup_html}
{sections_html}
</body></html>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nSaved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
