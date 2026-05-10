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
    / "reports" / "full_2023_2025" / "canonical"
    / "rf_price_importance_2023_2025_sided.html"
)

N_RUNS = 3
N_TREES = {
    "marginal_fuel_type": 300,
    "marginal_family_id": 200,
    "marginal_bmu_id": 200,
}
MAX_DEPTH = {
    "marginal_fuel_type": 16,
    "marginal_family_id": 24,
    "marginal_bmu_id": 24,
}
MIN_SAMPLES_LEAF = {
    "marginal_fuel_type": 5,
    "marginal_family_id": 3,
    "marginal_bmu_id": 3,
}
TOP_N   = 30

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


def make_sided_col(sp: pd.DataFrame, col: str) -> pd.Series:
    """Append _OFFER or _BID to each label based on niv_active_side."""
    return sp[col].fillna("MISSING") + "_" + sp["niv_active_side"].str.upper()


def run_rf(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_trees: int,
    max_depth: int,
    min_samples_leaf: int,
) -> tuple[pd.DataFrame, float]:
    imp_runs = np.zeros((len(feature_names), N_RUNS))
    r2_runs  = np.zeros(N_RUNS)
    for i in range(N_RUNS):
        rf = RandomForestRegressor(
            n_estimators=n_trees,
            n_jobs=-1,
            random_state=i,
            max_features="sqrt",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
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
    colours = ["#EF5350" if f.endswith("_BID") else "#3F51B5" for f in top.index]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(top) + 1)))
    ax.barh(top.index, top["mean"], xerr=top["std"], color=colours,
            edgecolor="white", error_kw=dict(ecolor="#B0BEC5", capsize=3))
    ax.set_xlabel("Mean MDI importance ± 1 SD across runs")
    ax.set_title(f"System price ~ {label} × side  (top {len(top)})", fontsize=11)
    ax.set_xlim(0, (top["mean"] + top["std"]).max() * 1.18)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#3F51B5", label="Offer-side marginal"),
        Patch(facecolor="#EF5350", label="Bid-side marginal"),
    ], fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_directional(sp: pd.DataFrame, sided_col: str, imp_df: pd.DataFrame, label: str, n_top: int = TOP_N) -> str:
    """Diverging bar chart: median systemPrice deviation from overall median for top MDI features."""
    from matplotlib.patches import Patch
    overall_median = sp["systemPrice"].median()
    top_features = imp_df.nlargest(n_top, "mean").index.tolist()

    rows = []
    for feat in top_features:
        subset = sp.loc[sp[sided_col] == feat, "systemPrice"].dropna()
        if len(subset) > 0:
            rows.append({"feature": feat, "deviation": subset.median() - overall_median, "n": len(subset)})
    dev_df = pd.DataFrame(rows).sort_values("deviation", ascending=True)

    colours = ["#4CAF50" if d >= 0 else "#EF5350" for d in dev_df["deviation"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(dev_df) + 1)))
    ax.barh(dev_df["feature"], dev_df["deviation"], color=colours, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Median system price deviation from overall median ({overall_median:.1f} £/MWh)")
    ax.set_title(f"Directional effect at margin: {label} × side  (top {n_top} by MDI)", fontsize=11)
    ax.legend(handles=[
        Patch(facecolor="#4CAF50", label="Above median → upward pressure on price"),
        Patch(facecolor="#EF5350", label="Below median → downward pressure on price"),
    ], fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_price_by_sided(sp: pd.DataFrame, sided_col: str, label: str, n_top: int = TOP_N) -> str:
    medians = sp.groupby(sided_col)["systemPrice"].median().sort_values(ascending=False)
    top_cats = medians.head(n_top).index.tolist()
    data = [sp.loc[sp[sided_col] == c, "systemPrice"].dropna().values for c in top_cats]
    colours = ["#EF5350" if c.endswith("_BID") else "#3F51B5" for c in top_cats]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(top_cats) + 2), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
    ax.set_xticklabels(top_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("System price (£/MWh)")
    ax.set_title(f"System price by {label} × side  (top {len(top_cats)} by median, IQR)", fontsize=11)
    ax.axhline(sp["systemPrice"].median(), color="grey", linestyle="--",
               linewidth=1, label=f"Overall median {sp['systemPrice'].median():.1f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def desc_table(sp: pd.DataFrame, sided_col: str, imp_df: pd.DataFrame) -> str:
    tbl = (
        sp.groupby(sided_col)["systemPrice"]
        .agg(n="count", median="median",
             p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9))
    )
    tbl.columns = ["N rows", "Median £/MWh", "P10 £/MWh", "P90 £/MWh"]
    tbl.columns = ["n", "median", "p10", "p90"]
    rf = imp_df[["mean", "std"]].copy()
    rf["RF rank"] = rf["mean"].rank(method="first", ascending=False).astype(int)
    tbl = tbl.join(rf, how="left").sort_values("RF rank").head(TOP_N)
    tbl = tbl.reset_index()
    tbl = tbl[[sided_col, "RF rank", "mean", "std", "n", "median", "p10", "p90"]]
    tbl.columns = [
        sided_col,
        "RF rank",
        "Mean MDI",
        "MDI SD",
        "N rows",
        "Median system price GBP/MWh",
        "P10 system price GBP/MWh",
        "P90 system price GBP/MWh",
    ]
    return tbl.round(4).to_html(index=False, border=0, classes="tbl")

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
    tree_desc = " | ".join(f"{label}: {N_TREES[col]}" for col, label in PREDICTORS)
    print(f"  {len(sp):,} rows  |  {N_RUNS} runs; trees per run = {tree_desc}")

    setup_html = f"""
<section>
  <h2>Setup</h2>
  <table class="tbl">
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Rows</td><td>{len(sp):,} marginal candidate rows</td></tr>
    <tr><td>Unique SPs</td><td>{sp[['settlementDate','settlementPeriod']].drop_duplicates().shape[0]:,}</td></tr>
    <tr><td>Date range</td><td>{sp['settlementDate'].dt.date.min()} to {sp['settlementDate'].dt.date.max()}</td></tr>
    <tr><td>Target</td><td>systemPrice (£/MWh) — regression</td></tr>
    <tr><td>Encoding</td><td>Label suffixed with _OFFER or _BID before one-hot encoding, e.g. BATTERY_OFFER, T_DINO-1_BID.
        Units that only appear on one side receive only that suffix — no spurious empty dummies.</td></tr>
    <tr><td>Training split</td><td>100% — MDI feature importance only</td></tr>
    <tr><td>Runs per model</td><td>{N_RUNS} (different random seeds, importances averaged)</td></tr>
    <tr><td>Trees per run</td><td>{tree_desc}</td></tr>
    <tr><td>RF regularisation</td><td>fuel depth 16 leaf 5; family/BMU depth 24 leaf 3</td></tr>
  </table>
</section>"""

    sections_html = ""

    for col, label in PREDICTORS:
        sided_col = f"{col}_sided"
        sp[sided_col] = make_sided_col(sp, col)
        n_unique = sp[sided_col].nunique()
        print(f"\n{label} sided ({n_unique} unique label×side combinations) ...")

        dummies = pd.get_dummies(sp[sided_col], prefix="", prefix_sep="")
        feature_names = dummies.columns.tolist()
        X = dummies.values.astype(float)

        imp_df, mean_r2 = run_rf(
            X,
            y,
            feature_names,
            N_TREES[col],
            MAX_DEPTH[col],
            MIN_SAMPLES_LEAF[col],
        )
        print(f"  Mean train R² = {mean_r2:.4f}")
        print(f"  Top 5: {imp_df.nlargest(5, 'mean').index.tolist()}")

        b64_imp = plot_importance(imp_df, label)
        b64_dir = plot_directional(sp, sided_col, imp_df, label)
        b64_box = plot_price_by_sided(sp, sided_col, label)

        stat_html = (
            f'<div>'
            f'<span class="stat"><strong>{n_unique}</strong>label×side combinations</span>'
            f'<span class="stat"><strong>{len(feature_names)}</strong>one-hot features</span>'
            f'<span class="stat"><strong>{mean_r2:.3f}</strong>mean train R²</span>'
            f'</div>'
        )

        sections_html += f"""
<section>
  <h2>Predictor: {label} × NIV side</h2>
  {stat_html}

  <h3>System price by {label} × side (top {TOP_N} by median) — blue=offer, red=bid</h3>
  <img src="data:image/png;base64,{b64_box}" style="width:100%;max-width:950px;">

  <h3>Descriptive stats and RF importance (top {TOP_N} by MDI)</h3>
  {desc_table(sp, sided_col, imp_df)}

  <h3>RF feature importance — blue=offer, red=bid</h3>
  <img src="data:image/png;base64,{b64_imp}" style="width:100%;max-width:850px;">
  <p class="note">Each label encodes both identity and NIV active side. A unit appearing only on one side
     has a single dummy; one appearing on both sides has two. Error bars = 1 SD across {N_RUNS} runs.</p>
  <h3>Directional effect — green=above median (upward), red=below median (downward)</h3>
  <img src="data:image/png;base64,{b64_dir}" style="width:100%;max-width:850px;">
  <p class="note">Bars show median system price for that label×side combination minus the overall median
     ({'{:.1f}'.format(sp['systemPrice'].median())} £/MWh). Green bars push price up; red bars push price down.
     Features ordered by direction not MDI — compare with the importance chart above to see which
     high-MDI features are upward vs downward drivers.</p>
</section>
"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>RF Price Importance (Sided) — 2023–2025</title>
{CSS}
</head><body>
<h1>RF Feature Importance: System Price ~ Marginal Identity × NIV Side — 2023–2025</h1>
<p>Labels suffixed with _OFFER or _BID before one-hot encoding, so the RF learns
   from each (identity, side) combination independently.
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>
{setup_html}
{sections_html}
</body></html>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nSaved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
