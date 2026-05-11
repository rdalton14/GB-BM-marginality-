from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

SP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "full_2023_2025" / "bid_offer_stack_2023_2025"
    / "bid_offer_stack_energy_2023_2025_niv_marginal_sp_summary.parquet"
)
REPORT_DIR = PROJECT_ROOT / "reports" / "full_2023_2025"
OUT_HTML = REPORT_DIR / "canonical" / "bmu" / "shap_bmu_price_importance_2023_2025.html"
OUT_CSV  = REPORT_DIR / "tables"    / "shap_bmu_price_importance_2023_2025.csv"

PREDICTOR_COL   = "marginal_bmu_id"
PREDICTOR_LABEL = "BMU ID"

# Proven BMU settings from the three-predictor script; SHAP rows modestly increased.
N_TREES          = 200
MAX_DEPTH        = 24
MIN_SAMPLES_LEAF = 3
SHAP_MAX         = 4_000

SHAP_GUARANTEE_TOP_MDI           = 100
SHAP_ACTIVE_ROWS_PER_TOP_FEATURE = 3

TOP_N      = 40
BEESWARM_N = 25


# ── helpers ────────────────────────────────────────────────────────────────────

def build_fuel_lookup(sp: pd.DataFrame) -> dict[str, str]:
    """Most common fuel type per BMU across all SP rows."""
    return (
        sp.groupby(PREDICTOR_COL)["marginal_fuel_type"]
        .agg(lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) > 0 else "UNKNOWN")
        .to_dict()
    )


def feature_fuel(feature_name: str, lookup: dict[str, str]) -> str:
    for suffix in ("_BID", "_OFFER", "_NEUTRAL"):
        if feature_name.endswith(suffix):
            return lookup.get(feature_name[: -len(suffix)], "UNKNOWN")
    return lookup.get(feature_name, "UNKNOWN")


def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_sided_col(sp: pd.DataFrame, col: str) -> pd.Series:
    return sp[col].fillna("MISSING") + "_" + sp["niv_active_side"].str.upper()


def fit_rf(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=N_TREES,
        n_jobs=-1,
        random_state=0,
        max_features="sqrt",
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )
    rf.fit(X, y)
    return rf


def choose_shap_rows(X: np.ndarray, required_feature_idx: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    n_rows = X.shape[0]
    if n_rows <= SHAP_MAX:
        return np.arange(n_rows)

    selected: list[int] = []
    seen: set[int] = set()
    for feat_idx in required_feature_idx:
        active_rows = np.flatnonzero(X[:, feat_idx] > 0.5)
        if len(active_rows) == 0:
            continue
        take_n = min(SHAP_ACTIVE_ROWS_PER_TOP_FEATURE, len(active_rows))
        for row_idx in rng.choice(active_rows, take_n, replace=False):
            row_int = int(row_idx)
            if row_int not in seen:
                selected.append(row_int)
                seen.add(row_int)
            if len(selected) >= SHAP_MAX:
                return np.array(selected, dtype=int)

    remaining = SHAP_MAX - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n_rows), np.array(selected, dtype=int), assume_unique=False)
        selected.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).astype(int).tolist())

    rng.shuffle(selected)
    return np.array(selected, dtype=int)


def compute_shap(
    rf: RandomForestRegressor, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    top_mdi_idx = np.argsort(rf.feature_importances_)[
        -min(SHAP_GUARANTEE_TOP_MDI, X.shape[1]):
    ]
    idx = choose_shap_rows(X, top_mdi_idx)
    X_s = X[idx]
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_s, check_additivity=False)
    ev = float(np.ravel(explainer.expected_value)[0])
    return sv, X_s, ev


def feature_side(feature: str) -> str:
    if feature.endswith("_BID"):
        return "BID"
    if feature.endswith("_OFFER"):
        return "OFFER"
    if feature.endswith("_NEUTRAL"):
        return "NEUTRAL"
    return "UNKNOWN"


def build_feature_summary(
    sv: np.ndarray,
    X_s: np.ndarray,
    feature_names: list[str],
    mdi: np.ndarray,
    r2: float,
    ev: float,
    fuel_lookup: dict[str, str],
) -> pd.DataFrame:
    active_counts = (X_s > 0.5).sum(axis=0)
    mean_abs = np.abs(sv).mean(axis=0)
    cond_mean  = np.full(len(feature_names), np.nan)
    share_pos  = np.full(len(feature_names), np.nan)
    share_neg  = np.full(len(feature_names), np.nan)

    for j in np.flatnonzero(active_counts > 0):
        active_sv = sv[X_s[:, j] > 0.5, j]
        cond_mean[j] = float(active_sv.mean())
        share_pos[j] = float((active_sv > 0).mean())
        share_neg[j] = float((active_sv < 0).mean())

    out = pd.DataFrame(
        {
            "predictor":    PREDICTOR_LABEL,
            "source_column": PREDICTOR_COL,
            "feature":      feature_names,
            "fuel_type":    [feature_fuel(fn, fuel_lookup) for fn in feature_names],
            "side":         [feature_side(fn) for fn in feature_names],
            "mean_abs_shap": mean_abs,
            "conditional_mean_shap_when_active": cond_mean,
            "share_positive_shap_when_active":   share_pos,
            "share_negative_shap_when_active":   share_neg,
            "active_rows_in_shap_sample":        active_counts,
            "mdi":              mdi,
            "model_train_r2":   r2,
            "shap_expected_value": ev,
        }
    )
    out["overall_shap_rank"] = (
        out["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    )
    out["side_shap_rank"] = (
        out.groupby("side")["mean_abs_shap"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out.sort_values("overall_shap_rank")


def top_side_table(summary: pd.DataFrame, side: str, n: int = 25) -> str:
    block = (
        summary.loc[summary["side"].eq(side) & summary["active_rows_in_shap_sample"].gt(0)]
        .sort_values("mean_abs_shap", ascending=False)
        .head(n)
    )
    show_cols = [
        "side_shap_rank", "overall_shap_rank", "feature", "fuel_type",
        "mean_abs_shap", "conditional_mean_shap_when_active",
        "share_positive_shap_when_active", "share_negative_shap_when_active",
        "active_rows_in_shap_sample", "mdi",
    ]
    if block.empty:
        return "<p class=\"note\">No active features for this side in the SHAP sample.</p>"
    return block[show_cols].round(4).to_html(index=False, border=0, classes="tbl")


# ── charts ─────────────────────────────────────────────────────────────────────

def plot_mean_shap_signed(
    sv: np.ndarray, X_s: np.ndarray, feature_names: list[str],
    fuel_lookup: dict[str, str], ev: float,
) -> str:
    cond_mean    = np.zeros(sv.shape[1])
    active_counts = np.zeros(sv.shape[1], dtype=int)
    for j in range(sv.shape[1]):
        mask = X_s[:, j] > 0.5
        active_counts[j] = int(mask.sum())
        if active_counts[j] > 0:
            cond_mean[j] = sv[mask, j].mean()

    mean_abs = np.abs(sv).mean(axis=0)
    eligible = np.flatnonzero(active_counts > 0)
    top_idx  = eligible[np.argsort(mean_abs[eligible])[-TOP_N:]]

    names    = [
        f"{feature_names[i]}  [{feature_fuel(feature_names[i], fuel_lookup)}]"
        for i in top_idx
    ]
    values   = cond_mean[top_idx]
    colours  = ["#4CAF50" if v >= 0 else "#EF5350" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * TOP_N + 1)))
    ax.barh(names, values, color=colours, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        f"Conditional mean SHAP (£/MWh) when feature=1  [baseline = {ev:.1f} £/MWh]"
    )
    ax.set_title(
        f"BMU ID × NIV side — conditional mean SHAP  (top {TOP_N} by mean |SHAP|)", fontsize=11
    )
    ax.legend(handles=[
        Patch(facecolor="#4CAF50", label="Positive → raises predicted price when at margin"),
        Patch(facecolor="#EF5350", label="Negative → lowers predicted price when at margin"),
    ], fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_mdi_vs_shap(
    mdi: np.ndarray, sv: np.ndarray, feature_names: list[str],
    fuel_lookup: dict[str, str],
) -> str:
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mdi)[-TOP_N:]
    names    = [
        f"{feature_names[i]}  [{feature_fuel(feature_names[i], fuel_lookup)}]"
        for i in top_idx
    ]
    mdi_vals  = mdi[top_idx]
    shap_vals = mean_abs[top_idx]
    mdi_n  = mdi_vals  / mdi_vals.max()
    shap_n = shap_vals / shap_vals.max() if shap_vals.max() > 0 else shap_vals

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * TOP_N + 1)))
    ax.barh(y - 0.18, mdi_n,  0.35, label="MDI (normalised)",         color="#7986CB", edgecolor="white")
    ax.barh(y + 0.18, shap_n, 0.35, label="Mean |SHAP| (normalised)", color="#FF7043", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Normalised importance (each method scaled to max = 1)")
    ax.set_title(f"MDI vs mean |SHAP| — BMU ID × side  (top {TOP_N} by MDI)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_beeswarm(
    X_s: np.ndarray, sv: np.ndarray, feature_names: list[str],
    fuel_lookup: dict[str, str],
) -> str:
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-BEESWARM_N:]

    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * BEESWARM_N + 1.5)))
    for plot_row, feat_idx in enumerate(top_idx):
        sv_col  = sv[:, feat_idx]
        fv_col  = X_s[:, feat_idx]
        jitter  = rng.uniform(-0.3, 0.3, size=len(sv_col))
        colours = np.where(fv_col > 0.5, "#EF5350", "#3F51B5")
        ax.scatter(sv_col, plot_row + jitter, c=colours, alpha=0.25, s=7, linewidths=0)

    labels = [
        f"{feature_names[i]}  [{feature_fuel(feature_names[i], fuel_lookup)}]"
        for i in top_idx
    ]
    ax.set_yticks(range(BEESWARM_N))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (£/MWh)")
    ax.set_title(
        f"SHAP beeswarm — BMU ID × side  (top {BEESWARM_N} by mean |SHAP|)", fontsize=11
    )
    ax.legend(handles=[
        Patch(facecolor="#EF5350", label="Feature = 1  (this BMU×side is at margin)"),
        Patch(facecolor="#3F51B5", label="Feature = 0  (not at margin)"),
    ], fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── HTML ───────────────────────────────────────────────────────────────────────

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #212121; }
  h1   { border-bottom: 3px solid #3F51B5; padding-bottom: 8px; color: #3F51B5; }
  h2   { margin-top: 44px; color: #283593; border-left: 4px solid #7986CB; padding-left: 10px; }
  h3   { color: #37474F; margin-top: 24px; }
  section { margin-bottom: 48px; }
  .tbl { border-collapse: collapse; font-size: 12px; width: 100%; margin-bottom: 16px; }
  .tbl th { background: #3F51B5; color: white; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
  .tbl tr:nth-child(even) td { background: #f5f5f5; }
  img  { display: block; margin: 12px 0; border-radius: 6px;
         box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  p.note { font-size: 12px; color: #757575; font-style: italic; }
  .stat { display:inline-block; background:#E8EAF6; border-radius:8px;
          padding:8px 18px; margin:4px; font-size:14px; }
  .stat strong { display:block; font-size:22px; color:#3F51B5; }
</style>
"""


def make_html(
    sections: list[str],
    setup_html: str,
    n_rows: int,
    n_sps: int,
    sp: pd.DataFrame,
) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>BMU SHAP Price Importance — 2023–2025</title>
{CSS}
</head><body>
<h1>BMU ID SHAP: System Price ~ Marginal BMU × NIV Side — 2023–2025</h1>
<p>TreeSHAP values from a single Random Forest regressor fitted on BMU ID × NIV side
   dummies. This is the primary analytical level: BMU ID is the most granular available
   identity; fuel type is shown as an annotation alongside each BMU in all tables and charts.
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>
{setup_html}
{"".join(sections)}
</body></html>"""


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    y = sp["systemPrice"].values
    n_rows = len(sp)
    n_sps  = sp[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    print(f"  {n_rows:,} rows | {n_sps:,} unique SPs")

    fuel_lookup = build_fuel_lookup(sp)

    sided_col = f"{PREDICTOR_COL}_sided"
    sp[sided_col] = make_sided_col(sp, PREDICTOR_COL)
    n_unique = sp[sided_col].nunique()
    print(
        f"\nBMU ID sided ({n_unique} BMU×side combinations, "
        f"trees={N_TREES}, max_depth={MAX_DEPTH}, "
        f"min_leaf={MIN_SAMPLES_LEAF}, SHAP rows={SHAP_MAX:,}) ..."
    )

    dummies = pd.get_dummies(sp[sided_col], prefix="", prefix_sep="")
    feature_names = dummies.columns.tolist()
    X = dummies.values.astype(float)

    print("  Fitting RF ...")
    rf = fit_rf(X, y)
    r2 = rf.score(X, y)
    print(f"  Train R² = {r2:.4f}")

    print(f"  Computing TreeSHAP (n_features={len(feature_names)}, rows≤{SHAP_MAX:,}) ...")
    sv, X_s, ev = compute_shap(rf, X)

    top5 = (
        pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
        .nlargest(5).index.tolist()
    )
    print(f"  Expected value = {ev:.2f} £/MWh")
    print(f"  Top 5 by |mean SHAP|: {top5}")

    summary = build_feature_summary(sv, X_s, feature_names, rf.feature_importances_, r2, ev, fuel_lookup)

    b64_signed = plot_mean_shap_signed(sv, X_s, feature_names, fuel_lookup, ev)
    b64_comp   = plot_mdi_vs_shap(rf.feature_importances_, sv, feature_names, fuel_lookup)
    b64_bee    = plot_beeswarm(X_s, sv, feature_names, fuel_lookup)

    offer_table = top_side_table(summary, "OFFER")
    bid_table   = top_side_table(summary, "BID")

    stat_html = (
        f'<div>'
        f'<span class="stat"><strong>{n_unique}</strong>BMU×side combinations</span>'
        f'<span class="stat"><strong>{N_TREES}</strong>trees</span>'
        f'<span class="stat"><strong>{r2:.3f}</strong>train R²</span>'
        f'<span class="stat"><strong>{ev:.1f} £/MWh</strong>SHAP baseline</span>'
        f'<span class="stat"><strong>{len(X_s):,}</strong>SHAP rows</span>'
        f'</div>'
    )

    setup_html = f"""
<section>
  <h2>Setup</h2>
  <table class="tbl">
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Rows</td><td>{n_rows:,} marginal candidate rows</td></tr>
    <tr><td>Unique SPs</td><td>{n_sps:,}</td></tr>
    <tr><td>Date range</td><td>{sp['settlementDate'].dt.date.min()} to {sp['settlementDate'].dt.date.max()}</td></tr>
    <tr><td>Target</td><td>systemPrice (£/MWh) — regression</td></tr>
    <tr><td>Predictor</td><td>marginal_bmu_id × NIV side (BMU ID, most granular identity level)</td></tr>
    <tr><td>BMU×side combinations</td><td>{n_unique}</td></tr>
    <tr><td>RF hyperparameters</td><td>{N_TREES} trees, max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}, max_features=sqrt, random_state=0</td></tr>
    <tr><td>SHAP method</td><td>TreeSHAP exact (tree_path_dependent)</td></tr>
    <tr><td>SHAP rows</td><td>{len(X_s):,} (budget {SHAP_MAX:,}, seed=0)</td></tr>
    <tr><td>Fuel type</td><td>Annotated from marginal_fuel_type modal value per BMU — shown alongside each BMU in all tables and charts. Not used in model fitting.</td></tr>
    <tr><td>Encoding</td><td>BMU ID suffixed _OFFER or _BID, then one-hot encoded</td></tr>
  </table>
  <p class="note">SHAP values are model associations, not causal estimates. When a BMU×side
     dummy is active, the model's predicted price shifts above or below the baseline. Fuel type
     is an annotation derived from the data; it does not enter the RF model.</p>
</section>"""

    section = f"""
<section>
  <h2>BMU ID × NIV side</h2>
  {stat_html}

  <h3>Conditional mean SHAP — directional impact on system price</h3>
  <img src="data:image/png;base64,{b64_signed}" style="width:100%;max-width:950px;">
  <p class="note">Conditional mean SHAP: average SHAP value across SPs where that BMU×side
     combination is at the margin (feature=1). Green bars push predicted price above the
     baseline ({ev:.1f} £/MWh); red bars suppress it. Fuel type shown in brackets.
     Ranked by mean |SHAP| (unconditional).</p>

  <h3>MDI vs mean |SHAP| — importance method comparison</h3>
  <img src="data:image/png;base64,{b64_comp}" style="width:100%;max-width:950px;">
  <p class="note">Both metrics normalised to max=1 for visual comparison (top {TOP_N} by MDI).
     Close agreement validates the MDI ranking. Divergence highlights split-frequency inflation
     (high MDI, low SHAP) or infrequent but high-impact features (low MDI, high SHAP).</p>

  <h3>SHAP beeswarm — distribution of per-SP effects (top {BEESWARM_N} BMU×side features)</h3>
  <img src="data:image/png;base64,{b64_bee}" style="width:100%;max-width:950px;">
  <p class="note">Each point is one settlement period ({len(X_s):,} sampled). Red = feature
     active (this BMU×side is at margin); blue = inactive. Horizontal spread of red dots shows
     price impact variance across SPs where that BMU is marginal.</p>

  <h3>Top OFFER-side BMUs by mean |SHAP|</h3>
  {offer_table}

  <h3>Top BID-side BMUs by mean |SHAP|</h3>
  {bid_table}
</section>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(
        make_html([section], setup_html, n_rows, n_sps, sp), encoding="utf-8"
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    print(f"\nDone -> {OUT_HTML}")
    print(f"Done -> {OUT_CSV}")


if __name__ == "__main__":
    main()
