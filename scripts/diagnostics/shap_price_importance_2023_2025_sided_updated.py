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
OUT_HTML = REPORT_DIR / "canonical" / "shap_price_importance_2023_2025_sided_updated.html"
OUT_CSV = REPORT_DIR / "tables" / "shap_price_importance_2023_2025_sided_updated.csv"

TOP_N      = 30
BEESWARM_N = 20

# Per-predictor settings: family/BMU have 749/1043 features — fewer trees keeps both RF
# fitting and TreeSHAP traversal fast; 100 trees gives stable SHAP rankings.
N_TREES = {
    "marginal_fuel_type": 300,
    "marginal_family_id": 200,
    "marginal_bmu_id":    200,
}
MAX_DEPTH = {
    "marginal_fuel_type": 16,
    "marginal_family_id": 24,
    "marginal_bmu_id":    24,
}
MIN_SAMPLES_LEAF = {
    "marginal_fuel_type": 5,
    "marginal_family_id": 3,
    "marginal_bmu_id":    3,
}
SHAP_MAX = {
    "marginal_fuel_type": 10_000,
    "marginal_family_id":  5_000,
    "marginal_bmu_id":     2_500,
}
SHAP_GUARANTEE_TOP_MDI = 100
SHAP_ACTIVE_ROWS_PER_TOP_FEATURE = 3

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
    return sp[col].fillna("MISSING") + "_" + sp["niv_active_side"].str.upper()


def fit_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_trees: int,
    max_depth: int,
    min_samples_leaf: int,
) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_trees,
        n_jobs=-1,
        random_state=0,
        max_features="sqrt",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    rf.fit(X, y)
    return rf


def choose_shap_rows(
    X: np.ndarray,
    shap_max: int,
    required_feature_idx: np.ndarray,
) -> np.ndarray:
    """Random SHAP sample, seeded with active rows for important sparse dummies."""
    rng = np.random.default_rng(0)
    n_rows = X.shape[0]
    if n_rows <= shap_max:
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
            if len(selected) >= shap_max:
                return np.array(selected, dtype=int)

    remaining = shap_max - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n_rows), np.array(selected, dtype=int), assume_unique=False)
        selected.extend(rng.choice(pool, remaining, replace=False).astype(int).tolist())

    rng.shuffle(selected)
    return np.array(selected, dtype=int)


def compute_shap(
    rf: RandomForestRegressor,
    X: np.ndarray,
    shap_max: int,
    required_feature_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (shap_values, X_shap_subset, expected_value)."""
    idx = choose_shap_rows(X, shap_max, required_feature_idx)
    X_s = X[idx]
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_s, check_additivity=False)
    ev = float(np.ravel(explainer.expected_value)[0])
    return sv, X_s, ev


# ── charts ────────────────────────────────────────────────────────────────────

def plot_mean_shap_signed(
    sv: np.ndarray, X_s: np.ndarray, feature_names: list[str], label: str, ev: float
) -> str:
    """Directional bar: conditional mean SHAP per feature (mean over rows where feature=1).

    Averaging sv over ALL rows gives ~0 for binary dummies because SHAP's conservation
    property causes positive contributions when active to cancel small negatives when
    inactive.  The meaningful quantity is E[SHAP | feature=1]: the average price impact
    in SPs where that label×side is actually at the margin.
    """
    cond_mean = np.zeros(sv.shape[1])
    active_counts = np.zeros(sv.shape[1], dtype=int)
    for j in range(sv.shape[1]):
        mask = X_s[:, j] > 0.5
        active_counts[j] = int(mask.sum())
        if active_counts[j] > 0:
            cond_mean[j] = sv[mask, j].mean()

    mean_abs = np.abs(sv).mean(axis=0)   # unconditional |SHAP| for ranking
    eligible = np.flatnonzero(active_counts > 0)
    top_idx  = eligible[np.argsort(mean_abs[eligible])[-TOP_N:]]
    names    = [feature_names[i] for i in top_idx]
    values   = cond_mean[top_idx]
    colours  = ["#4CAF50" if v >= 0 else "#EF5350" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * TOP_N + 1)))
    ax.barh(names, values, color=colours, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Conditional mean SHAP (£/MWh) when feature=1  [baseline = {ev:.1f} £/MWh]")
    ax.set_title(f"SHAP directional effect — {label} × side  (top {TOP_N} by mean |SHAP|)", fontsize=11)
    ax.legend(handles=[
        Patch(facecolor="#4CAF50", label="Positive → raises predicted price when at margin"),
        Patch(facecolor="#EF5350", label="Negative → lowers predicted price when at margin"),
    ], fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_mdi_vs_shap(
    mdi: np.ndarray, sv: np.ndarray, feature_names: list[str], label: str
) -> str:
    """Side-by-side normalised bar comparing MDI rank with mean |SHAP| rank."""
    mean_abs  = np.abs(sv).mean(axis=0)
    top_idx   = np.argsort(mdi)[-TOP_N:]             # rank by MDI, ascending
    names     = [feature_names[i] for i in top_idx]
    mdi_vals  = mdi[top_idx]
    shap_vals = mean_abs[top_idx]
    mdi_n     = mdi_vals  / mdi_vals.max()
    shap_n    = shap_vals / shap_vals.max() if shap_vals.max() > 0 else shap_vals

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * TOP_N + 1)))
    ax.barh(y - 0.18, mdi_n,  0.35, label="MDI (normalised)",         color="#7986CB", edgecolor="white")
    ax.barh(y + 0.18, shap_n, 0.35, label="Mean |SHAP| (normalised)", color="#FF7043", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Normalised importance (each method scaled to max = 1)")
    ax.set_title(f"MDI vs mean |SHAP| — {label} × side  (top {TOP_N} by MDI)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_beeswarm(
    X_s: np.ndarray, sv: np.ndarray, feature_names: list[str], label: str
) -> str:
    """Beeswarm: one dot per SP per feature, x=SHAP value, colour=feature value."""
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-BEESWARM_N:]    # ascending → bottom-to-top

    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * BEESWARM_N + 1.5)))

    for plot_row, feat_idx in enumerate(top_idx):
        sv_col  = sv[:, feat_idx]
        fv_col  = X_s[:, feat_idx]
        jitter  = rng.uniform(-0.3, 0.3, size=len(sv_col))
        colours = np.where(fv_col > 0.5, "#EF5350", "#3F51B5")
        ax.scatter(sv_col, plot_row + jitter, c=colours, alpha=0.25, s=7, linewidths=0)

    ax.set_yticks(range(BEESWARM_N))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (£/MWh)")
    ax.set_title(
        f"SHAP beeswarm — {label} × side  (top {BEESWARM_N} by mean |SHAP|)", fontsize=11
    )
    ax.legend(handles=[
        Patch(facecolor="#EF5350", label="Feature = 1  (this label×side is at margin)"),
        Patch(facecolor="#3F51B5", label="Feature = 0  (not at margin)"),
    ], fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


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
    predictor: str,
    source_column: str,
    r2: float,
    ev: float,
) -> pd.DataFrame:
    active_counts = (X_s > 0.5).sum(axis=0)
    mean_abs = np.abs(sv).mean(axis=0)
    cond_mean = np.full(len(feature_names), np.nan)
    share_pos = np.full(len(feature_names), np.nan)
    share_neg = np.full(len(feature_names), np.nan)

    for j in np.flatnonzero(active_counts > 0):
        active_sv = sv[X_s[:, j] > 0.5, j]
        cond_mean[j] = float(active_sv.mean())
        share_pos[j] = float((active_sv > 0).mean())
        share_neg[j] = float((active_sv < 0).mean())

    out = pd.DataFrame(
        {
            "predictor": predictor,
            "source_column": source_column,
            "feature": feature_names,
            "side": [feature_side(name) for name in feature_names],
            "mean_abs_shap": mean_abs,
            "conditional_mean_shap_when_active": cond_mean,
            "share_positive_shap_when_active": share_pos,
            "share_negative_shap_when_active": share_neg,
            "active_rows_in_shap_sample": active_counts,
            "mdi": mdi,
            "model_train_r2": r2,
            "shap_expected_value": ev,
        }
    )
    out["overall_shap_rank"] = out["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    out["side_shap_rank"] = (
        out.groupby("side")["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    )
    return out.sort_values("overall_shap_rank")


def top_side_table(summary: pd.DataFrame, side: str, n: int = 20) -> str:
    block = summary.loc[
        summary["side"].eq(side) & summary["active_rows_in_shap_sample"].gt(0)
    ].sort_values("mean_abs_shap", ascending=False).head(n)
    show_cols = [
        "side_shap_rank",
        "overall_shap_rank",
        "feature",
        "mean_abs_shap",
        "conditional_mean_shap_when_active",
        "share_positive_shap_when_active",
        "share_negative_shap_when_active",
        "active_rows_in_shap_sample",
        "mdi",
    ]
    if block.empty:
        return "<p class=\"note\">No active features for this side in the SHAP sample.</p>"
    return block[show_cols].round(4).to_html(index=False, border=0, classes="tbl")


# ── HTML boilerplate ──────────────────────────────────────────────────────────

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


# ── incremental HTML writer ───────────────────────────────────────────────────

def write_html(sections: list[str], setup_html: str, out_path: Path) -> None:
    """Write a valid HTML file from whatever sections have completed so far."""
    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SHAP Price Importance (Sided Updated) — 2023–2025</title>
{CSS}
</head><body>
<h1>SHAP Updated: System Price ~ Marginal Identity × NIV Side — 2023–2025</h1>
<p>Updated TreeSHAP values from one selected flex-3 RF regressor per predictor family (seed=0).
   SHAP decomposes each prediction into additive per-feature contributions, giving both
   direction and magnitude of each label×side combination's effect on predicted system price.
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>
{setup_html}
{body}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    y = sp["systemPrice"].values
    n_rows = len(sp)
    n_sps  = sp[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    print(f"  {n_rows:,} rows | {n_sps:,} unique SPs")

    shap_max_rows = {col: SHAP_MAX[col] for col, _ in PREDICTORS}
    n_trees_map   = {col: N_TREES[col]  for col, _ in PREDICTORS}
    max_depth_map = {col: MAX_DEPTH[col] for col, _ in PREDICTORS}
    min_leaf_map  = {col: MIN_SAMPLES_LEAF[col] for col, _ in PREDICTORS}

    setup_html = f"""
<section>
  <h2>Setup</h2>
  <table class="tbl">
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Rows</td><td>{n_rows:,} marginal candidate rows</td></tr>
    <tr><td>Unique SPs</td><td>{n_sps:,}</td></tr>
    <tr><td>Date range</td><td>{sp['settlementDate'].dt.date.min()} to {sp['settlementDate'].dt.date.max()}</td></tr>
    <tr><td>Target</td><td>systemPrice (£/MWh) — regression</td></tr>
    <tr><td>Profile</td><td>Updated selected profile: flex-3 depth/leaf settings chosen from the runtime/stability sweep.</td></tr>
    <tr><td>RF trees</td><td>fuel type {N_TREES['marginal_fuel_type']} | family {N_TREES['marginal_family_id']} | BMU {N_TREES['marginal_bmu_id']} — random_state=0, max_features=sqrt</td></tr>
    <tr><td>RF regularisation</td><td>max_depth fuel {MAX_DEPTH['marginal_fuel_type']} | family {MAX_DEPTH['marginal_family_id']} | BMU {MAX_DEPTH['marginal_bmu_id']}; min_samples_leaf={MIN_SAMPLES_LEAF['marginal_bmu_id']}</td></tr>
    <tr><td>SHAP method</td><td>TreeSHAP exact (tree_path_dependent perturbation)</td></tr>
    <tr><td>SHAP rows</td><td>fuel type {SHAP_MAX['marginal_fuel_type']:,} | family {SHAP_MAX['marginal_family_id']:,} | BMU {SHAP_MAX['marginal_bmu_id']:,} (random subsample, seed=0)</td></tr>
    <tr><td>Encoding</td><td>Label suffixed _OFFER or _BID before one-hot encoding — same as sided MDI script</td></tr>
  </table>
  <p class="note">SHAP decomposes each prediction into additive per-feature contributions around a global
     baseline (expected model output). For binary dummy features, read directional values as
     model associations: when this label-side dummy is active, the model's predicted price is
     pushed above or below baseline. These are not causal unit-offer experiments.</p>
</section>"""

    sections: list[str] = []
    all_summaries: list[pd.DataFrame] = []

    for col, label in PREDICTORS:
        shap_max = shap_max_rows[col]
        n_trees  = n_trees_map[col]
        max_depth = max_depth_map[col]
        min_samples_leaf = min_leaf_map[col]
        sided_col = f"{col}_sided"
        sp[sided_col] = make_sided_col(sp, col)
        n_unique = sp[sided_col].nunique()
        print(
            f"\n{label} sided ({n_unique} label×side combinations, trees={n_trees}, "
            f"max_depth={max_depth}, min_leaf={min_samples_leaf}, SHAP rows={shap_max:,}) ..."
        )

        dummies = pd.get_dummies(sp[sided_col], prefix="", prefix_sep="")
        feature_names = dummies.columns.tolist()
        X = dummies.values.astype(float)

        print("  Fitting RF ...")
        rf = fit_rf(X, y, n_trees, max_depth, min_samples_leaf)
        r2 = rf.score(X, y)
        print(f"  Train R² = {r2:.4f}")

        print(f"  Computing TreeSHAP (n_features={len(feature_names)}, rows={shap_max:,}) ...")
        top_mdi_idx = np.argsort(rf.feature_importances_)[-min(SHAP_GUARANTEE_TOP_MDI, len(feature_names)):]
        sv, X_s, ev = compute_shap(rf, X, shap_max, top_mdi_idx)

        top5_shap = (
            pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
            .nlargest(5).index.tolist()
        )
        print(f"  Expected value = {ev:.2f} £/MWh")
        print(f"  Top 5 by |mean SHAP|: {top5_shap}")

        b64_signed = plot_mean_shap_signed(sv, X_s, feature_names, label, ev)
        b64_comp   = plot_mdi_vs_shap(rf.feature_importances_, sv, feature_names, label)
        b64_bee    = plot_beeswarm(X_s, sv, feature_names, label)
        summary = build_feature_summary(
            sv,
            X_s,
            feature_names,
            rf.feature_importances_,
            label,
            col,
            float(r2),
            ev,
        )
        all_summaries.append(summary)
        offer_table = top_side_table(summary, "OFFER")
        bid_table = top_side_table(summary, "BID")

        stat_html = (
            f'<div>'
            f'<span class="stat"><strong>{n_unique}</strong>label×side combinations</span>'
            f'<span class="stat"><strong>{n_trees}</strong>trees</span>'
            f'<span class="stat"><strong>{r2:.3f}</strong>train R²</span>'
            f'<span class="stat"><strong>{ev:.1f} £/MWh</strong>SHAP baseline</span>'
            f'</div>'
        )

        sections.append(f"""
<section>
  <h2>Predictor: {label} × NIV side</h2>
  {stat_html}

  <h3>Mean SHAP — directional impact on system price</h3>
  <img src="data:image/png;base64,{b64_signed}" style="width:100%;max-width:850px;">
  <p class="note">Conditional mean SHAP: average SHAP value across SPs where that label×side
     is actually at the margin (feature=1).  Green bars push predicted price above the baseline
     ({ev:.1f} £/MWh) when that technology sets the margin; red bars suppress it.
     Ranked by mean |SHAP| (unconditional) so the most price-relevant features appear at the top.</p>

  <h3>MDI vs mean |SHAP| — importance method comparison</h3>
  <img src="data:image/png;base64,{b64_comp}" style="width:100%;max-width:850px;">
  <p class="note">Both metrics normalised to max=1 for visual comparison (top {TOP_N} features by MDI).
     Close agreement validates the MDI ranking. Divergence highlights cases where MDI overstates
     importance (high split frequency but low actual price contribution) or understates it
     (infrequent but high-impact splits).</p>

  <h3>SHAP beeswarm — distribution of per-SP effects (top {BEESWARM_N} features)</h3>
  <img src="data:image/png;base64,{b64_bee}" style="width:100%;max-width:850px;">
  <p class="note">Each point is one settlement period ({shap_max:,} sampled).
     Red = feature active (this label×side is at margin); blue = feature inactive.
     Horizontal spread of red dots shows how price impact varies across SPs where that
     label×side is marginal — a wide spread indicates context-dependent or volatile pricing.</p>

  <h3>Top OFFER features by mean |SHAP|</h3>
  {offer_table}
  <p class="note">The positive/negative shares show how consistently active SHAP values point
     in one direction. A high-importance feature with mixed shares is important but
     context-dependent.</p>

  <h3>Top BID features by mean |SHAP|</h3>
  {bid_table}
  <p class="note">BID rows are price-regime associations within the marginal system state, not
     the SO cashflow cost/benefit. Use the separate BID cost/benefit diagnostic for that
     accounting interpretation.</p>
</section>""")

        # write after every section so a restart doesn't lose completed work
        write_html(sections, setup_html, OUT_HTML)
        pd.concat(all_summaries, ignore_index=True).to_csv(OUT_CSV, index=False)
        print(f"  Saved ({len(sections)}/3 sections) -> {OUT_HTML}")

    print(f"\nDone -> {OUT_HTML}")
    print(f"Done -> {OUT_CSV}")


if __name__ == "__main__":
    main()
