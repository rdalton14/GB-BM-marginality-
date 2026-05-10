from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

SP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "q1_2026" / "bid_offer_stack_q1_2026"
    / "bid_offer_stack_energy_q1_2026_niv_marginal_sp_summary.parquet"
)
OUT_HTML = PROJECT_ROOT / "reports" / "q1_2026" / "canonical" / "rf_feature_importance_q1_2026.html"

TARGETS = [
    ("marginal_fuel_type",  "Generation type"),
    ("marginal_family_id",  "Family ID"),
    ("marginal_bmu_id",     "BMU ID"),
]

RF_PARAMS = dict(n_estimators=300, n_jobs=-1, random_state=42, max_features="sqrt")

FEATURE_LABELS = {
    "settlementPeriod":            "Settlement period (1–48)",
    "day_of_week":                 "Day of week",
    "month":                       "Month",
    "week":                        "ISO week",
    "is_weekend":                  "Is weekend",
    "niv_volume":                  "NIV volume (MWh)",
    "niv_active_side_enc":         "NIV active side (offer=1)",
    "systemBuyPrice":              "System buy price (£/MWh)",
    "systemSellPrice":             "System sell price (£/MWh)",
    "systemPrice":                 "System price (£/MWh)",
    "n_tied_marginal_candidates":  "Co-marginal candidates",
    "marginal_stack_rank":         "Stack rank",
    "marginal_bid_rank_filled":    "Bid rank (0 if offer SP)",
    "marginal_volume":             "Accepted volume (MWh)",
    "marginal_gen_capacity_mw":    "Generator capacity (MW)",
    "marginal_bmu_type_enc":       "BMU type (T/E/S/V/I)",
    "marginal_gsp_group_enc":      "GSP group",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def encode_cat(series: pd.Series) -> np.ndarray:
    le = LabelEncoder()
    filled = series.fillna("__MISSING__").astype(str)
    return le.fit_transform(filled)


def build_features(sp: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=sp.index)
    X["settlementPeriod"]           = sp["settlementPeriod"].astype(float)
    X["day_of_week"]                = sp["settlementDate"].dt.dayofweek.astype(float)
    X["month"]                      = sp["settlementDate"].dt.month.astype(float)
    X["week"]                       = sp["settlementDate"].dt.isocalendar().week.astype(float)
    X["is_weekend"]                 = (sp["settlementDate"].dt.dayofweek >= 5).astype(float)
    X["niv_volume"]                 = sp["niv_volume"]
    X["niv_active_side_enc"]        = (sp["niv_active_side"] == "offer").astype(float)
    X["systemBuyPrice"]             = sp["systemBuyPrice"]
    X["systemSellPrice"]            = sp["systemSellPrice"]
    X["systemPrice"]                = sp["systemPrice"]
    X["n_tied_marginal_candidates"] = sp["n_tied_marginal_candidates"].astype(float)
    X["marginal_stack_rank"]        = sp["marginal_stack_rank"].astype(float)
    X["marginal_bid_rank_filled"]   = sp["marginal_bid_rank"].fillna(0).astype(float)
    X["marginal_volume"]            = sp["marginal_volume"]
    X["marginal_gen_capacity_mw"]   = sp["marginal_gen_capacity_mw"].fillna(0)
    X["marginal_bmu_type_enc"]      = encode_cat(sp["marginal_bmu_type"])
    X["marginal_gsp_group_enc"]     = encode_cat(sp["marginal_gsp_group"])
    return X


def plot_importance(importances: pd.Series, target_label: str, n_top: int = 15) -> str:
    top = importances.sort_values(ascending=False).head(n_top)
    top_named = top.rename(index=lambda k: FEATURE_LABELS.get(k, k))
    top_named = top_named.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(top_named) + 1.2))
    bars = ax.barh(top_named.index, top_named.values, color="#3F51B5", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_xlabel("Mean decrease in impurity (MDI)")
    ax.set_title(f"Feature importance — {target_label}", fontsize=11)
    ax.set_xlim(0, top_named.values.max() * 1.18)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_comparison(all_importances: dict[str, pd.Series]) -> str:
    """Normalised importance of each feature across all three targets."""
    df = pd.DataFrame(all_importances)
    df = df.rename(index=lambda k: FEATURE_LABELS.get(k, k))
    # normalise each column to sum to 1 for comparability
    df = df.div(df.sum())
    df = df.loc[df.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df) + 1.5))
    x = np.arange(len(df))
    width = 0.25
    colours = ["#3F51B5", "#E91E63", "#FF9800"]
    for i, (col, colour) in enumerate(zip(df.columns, colours)):
        ax.barh(x + i * width, df[col], width, label=col, color=colour, edgecolor="white")
    ax.set_yticks(x + width)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel("Normalised MDI importance")
    ax.set_title("Feature importance comparison across targets")
    ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig)


CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #212121; }
  h1   { border-bottom: 3px solid #3F51B5; padding-bottom: 8px; color: #3F51B5; }
  h2   { margin-top: 40px; color: #283593; border-left: 4px solid #7986CB;
         padding-left: 10px; }
  h3   { color: #37474F; }
  section { margin-bottom: 40px; }
  .tbl { border-collapse: collapse; font-size: 13px; width: 100%; margin-bottom: 16px; }
  .tbl th { background: #3F51B5; color: white; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
  .tbl tr:nth-child(even) td { background: #f5f5f5; }
  img  { display: block; margin: 12px 0; border-radius: 6px;
         box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  p.note { font-size: 12px; color: #757575; font-style: italic; }
  .chip { display:inline-block; padding:2px 8px; border-radius:12px;
          background:#E8EAF6; font-size:12px; margin:2px; }
</style>
"""


def class_summary_table(sp: pd.DataFrame, col: str, n: int = 10) -> str:
    counts = sp[col].value_counts().head(n).reset_index()
    counts.columns = [col, "n_candidates"]
    counts["pct"] = (counts["n_candidates"] / len(sp) * 100).round(1).astype(str) + "%"
    return counts.to_html(border=0, classes="tbl", index=False)


def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    print(f"  {len(sp):,} rows, {sp[['settlementDate','settlementPeriod']].drop_duplicates().shape[0]:,} SPs")

    print("Building feature matrix ...")
    X = build_features(sp)
    feature_names = list(X.columns)
    print(f"  {len(feature_names)} features, {len(X):,} rows")

    sections_html = ""
    all_importances: dict[str, pd.Series] = {}

    for col, label in TARGETS:
        print(f"Training RF for target: {label} ...")
        y_raw = sp[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
        print(f"  {n_classes} unique classes")

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X, y)

        imp = pd.Series(rf.feature_importances_, index=feature_names)
        all_importances[label] = imp

        top5 = (
            imp.sort_values(ascending=False).head(5)
            .rename(index=lambda k: FEATURE_LABELS.get(k, k))
        )
        top5_chips = "".join(f'<span class="chip">{k}: {v:.4f}</span>' for k, v in top5.items())

        sections_html += f"""
<section>
  <h2>Target: {label} <small style="font-size:13px;color:#757575;">({n_classes} classes)</small></h2>
  <h3>Top-10 most frequent classes</h3>
  {class_summary_table(sp, col)}
  <h3>Top 5 features</h3>
  <p>{top5_chips}</p>
  <h3>Feature importance (top 15)</h3>
  <img src="data:image/png;base64,{plot_importance(imp, label)}" style="width:100%;max-width:850px;">
  <p class="note">MDI (mean decrease in impurity) on 100% training data, {RF_PARAMS['n_estimators']} trees.
     Features with high MDI drive splits across the forest.</p>
</section>
"""
        print(f"  Done. Top feature: {FEATURE_LABELS.get(imp.idxmax(), imp.idxmax())} ({imp.max():.4f})")

    print("Building comparison chart ...")
    comp_b64 = plot_comparison(all_importances)

    # summary table: top feature per target
    summary_rows = ""
    for label, imp in all_importances.items():
        top3 = imp.sort_values(ascending=False).head(3)
        top3_str = ", ".join(f"{FEATURE_LABELS.get(k,k)} ({v:.3f})" for k, v in top3.items())
        summary_rows += f"<tr><td>{label}</td><td>{top3_str}</td></tr>"

    preamble = f"""
<section>
  <h2>Setup</h2>
  <table class="tbl">
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Rows (marginal candidates)</td><td>{len(sp):,}</td></tr>
    <tr><td>Unique SPs</td><td>{sp[['settlementDate','settlementPeriod']].drop_duplicates().shape[0]:,}</td></tr>
    <tr><td>Features</td><td>{len(feature_names)}: time, NIV, system prices, stack position, unit characteristics</td></tr>
    <tr><td>Training split</td><td>100% (no held-out test set — MDI only)</td></tr>
    <tr><td>Estimators</td><td>{RF_PARAMS['n_estimators']}</td></tr>
    <tr><td>Max features</td><td>{RF_PARAMS['max_features']}</td></tr>
    <tr><td>Note</td><td>Co-marginal SPs contribute multiple rows with identical SP-level features but different unit-level targets</td></tr>
  </table>

  <h2>Cross-target comparison</h2>
  <table class="tbl">
    <tr><th>Target</th><th>Top 3 features</th></tr>
    {summary_rows}
  </table>
  <img src="data:image/png;base64,{comp_b64}" style="width:100%;max-width:950px;">
  <p class="note">Each column normalised to sum to 1 for comparability across targets with different class counts.</p>
</section>
"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>RF Feature Importance — Q1 2026</title>
{CSS}
</head><body>
<h1>Random Forest Feature Importance — Q1 2026 Marginal SP Summary</h1>
<p>Three multiclass classifiers predicting what ends up at the margin, trained on 100% of data.
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>
{preamble}
{sections_html}
</body></html>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
