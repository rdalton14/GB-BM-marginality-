"""
Diagnostic: SHAP coverage curves — how many features are needed to capture
X% of total mean |SHAP| mass at each predictor level (fuel type, family, BMU).

Reads the existing SHAP CSV — no model refitting required.
Output: self-contained HTML to reports/full_2023_2025/diagnostics/
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

SHAP_CSV = (
    PROJECT_ROOT
    / "reports" / "full_2023_2025" / "tables"
    / "shap_price_importance_2023_2025_sided_updated.csv"
)
OUT_HTML = (
    PROJECT_ROOT
    / "reports" / "full_2023_2025" / "diagnostics"
    / "probe_shap_coverage_curve_2023_2025.html"
)

THRESHOLDS  = [0.70, 0.80, 0.85, 0.90, 0.95]
THRESHOLD_COLOURS = {
    0.70: "#BDBDBD",
    0.80: "#90CAF9",
    0.85: "#42A5F5",
    0.90: "#1565C0",
    0.95: "#0D47A1",
}

PREDICTOR_ORDER = [
    ("marginal_fuel_type", "Generation type (fuel)"),
    ("marginal_family_id", "Plant family"),
    ("marginal_bmu_id",    "BMU"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:900px;">'


def coverage_stats(series: pd.Series) -> pd.DataFrame:
    """Given a mean_abs_shap series (one row per feature), return cumulative coverage frame."""
    s = series.sort_values(ascending=False).reset_index(drop=True)
    total = s.sum()
    cumulative = s.cumsum() / total
    return pd.DataFrame({
        "rank":       s.index + 1,
        "mean_abs_shap": s.values,
        "cumulative_coverage": cumulative.values,
    })


def n_features_at_threshold(cov_df: pd.DataFrame, threshold: float) -> int:
    mask = cov_df["cumulative_coverage"] >= threshold
    if mask.any():
        return int(cov_df.loc[mask, "rank"].iloc[0])
    return len(cov_df)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_coverage_curves(shap: pd.DataFrame) -> str:
    """One panel per predictor level — coverage curve with threshold annotations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col, label) in zip(axes, PREDICTOR_ORDER):
        sub = shap[shap["source_column"] == col]["mean_abs_shap"].dropna()
        if sub.empty:
            ax.set_title(f"{label}\n(no data)")
            continue

        cov = coverage_stats(sub)
        n_total = len(cov)

        ax.plot(cov["rank"], cov["cumulative_coverage"] * 100,
                color="#3F51B5", linewidth=2)
        ax.fill_between(cov["rank"], cov["cumulative_coverage"] * 100,
                        alpha=0.08, color="#3F51B5")

        for t in THRESHOLDS:
            k = n_features_at_threshold(cov, t)
            colour = THRESHOLD_COLOURS[t]
            ax.axhline(t * 100, color=colour, linestyle="--", linewidth=1)
            ax.axvline(k, color=colour, linestyle=":", linewidth=1)
            # label at the right margin
            ax.text(n_total * 1.01, t * 100, f"{int(t*100)}% → {k}",
                    fontsize=7.5, color=colour, va="center")

        ax.set_xlim(1, n_total)
        ax.set_ylim(0, 103)
        ax.set_xlabel("Features ranked by mean |SHAP| (descending)")
        ax.set_ylabel("Cumulative % of total mean |SHAP|")
        ax.set_title(f"{label}\n({n_total} features total)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.suptitle("SHAP coverage curves — how many features to capture X% of total price-formation effect",
                 fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_marginal_contribution(shap: pd.DataFrame) -> str:
    """Bar chart of each feature's individual % contribution — top 40 per predictor."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for ax, (col, label) in zip(axes, PREDICTOR_ORDER):
        sub = (
            shap[shap["source_column"] == col][["feature", "mean_abs_shap"]]
            .dropna()
            .sort_values("mean_abs_shap", ascending=False)
            .head(40)
        )
        if sub.empty:
            continue
        total = shap[shap["source_column"] == col]["mean_abs_shap"].sum()
        sub["pct"] = sub["mean_abs_shap"] / total * 100

        ax.barh(sub["feature"][::-1], sub["pct"][::-1],
                color="#5C6BC0", edgecolor="white", height=0.7)
        ax.set_xlabel("Individual share of total mean |SHAP| (%)")
        ax.set_title(f"{label} — top 40 individual contributions")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Individual feature SHAP contribution as % of total (top 40 per predictor level)",
                 fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── tables ────────────────────────────────────────────────────────────────────

def threshold_summary_table(shap: pd.DataFrame) -> str:
    """For each predictor × threshold: n features needed and % of features used."""
    rows = []
    for col, label in PREDICTOR_ORDER:
        sub = shap[shap["source_column"] == col]["mean_abs_shap"].dropna()
        n_total = len(sub)
        if n_total == 0:
            continue
        cov = coverage_stats(sub)
        for t in THRESHOLDS:
            k = n_features_at_threshold(cov, t)
            rows.append({
                "Predictor":       label,
                "Coverage target": f"{int(t*100)}%",
                "Features needed": k,
                "Total features":  n_total,
                "% of features":   f"{k/n_total*100:.1f}%",
            })
    return pd.DataFrame(rows).to_html(index=False, border=0, classes="tbl")


def feature_list_at_threshold(
    shap: pd.DataFrame,
    col: str,
    threshold: float,
    sp: pd.DataFrame | None = None,
) -> str:
    """Ordered feature list up to coverage threshold for one predictor level."""
    sub = (
        shap[shap["source_column"] == col]
        [["feature", "mean_abs_shap", "conditional_mean_shap_when_active",
          "active_rows_in_shap_sample", "overall_shap_rank"]]
        .dropna(subset=["mean_abs_shap"])
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    if sub.empty:
        return "<p class='note'>No data.</p>"

    total = sub["mean_abs_shap"].sum()
    sub["cumulative_coverage"] = sub["mean_abs_shap"].cumsum() / total
    sub["individual_share"]    = sub["mean_abs_shap"] / total

    # Trim to threshold
    mask = sub["cumulative_coverage"] <= threshold
    # include the row that crosses the threshold
    if not mask.all():
        first_over = sub[~mask].index[0]
        mask.iloc[first_over] = True
    sub = sub[mask].copy()

    sub["rank"] = range(1, len(sub) + 1)
    sub["individual_share"]    = sub["individual_share"].map("{:.2%}".format)
    sub["cumulative_coverage"] = sub["cumulative_coverage"].map("{:.2%}".format)
    sub["mean_abs_shap"]       = sub["mean_abs_shap"].round(4)
    sub["conditional_mean_shap_when_active"] = sub["conditional_mean_shap_when_active"].round(2)

    display_cols = [
        "rank", "feature", "mean_abs_shap", "conditional_mean_shap_when_active",
        "individual_share", "cumulative_coverage",
        "active_rows_in_shap_sample", "overall_shap_rank",
    ]
    sub = sub[[c for c in display_cols if c in sub.columns]]
    sub.columns = [c.replace("_", " ") for c in sub.columns]

    return sub.to_html(index=False, border=0, classes="tbl")


def long_tail_table(shap: pd.DataFrame) -> str:
    """Show the features NOT in top-85% — the long tail."""
    rows = []
    for col, label in PREDICTOR_ORDER:
        sub = (
            shap[shap["source_column"] == col]["mean_abs_shap"]
            .dropna()
            .sort_values(ascending=False)
        )
        if sub.empty:
            continue
        total   = sub.sum()
        cov     = sub.cumsum() / total
        tail    = sub[cov > 0.85]
        rows.append({
            "Predictor":       label,
            "Total features":  len(sub),
            "In top 85%":      int((cov <= 0.85).sum()) + 1,
            "Long tail (>85%)":len(tail),
            "Tail SHAP mass":  f"{tail.sum()/total*100:.1f}%",
            "Tail min |SHAP|": f"{tail.min():.4f}",
            "Tail max |SHAP|": f"{tail.max():.4f}",
        })
    return pd.DataFrame(rows).to_html(index=False, border=0, classes="tbl")


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #212121; }
  h1   { border-bottom: 3px solid #3F51B5; padding-bottom: 8px; color: #3F51B5; }
  h2   { margin-top: 36px; color: #283593; border-left: 4px solid #7986CB; padding-left: 10px; }
  h3   { color: #37474F; margin-top: 20px; }
  .tbl { border-collapse: collapse; font-size: 13px; width: 100%; margin-bottom: 16px; }
  .tbl th { background: #3F51B5; color: white; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
  .tbl tr:nth-child(even) td { background: #f5f5f5; }
  img  { display: block; margin: 12px 0; border-radius: 6px;
         box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  p.note { font-size: 12px; color: #757575; font-style: italic; }
  .callout { background: #f5f7fa; border-left: 4px solid #3F51B5;
             padding: 12px 14px; margin: 18px 0 24px 0; }
</style>
"""


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading SHAP CSV ...")
    shap = pd.read_csv(SHAP_CSV)
    print(f"  {len(shap):,} feature rows across {shap['source_column'].nunique()} predictor levels")

    for col, label in PREDICTOR_ORDER:
        sub = shap[shap["source_column"] == col]
        n   = len(sub)
        cov = coverage_stats(sub["mean_abs_shap"].dropna())
        print(f"\n  {label} ({n} features):")
        for t in THRESHOLDS:
            k = n_features_at_threshold(cov, t)
            print(f"    {int(t*100)}% coverage -> {k} features ({k/n*100:.1f}% of total)")

    print("\nBuilding charts ...")
    b64_curves = plot_coverage_curves(shap)
    b64_indiv  = plot_marginal_contribution(shap)

    print("Assembling HTML ...")
    body = f"""
{CSS}
<h1>SHAP Coverage Curves — 2023–2025 (diagnostic)</h1>
<p>Source: <code>{SHAP_CSV.name}</code><br>
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>

<div class="callout">
  <p>This diagnostic answers: <strong>how many features are needed to capture X% of the
  total mean |SHAP| mass?</strong> The current canonical script uses a fixed
  <code>TOP_N = 30</code> cutoff. These curves show whether a threshold-based cutoff
  (e.g. 85%) would produce a materially different feature set — and at which predictor
  level the distinction matters most.</p>
  <p>No model refitting: values are read directly from the existing SHAP CSV.</p>
</div>

<h2>1. Coverage curves</h2>
<p class="note">x = features ranked by descending mean |SHAP|; y = cumulative % of total
mean |SHAP| captured. Dashed horizontal lines = thresholds; dotted vertical lines = the
feature rank at which each threshold is crossed. Labels at right margin show
threshold → n features needed.</p>
{img_tag(b64_curves)}

<h2>2. Threshold summary</h2>
<p class="note">For each predictor level and coverage target, how many features are needed
and what fraction of the feature space they represent.</p>
{threshold_summary_table(shap)}

<h2>3. Individual feature contributions (top 40 per predictor)</h2>
<p class="note">Each bar is one feature's individual share of total mean |SHAP|.
The long tail starts where bars become visually indistinguishable from zero.</p>
{img_tag(b64_indiv)}

<h2>4. Long-tail breakdown (features beyond 85% threshold)</h2>
<p class="note">Shows how many features sit in the tail beyond 85% coverage and how
little collective SHAP mass they carry.</p>
{long_tail_table(shap)}

<h2>5. Feature list at 85% coverage</h2>
<h3>Generation type</h3>
{feature_list_at_threshold(shap, "marginal_fuel_type", 0.85)}
<h3>Plant family</h3>
{feature_list_at_threshold(shap, "marginal_family_id", 0.85)}
<h3>BMU</h3>
{feature_list_at_threshold(shap, "marginal_bmu_id", 0.85)}

<h2>6. Feature list at 90% coverage</h2>
<h3>Generation type</h3>
{feature_list_at_threshold(shap, "marginal_fuel_type", 0.90)}
<h3>Plant family</h3>
{feature_list_at_threshold(shap, "marginal_family_id", 0.90)}
<h3>BMU</h3>
{feature_list_at_threshold(shap, "marginal_bmu_id", 0.90)}
"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>SHAP Coverage Curves 2023–2025</title></head>"
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )
    print(f"\nSaved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
