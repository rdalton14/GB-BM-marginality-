from __future__ import annotations

import base64
import io
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.modelling import shap_bmu_price_importance_2023_2025 as shap_bmu  # noqa: E402

SP_SUMMARY_PATH = shap_bmu.SP_SUMMARY_PATH
REPORT_DIR = PROJECT_ROOT / "reports" / "full_2023_2025"
OUT_HTML = REPORT_DIR / "canonical" / "bmu" / "shap_bmu_regime_stability_2023_2025.html"
OUT_CSV  = REPORT_DIR / "tables"    / "shap_bmu_regime_stability_2023_2025.csv"

PREDICTOR_COL   = shap_bmu.PREDICTOR_COL
PREDICTOR_LABEL = shap_bmu.PREDICTOR_LABEL

TOP_N_PER_REGIME = 20
TOP_N_HEATMAP    = 18


def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def assign_regimes(sp: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    sp = sp.copy()
    price = pd.to_numeric(sp["systemPrice"], errors="coerce")
    quantiles = {
        "p10": float(price.quantile(0.10)),
        "p75": float(price.quantile(0.75)),
        "p90": float(price.quantile(0.90)),
        "p95": float(price.quantile(0.95)),
    }
    sp["price_regime"] = pd.Series(pd.NA, index=sp.index, dtype="object")
    sp.loc[price.le(quantiles["p10"]),                          "price_regime"] = "low_le_p10"
    sp.loc[price.gt(quantiles["p10"]) & price.le(quantiles["p75"]), "price_regime"] = "normal_p10_p75"
    sp.loc[price.gt(quantiles["p90"]) & price.le(quantiles["p95"]), "price_regime"] = "high_p90_p95"
    sp.loc[price.gt(quantiles["p95"]),                          "price_regime"] = "extreme_gt_p95"
    return sp.loc[sp["price_regime"].notna()].copy(), quantiles


REGIME_ORDER = ["low_le_p10", "normal_p10_p75", "high_p90_p95", "extreme_gt_p95"]
REGIME_LABELS = {
    "low_le_p10":      "Low <= p10",
    "normal_p10_p75":  "Normal p10–p75",
    "high_p90_p95":    "High p90–p95",
    "extreme_gt_p95":  "Extreme > p95",
}


def build_regime_summary(sp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIME_ORDER:
        block = sp.loc[sp["price_regime"].eq(regime)]
        rows.append(
            {
                "regime":        regime,
                "regime_label":  REGIME_LABELS[regime],
                "rows":          len(block),
                "unique_sps":    block[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0],
                "price_min":     float(block["systemPrice"].min()),
                "price_median":  float(block["systemPrice"].median()),
                "price_max":     float(block["systemPrice"].max()),
                "offer_share":   float(block["niv_active_side"].eq("offer").mean()),
                "bid_share":     float(block["niv_active_side"].eq("bid").mean()),
            }
        )
    return pd.DataFrame(rows)


def run_regime(
    sp: pd.DataFrame, regime: str, fuel_lookup: dict[str, str]
) -> pd.DataFrame:
    block = sp.loc[sp["price_regime"].eq(regime)].copy()
    sided_col = f"{PREDICTOR_COL}_sided"
    block[sided_col] = shap_bmu.make_sided_col(block, PREDICTOR_COL)
    y = block["systemPrice"].values

    dummies = pd.get_dummies(block[sided_col], prefix="", prefix_sep="")
    feature_names = dummies.columns.tolist()
    X = dummies.values.astype(float)

    rf  = shap_bmu.fit_rf(X, y)
    sv, X_s, ev = shap_bmu.compute_shap(rf, X)

    out = shap_bmu.build_feature_summary(
        sv=sv, X_s=X_s, feature_names=feature_names,
        mdi=rf.feature_importances_, r2=float(rf.score(X, y)),
        ev=ev, fuel_lookup=fuel_lookup,
    )
    out.insert(0, "regime",        regime)
    out.insert(1, "regime_label",  REGIME_LABELS[regime])
    out["regime_rows"]        = len(block)
    out["regime_unique_sps"]  = block[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    out["regime_shap_rows"]   = len(X_s)
    out["regime_price_min"]   = float(block["systemPrice"].min())
    out["regime_price_median"] = float(block["systemPrice"].median())
    out["regime_price_max"]   = float(block["systemPrice"].max())
    return out


def build_stability_table(df: pd.DataFrame) -> pd.DataFrame:
    top_features = (
        df.loc[df["overall_shap_rank"].le(TOP_N_PER_REGIME), "feature"]
        .drop_duplicates().tolist()
    )
    block = df.loc[df["feature"].isin(top_features)].copy()

    rows = []
    for feature, grp in block.groupby("feature", sort=False):
        side = grp["side"].dropna().iloc[0] if not grp["side"].dropna().empty else "UNKNOWN"
        fuel = grp["fuel_type"].dropna().iloc[0] if not grp["fuel_type"].dropna().empty else "UNKNOWN"
        top_regimes = grp.loc[grp["overall_shap_rank"].le(TOP_N_PER_REGIME), "regime_label"].tolist()
        signs = []
        for regime in REGIME_ORDER:
            match = grp.loc[grp["regime"].eq(regime)]
            if match.empty or pd.isna(match["conditional_mean_shap_when_active"].iloc[0]):
                signs.append("NA")
            else:
                v = match["conditional_mean_shap_when_active"].iloc[0]
                signs.append("+" if v > 0 else ("-" if v < 0 else "0"))
        rows.append(
            {
                "feature":           feature,
                "fuel_type":         fuel,
                "side":              side,
                "regimes_in_top20":  len(set(top_regimes)),
                "top20_regimes":     ", ".join(top_regimes),
                "best_rank":         int(grp["overall_shap_rank"].min()),
                "mean_abs_shap_mean": float(grp["mean_abs_shap"].mean()),
                "conditional_mean_shap_mean": float(grp["conditional_mean_shap_when_active"].mean()),
                "direction_pattern": " | ".join(signs),
                "min_active_rows":   int(grp["active_rows_in_shap_sample"].min()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["regimes_in_top20", "mean_abs_shap_mean"], ascending=[False, False])
    )


def plot_signed_heatmap(df: pd.DataFrame) -> str:
    stability = build_stability_table(df)
    features  = stability.head(TOP_N_HEATMAP)["feature"].tolist()
    block     = df.loc[df["feature"].isin(features)].copy()

    fuel_map = df.drop_duplicates("feature").set_index("feature")["fuel_type"].to_dict()
    pivot = block.pivot_table(
        index="feature", columns="regime_label",
        values="conditional_mean_shap_when_active", aggfunc="first",
    ).reindex(features)
    labels_col = [REGIME_LABELS[r] for r in REGIME_ORDER]
    pivot = pivot.reindex(columns=labels_col)

    row_labels = [f"{f}  [{fuel_map.get(f, '?')}]" for f in features]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(pivot) + 1.5)))
    vals = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels_col)))
    ax.set_xticklabels(labels_col, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Conditional mean SHAP by price regime — BMU ID", fontsize=11)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Conditional mean SHAP when active (£/MWh)")
    fig.tight_layout()
    return fig_to_b64(fig)


CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #212121; }
h1 { color: #3F51B5; border-bottom: 3px solid #3F51B5; padding-bottom: 8px; }
h2 { margin-top: 42px; color: #283593; border-left: 4px solid #7986CB; padding-left: 10px; }
h3 { color: #37474F; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 14px 0 28px; }
th { background: #3F51B5; color: white; text-align: left; padding: 6px 8px; }
td { border-bottom: 1px solid #e0e0e0; padding: 5px 8px; }
tr:nth-child(even) td { background: #f5f5f5; }
img { display:block; max-width: 1050px; width:100%; border-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,.15); }
p.note { color: #666; font-size: 13px; }
</style>
"""


def make_html(
    df: pd.DataFrame, sp: pd.DataFrame,
    regime_stats: pd.DataFrame, quantiles: dict[str, float],
) -> str:
    stability = build_stability_table(df)
    heatmap   = plot_signed_heatmap(df)
    show_cols = [
        "feature", "fuel_type", "side", "regimes_in_top20", "top20_regimes",
        "best_rank", "mean_abs_shap_mean", "conditional_mean_shap_mean",
        "direction_pattern", "min_active_rows",
    ]
    q_html = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in quantiles.items())
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>BMU SHAP Regime Stability 2023–2025</title>
{CSS}
</head><body>
<h1>BMU ID SHAP Price Regime Stability: 2023–2025</h1>
<p class="note">Separate TreeSHAP runs within non-overlapping system-price regimes using
the same BMU-level RF settings as the main diagnostic. Explains variation within each
regime, not the causal transition into that regime. Fuel type annotated per BMU.
Direction pattern: Low | Normal | High | Extreme.</p>
<table>
  <tr><th>Item</th><th>Value</th></tr>
  <tr><td>Source rows before regime filter</td><td>{len(sp):,}</td></tr>
  <tr><td>Date range</td><td>{sp['settlementDate'].dt.date.min()} to {sp['settlementDate'].dt.date.max()}</td></tr>
  <tr><td>Regimes</td><td>Low ≤ p10; Normal p10–p75; High p90–p95; Extreme > p95</td></tr>
  <tr><td>Excluded band</td><td>p75–p90 omitted to sharpen high/extreme contrast</td></tr>
  <tr><td>RF settings</td><td>{shap_bmu.N_TREES} trees, max_depth={shap_bmu.MAX_DEPTH}, min_leaf={shap_bmu.MIN_SAMPLES_LEAF}</td></tr>
</table>
<h2>System Price Thresholds</h2>
<table><tr><th>Quantile</th><th>£/MWh</th></tr>{q_html}</table>
<h2>Regime Summary</h2>
{regime_stats.round(4).to_html(index=False, border=0)}
<h2>Signed conditional SHAP heatmap</h2>
<img src="data:image/png;base64,{heatmap}">
<h2>BMUs recurring in regime top {TOP_N_PER_REGIME}</h2>
{stability[show_cols].head(55).round(4).to_html(index=False, border=0)}
</body></html>"""


def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    sp["systemPrice"]    = pd.to_numeric(sp["systemPrice"], errors="coerce")
    sp, quantiles        = assign_regimes(sp)
    regime_stats         = build_regime_summary(sp)
    fuel_lookup          = shap_bmu.build_fuel_lookup(sp)
    print(regime_stats[["regime_label", "rows", "price_min", "price_median", "price_max"]].to_string(index=False))

    blocks = []
    for regime in REGIME_ORDER:
        print(f"\n{REGIME_LABELS[regime]}: BMU regime SHAP run ...")
        block = run_regime(sp, regime, fuel_lookup)
        blocks.append(block)
        print(f"  top5={', '.join(block.head(5)['feature'].tolist())}")

    out = pd.concat(blocks, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(make_html(out, sp, regime_stats, quantiles), encoding="utf-8")
    print(f"\nSaved -> {OUT_CSV}")
    print(f"Saved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
