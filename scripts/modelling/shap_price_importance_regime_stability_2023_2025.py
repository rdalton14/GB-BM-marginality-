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

from scripts.modelling import shap_price_importance_2023_2025_sided_updated as shap_updated  # noqa: E402

SP_SUMMARY_PATH = shap_updated.SP_SUMMARY_PATH
REPORT_DIR = PROJECT_ROOT / "reports" / "full_2023_2025"
OUT_HTML = REPORT_DIR / "canonical" / "shap_price_importance_regime_stability_2023_2025.html"
OUT_CSV = REPORT_DIR / "tables" / "shap_price_importance_regime_stability_2023_2025.csv"

TOP_N_PER_REGIME = 20
TOP_N_HEATMAP = 18
PREDICTORS = shap_updated.PREDICTORS


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
    sp.loc[price.le(quantiles["p10"]), "price_regime"] = "low_le_p10"
    sp.loc[price.gt(quantiles["p10"]) & price.le(quantiles["p75"]), "price_regime"] = "normal_p10_p75"
    sp.loc[price.gt(quantiles["p90"]) & price.le(quantiles["p95"]), "price_regime"] = "high_p90_p95"
    sp.loc[price.gt(quantiles["p95"]), "price_regime"] = "extreme_gt_p95"
    return sp.loc[sp["price_regime"].notna()].copy(), quantiles


def regime_order() -> list[str]:
    return ["low_le_p10", "normal_p10_p75", "high_p90_p95", "extreme_gt_p95"]


def regime_label(regime: str) -> str:
    return {
        "low_le_p10": "Low <= p10",
        "normal_p10_p75": "Normal p10-p75",
        "high_p90_p95": "High p90-p95",
        "extreme_gt_p95": "Extreme > p95",
    }[regime]


def run_regime_predictor(sp: pd.DataFrame, regime: str, col: str, label: str) -> pd.DataFrame:
    block = sp.loc[sp["price_regime"].eq(regime)].copy()
    sided_col = f"{col}_sided"
    block[sided_col] = shap_updated.make_sided_col(block, col)
    y = block["systemPrice"].values

    dummies = pd.get_dummies(block[sided_col], prefix="", prefix_sep="")
    feature_names = dummies.columns.tolist()
    X = dummies.values.astype(float)

    rf = shap_updated.fit_rf(
        X,
        y,
        shap_updated.N_TREES[col],
        shap_updated.MAX_DEPTH[col],
        shap_updated.MIN_SAMPLES_LEAF[col],
    )
    top_mdi_idx = np.argsort(rf.feature_importances_)[
        -min(shap_updated.SHAP_GUARANTEE_TOP_MDI, len(feature_names)):
    ]
    sv, X_s, ev = shap_updated.compute_shap(rf, X, shap_updated.SHAP_MAX[col], top_mdi_idx)

    out = shap_updated.build_feature_summary(
        sv=sv,
        X_s=X_s,
        feature_names=feature_names,
        mdi=rf.feature_importances_,
        predictor=label,
        source_column=col,
        r2=float(rf.score(X, y)),
        ev=ev,
    )
    out.insert(0, "regime", regime)
    out.insert(1, "regime_label", regime_label(regime))
    out["regime_rows"] = len(block)
    out["regime_unique_sps"] = block[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    out["regime_shap_rows"] = len(X_s)
    out["regime_price_min"] = float(block["systemPrice"].min())
    out["regime_price_median"] = float(block["systemPrice"].median())
    out["regime_price_max"] = float(block["systemPrice"].max())
    return out


def build_regime_summary(sp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in regime_order():
        block = sp.loc[sp["price_regime"].eq(regime)]
        rows.append(
            {
                "regime": regime,
                "regime_label": regime_label(regime),
                "rows": len(block),
                "unique_sps": block[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0],
                "price_min": float(block["systemPrice"].min()),
                "price_p25": float(block["systemPrice"].quantile(0.25)),
                "price_median": float(block["systemPrice"].median()),
                "price_p75": float(block["systemPrice"].quantile(0.75)),
                "price_max": float(block["systemPrice"].max()),
                "offer_share": float(block["niv_active_side"].eq("offer").mean()),
                "bid_share": float(block["niv_active_side"].eq("bid").mean()),
            }
        )
    return pd.DataFrame(rows)


def build_stability_table(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    block = df.loc[df["predictor"].eq(predictor)].copy()
    top_features = (
        block.loc[block["overall_shap_rank"].le(TOP_N_PER_REGIME), "feature"]
        .drop_duplicates()
        .tolist()
    )
    block = block.loc[block["feature"].isin(top_features)].copy()

    rows = []
    for feature, grp in block.groupby("feature", sort=False):
        side = grp["side"].dropna().iloc[0] if not grp["side"].dropna().empty else "UNKNOWN"
        top_regimes = grp.loc[grp["overall_shap_rank"].le(TOP_N_PER_REGIME), "regime_label"].tolist()
        signs = []
        for regime in regime_order():
            match = grp.loc[grp["regime"].eq(regime)]
            if match.empty:
                signs.append("NA")
                continue
            value = match["conditional_mean_shap_when_active"].iloc[0]
            if pd.isna(value):
                signs.append("NA")
            elif value > 0:
                signs.append("+")
            elif value < 0:
                signs.append("-")
            else:
                signs.append("0")
        rows.append(
            {
                "predictor": predictor,
                "feature": feature,
                "side": side,
                "regimes_in_top20": len(top_regimes),
                "top20_regimes": ", ".join(top_regimes),
                "best_rank": int(grp["overall_shap_rank"].min()),
                "mean_abs_shap_mean": float(grp["mean_abs_shap"].mean()),
                "conditional_mean_shap_mean": float(grp["conditional_mean_shap_when_active"].mean()),
                "direction_pattern": " | ".join(signs),
                "min_active_rows_in_shap_sample": int(grp["active_rows_in_shap_sample"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["regimes_in_top20", "mean_abs_shap_mean"], ascending=[False, False]
    )


def plot_signed_heatmap(df: pd.DataFrame, predictor: str) -> str:
    stability = build_stability_table(df, predictor)
    features = stability.head(TOP_N_HEATMAP)["feature"].tolist()
    block = df.loc[df["predictor"].eq(predictor) & df["feature"].isin(features)].copy()
    pivot = block.pivot_table(
        index="feature",
        columns="regime_label",
        values="conditional_mean_shap_when_active",
        aggfunc="first",
    ).reindex(features)
    labels = [regime_label(r) for r in regime_order()]
    pivot = pivot.reindex(columns=labels)

    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.36 * len(pivot) + 1.2)))
    vals = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"Signed conditional SHAP by price regime - {predictor}", fontsize=11)
    ax.set_xlabel("Price regime")
    ax.set_ylabel("Feature")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Conditional mean SHAP when active")
    fig.tight_layout()
    return fig_to_b64(fig)


CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1180px; margin: 40px auto; padding: 0 20px; color: #212121; }
h1 { color: #3F51B5; border-bottom: 3px solid #3F51B5; padding-bottom: 8px; }
h2 { margin-top: 42px; color: #283593; border-left: 4px solid #7986CB; padding-left: 10px; }
h3 { color: #37474F; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 14px 0 28px; }
th { background: #3F51B5; color: white; text-align: left; padding: 6px 8px; }
td { border-bottom: 1px solid #e0e0e0; padding: 5px 8px; vertical-align: top; }
tr:nth-child(even) td { background: #f5f5f5; }
img { display:block; max-width: 980px; width:100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,.15); }
p.note { color: #666; font-size: 13px; }
</style>
"""


def make_html(df: pd.DataFrame, sp: pd.DataFrame, regime_stats: pd.DataFrame, quantiles: dict[str, float]) -> str:
    sections = []
    for _, label in PREDICTORS:
        stability = build_stability_table(df, label)
        heatmap = plot_signed_heatmap(df, label)

        show_cols = [
            "feature",
            "side",
            "regimes_in_top20",
            "top20_regimes",
            "best_rank",
            "mean_abs_shap_mean",
            "conditional_mean_shap_mean",
            "direction_pattern",
            "min_active_rows_in_shap_sample",
        ]
        sections.append(
            f"""
<section>
  <h2>{label}</h2>
  <img src="data:image/png;base64,{heatmap}">
  <h3>Features recurring in regime top {TOP_N_PER_REGIME}</h3>
  {stability[show_cols].head(45).round(4).to_html(index=False, border=0)}
</section>"""
        )

    q_html = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in quantiles.items()
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SHAP Price Regime Stability 2023-2025</title>
{CSS}
</head><body>
<h1>SHAP Price Regime Stability: 2023-2025</h1>
<p class="note">Separate TreeSHAP runs using the same updated/flex-3 RF settings as the main
SHAP diagnostic, but fitted within non-overlapping system-price regimes. This explains variation
within each regime, not the causal transition from normal prices into spikes.</p>
<table>
  <tr><th>Item</th><th>Value</th></tr>
  <tr><td>Source rows before regime filter</td><td>{len(sp):,}</td></tr>
  <tr><td>Date range</td><td>{sp['settlementDate'].dt.date.min()} to {sp['settlementDate'].dt.date.max()}</td></tr>
  <tr><td>Regimes</td><td>Low <= p10; Normal p10-p75; High p90-p95; Extreme > p95</td></tr>
  <tr><td>Excluded band</td><td>p75-p90 omitted to keep high/extreme contrast clean</td></tr>
  <tr><td>Direction pattern order</td><td>Low | Normal | High | Extreme</td></tr>
</table>
<h2>System Price Thresholds</h2>
<table><tr><th>Quantile</th><th>System price</th></tr>{q_html}</table>
<h2>Regime Summary</h2>
{regime_stats.round(4).to_html(index=False, border=0)}
{''.join(sections)}
</body></html>"""


def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    sp["systemPrice"] = pd.to_numeric(sp["systemPrice"], errors="coerce")
    sp, quantiles = assign_regimes(sp)
    regime_stats = build_regime_summary(sp)
    print(regime_stats[["regime_label", "rows", "price_min", "price_median", "price_max"]].to_string(index=False))

    blocks = []
    for regime in regime_order():
        for col, label in PREDICTORS:
            print(f"\n{regime_label(regime)} {label}: regime SHAP run ...")
            block = run_regime_predictor(sp, regime, col, label)
            blocks.append(block)
            top5 = block.head(5)["feature"].tolist()
            print(f"  top5={', '.join(top5)}")

    out = pd.concat(blocks, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_HTML.write_text(make_html(out, sp, regime_stats, quantiles), encoding="utf-8")
    print(f"\nSaved -> {OUT_CSV}")
    print(f"Saved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
