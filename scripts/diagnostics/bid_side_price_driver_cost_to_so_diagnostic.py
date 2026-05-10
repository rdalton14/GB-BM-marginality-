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

from scripts.diagnostics import shap_price_importance_2023_2025_sided_updated as shap_updated  # noqa: E402

SP_SUMMARY_PATH = shap_updated.SP_SUMMARY_PATH
REPORT_DIR = PROJECT_ROOT / "reports" / "full_2023_2025"
OUT_CSV = REPORT_DIR / "tables" / "bid_side_price_driver_cost_to_so_diagnostic.csv"
OUT_HTML = REPORT_DIR / "canonical" / "bid_side_price_driver_cost_to_so_diagnostic.html"

TOP_N_BID_FEATURES = 30
COST_EPSILON = 1e-9

PREDICTORS = [
    ("marginal_fuel_type", "Generation type"),
    ("marginal_family_id", "Family ID"),
    ("marginal_bmu_id", "BMU ID"),
]


def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return np.nan
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def weighted_share(mask: pd.Series, weights: pd.Series) -> float:
    valid = mask.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(weights.loc[valid & mask].sum() / weights.loc[valid].sum())


def classify_raw_bid_direction(so_pays_share: float, so_receives_share: float) -> str:
    if pd.isna(so_pays_share) or pd.isna(so_receives_share):
        return "unknown"
    if so_pays_share >= 0.75:
        return "mostly negative bid (SO pays)"
    if so_receives_share >= 0.75:
        return "mostly positive bid (SO receives)"
    if so_pays_share >= 0.55:
        return "leans negative bid (SO pays)"
    if so_receives_share >= 0.55:
        return "leans positive bid (SO receives)"
    return "mixed"


def build_shap_bid_table(sp: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    sided_col = f"{col}_sided"
    sp[sided_col] = shap_updated.make_sided_col(sp, col)
    y = sp["systemPrice"].values

    dummies = pd.get_dummies(sp[sided_col], prefix="", prefix_sep="")
    feature_names = dummies.columns.to_numpy()
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

    active_counts = (X_s > 0.5).sum(axis=0)
    mean_abs = np.abs(sv).mean(axis=0)
    cond_mean = np.full(sv.shape[1], np.nan)
    for j in np.flatnonzero(active_counts > 0):
        cond_mean[j] = float(sv[X_s[:, j] > 0.5, j].mean())

    shap_df = pd.DataFrame(
        {
            "predictor": label,
            "source_column": col,
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "conditional_mean_shap_when_active": cond_mean,
            "active_rows_in_shap_sample": active_counts,
            "mdi": rf.feature_importances_,
            "model_train_r2": float(rf.score(X, y)),
            "shap_expected_value": ev,
        }
    )
    shap_df["overall_shap_rank"] = (
        shap_df["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    )
    bid = shap_df[
        shap_df["feature"].str.endswith("_BID")
        & shap_df["active_rows_in_shap_sample"].gt(0)
    ].copy()
    bid["bid_shap_rank"] = bid["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    return bid.sort_values("bid_shap_rank").head(TOP_N_BID_FEATURES)


def aggregate_bid_prices(sp: pd.DataFrame, col: str) -> pd.DataFrame:
    sided_col = f"{col}_sided"
    bid = sp.loc[sp["niv_active_side"].eq("bid")].copy()
    bid[sided_col] = shap_updated.make_sided_col(bid, col)
    bid["marginal_final_price"] = pd.to_numeric(
        bid["marginal_final_price"], errors="coerce"
    )
    bid["systemPrice"] = pd.to_numeric(bid["systemPrice"], errors="coerce")
    bid["tie_weight"] = 1 / pd.to_numeric(
        bid["n_tied_marginal_candidates"], errors="coerce"
    ).clip(lower=1).fillna(1)

    rows = []
    for feature, grp in bid.groupby(sided_col, dropna=False):
        raw_bid_price = grp["marginal_final_price"]
        weights = grp["tie_weight"]
        so_pays_mask = raw_bid_price.lt(-COST_EPSILON)
        so_receives_mask = raw_bid_price.gt(COST_EPSILON)
        zero_bid_mask = raw_bid_price.abs().le(COST_EPSILON)
        so_pays_share = weighted_share(so_pays_mask, weights)
        so_receives_share = weighted_share(so_receives_mask, weights)
        zero_bid_share = weighted_share(zero_bid_mask, weights)
        rows.append(
            {
                "feature": feature,
                "bid_rows": int(len(grp)),
                "weighted_bid_rows": float(weights.sum()),
                "unique_settlement_periods": int(
                    grp[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
                ),
                "median_system_price": float(grp["systemPrice"].median()),
                "mean_raw_bid_price": float(raw_bid_price.mean()),
                "median_raw_bid_price": float(raw_bid_price.median()),
                "weighted_mean_raw_bid_price": weighted_mean(raw_bid_price, weights),
                "share_so_pays_weighted": so_pays_share,
                "share_so_receives_weighted": so_receives_share,
                "share_zero_bid_weighted": zero_bid_share,
                "raw_bid_price_direction": classify_raw_bid_direction(
                    so_pays_share, so_receives_share
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_direction_mix(df: pd.DataFrame, label: str) -> str:
    block = df.sort_values("mean_abs_shap", ascending=True).tail(TOP_N_BID_FEATURES)
    y = np.arange(len(block))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.34 * len(block) + 1)))
    ax.barh(y, block["share_so_pays_weighted"], color="#D32F2F", label="SO pays / negative bid")
    ax.barh(
        y,
        block["share_so_receives_weighted"],
        left=block["share_so_pays_weighted"],
        color="#2E7D32",
        label="SO receives / positive bid",
    )
    ax.barh(
        y,
        block["share_zero_bid_weighted"],
        left=block["share_so_pays_weighted"] + block["share_so_receives_weighted"],
        color="#BDBDBD",
        label="Zero bid",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(block["feature"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Tie-weighted share of bid-side marginal rows")
    ax.set_title(f"Raw bid-price direction among top bid-side SHAP drivers - {label}", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig_to_b64(fig)


def build_report(sp: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    for col, label in PREDICTORS:
        print(f"\n{label}: fitting updated SHAP model and aggregating raw bid prices ...")
        shap_bid = build_shap_bid_table(sp, col, label)
        price_stats = aggregate_bid_prices(sp, col)
        merged = shap_bid.merge(price_stats, on="feature", how="left")
        all_rows.append(merged)
        print(
            "  top bid features: "
            + ", ".join(merged["feature"].head(5).astype(str).tolist())
        )
    out = pd.concat(all_rows, ignore_index=True)
    cols = [
        "predictor",
        "source_column",
        "bid_shap_rank",
        "overall_shap_rank",
        "feature",
        "mean_abs_shap",
        "conditional_mean_shap_when_active",
        "model_train_r2",
        "bid_rows",
        "weighted_bid_rows",
        "unique_settlement_periods",
        "median_system_price",
        "mean_raw_bid_price",
        "median_raw_bid_price",
        "weighted_mean_raw_bid_price",
        "share_so_pays_weighted",
        "share_so_receives_weighted",
        "share_zero_bid_weighted",
        "raw_bid_price_direction",
        "active_rows_in_shap_sample",
        "mdi",
        "shap_expected_value",
    ]
    return out[cols]


CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1150px; margin: 40px auto; padding: 0 20px; color: #212121; }
h1 { color: #3F51B5; border-bottom: 3px solid #3F51B5; padding-bottom: 8px; }
h2 { margin-top: 42px; color: #283593; border-left: 4px solid #7986CB; padding-left: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 14px 0 28px; }
th { background: #3F51B5; color: white; text-align: left; padding: 6px 8px; }
td { border-bottom: 1px solid #e0e0e0; padding: 5px 8px; vertical-align: top; }
tr:nth-child(even) td { background: #f5f5f5; }
img { display:block; max-width: 900px; width:100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,.15); }
p.note { color: #666; font-size: 13px; }
.stat { display:inline-block; background:#E8EAF6; border-radius:8px;
        padding:8px 18px; margin:4px; font-size:14px; }
.stat strong { display:block; font-size:22px; color:#3F51B5; }
.callout { background:#F5F7FA; border-left:5px solid #2E7D32;
           padding:12px 16px; margin:18px 0 22px 0; }
.callout strong { color:#2E7D32; }
</style>
"""


def make_html(df: pd.DataFrame, sp: pd.DataFrame) -> str:
    sections = []
    for label, block in df.groupby("predictor", sort=False):
        plot_b64 = plot_direction_mix(block, label)
        direction_counts = block["raw_bid_price_direction"].value_counts().to_dict()
        stat_html = "".join(
            f'<span class="stat"><strong>{v}</strong>{k}</span>'
            for k, v in direction_counts.items()
        )
        show_cols = [
            "bid_shap_rank",
            "overall_shap_rank",
            "feature",
            "mean_abs_shap",
            "bid_rows",
            "weighted_bid_rows",
            "median_system_price",
            "median_raw_bid_price",
            "weighted_mean_raw_bid_price",
            "share_so_pays_weighted",
            "share_so_receives_weighted",
            "share_zero_bid_weighted",
            "raw_bid_price_direction",
        ]
        sections.append(
            f"""
<section>
  <h2>{label}</h2>
  <div>{stat_html}</div>
  <h3>Raw bid-price sign mix of main-model BID drivers</h3>
  <img src="data:image/png;base64,{plot_b64}">
  {block[show_cols].round(4).to_html(index=False, border=0)}
</section>"""
        )

    bid_rows = sp.loc[sp["niv_active_side"].eq("bid")]
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Bid-side price drivers and raw bid-price direction</title>
{CSS}
</head><body>
<h1>Main-Model BID Drivers: Raw Bid-Price Direction</h1>
<div class="callout">
  <p><strong>Canonical bid-side accounting interpretation.</strong> This report is the
  preferred reconciliation layer: it starts from the main pooled price model's important
  <code>_BID</code> features, then checks whether those same marginal bid rows are usually
  negative bids where the SO pays, or positive bids where the SO receives.</p>
  <p>It should be read as an accounting diagnostic attached to the main SHAP/RF price model,
  not as a separate bid-side price model.</p>
</div>
<p class="note">This diagnostic keeps the general bid/offer SHAP price model as the source of
truth for price importance, filters to the top {TOP_N_BID_FEATURES} <code>_BID</code> drivers,
then joins those same marginal rows back to the original <code>marginal_final_price</code>. For
bid-side rows, negative raw bid prices mean the SO pays; positive raw bid prices mean the SO
receives. The SHAP columns are used only to decide which BID drivers came up in the main price
model; the body of this report focuses on the raw bid-price sign. Shares are tie-weighted using
<code>1 / n_tied_marginal_candidates</code>.</p>
<table>
  <tr><th>Item</th><th>Value</th></tr>
  <tr><td>Source rows</td><td>{len(sp):,}</td></tr>
  <tr><td>Bid-side marginal rows</td><td>{len(bid_rows):,}</td></tr>
  <tr><td>Date range</td><td>{pd.to_datetime(sp['settlementDate']).dt.date.min()} to {pd.to_datetime(sp['settlementDate']).dt.date.max()}</td></tr>
  <tr><td>SHAP profile</td><td>Updated/flex-3: fuel 300 trees depth 16 leaf 5; family/BMU 200 trees depth 24 leaf 3</td></tr>
</table>
{''.join(sections)}
</body></html>"""


def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    df = build_report(sp)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_HTML.write_text(make_html(df, sp), encoding="utf-8")
    print(f"\nSaved -> {OUT_CSV}")
    print(f"Saved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
