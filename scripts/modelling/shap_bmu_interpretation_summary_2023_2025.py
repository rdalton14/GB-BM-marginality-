from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

REPORT_DIR = PROJECT_ROOT / "reports" / "full_2023_2025"
TABLE_DIR  = REPORT_DIR / "tables"

MAIN_CSV     = TABLE_DIR / "shap_bmu_price_importance_2023_2025.csv"
YEAR_CSV     = TABLE_DIR / "shap_bmu_year_stability_2023_2025.csv"
REGIME_CSV   = TABLE_DIR / "shap_bmu_regime_stability_2023_2025.csv"
BID_COST_CSV = TABLE_DIR / "bid_side_bmu_cost_to_so_diagnostic.csv"

OUT_HTML = REPORT_DIR / "canonical" / "bmu" / "shap_bmu_interpretation_summary_2023_2025.html"
OUT_DIR  = TABLE_DIR / "summary_tables" / "bmu"

TOP_N = 15


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _num_cols = [
        "mean_abs_shap", "conditional_mean_shap_when_active",
        "share_positive_shap_when_active", "share_negative_shap_when_active",
        "active_rows_in_shap_sample", "overall_shap_rank", "side_shap_rank",
        "model_train_r2", "median_raw_bid_price", "weighted_mean_raw_bid_price",
        "share_so_pays_weighted", "share_so_receives_weighted",
        "share_zero_bid_weighted", "bid_shap_rank", "bid_rows",
    ]
    main     = numeric(pd.read_csv(MAIN_CSV), _num_cols)
    year     = numeric(pd.read_csv(YEAR_CSV), _num_cols + ["year"])
    regime   = numeric(pd.read_csv(REGIME_CSV), _num_cols)
    bid_cost = numeric(pd.read_csv(BID_COST_CSV), _num_cols)
    return main, year, regime, bid_cost


def shap_direction(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    return "raises predicted price" if value > 0 else "suppresses predicted price" if value < 0 else "neutral"


def top_overall(main: pd.DataFrame) -> pd.DataFrame:
    top = main.sort_values("mean_abs_shap", ascending=False).head(TOP_N * 2).copy()
    top["direction"] = top["conditional_mean_shap_when_active"].map(shap_direction)
    cols = [
        "overall_shap_rank", "feature", "fuel_type", "side",
        "mean_abs_shap", "conditional_mean_shap_when_active", "direction",
        "share_positive_shap_when_active", "share_negative_shap_when_active",
        "active_rows_in_shap_sample",
    ]
    return top[cols]


def side_summary(main: pd.DataFrame) -> pd.DataFrame:
    block = main.loc[main["side"].isin(["OFFER", "BID"])].copy()
    return (
        block.groupby("side", as_index=False)
        .agg(
            top20_mean_abs_shap=("mean_abs_shap", lambda s: s.nlargest(20).mean()),
            top20_directional_mean=("conditional_mean_shap_when_active", lambda s: s.loc[s.abs().nlargest(min(20, len(s))).index].mean()),
            n_features=("feature", "count"),
            n_active_features=("active_rows_in_shap_sample", lambda s: int((s > 0).sum())),
        )
        .sort_values("side")
    )


def persistent_year_drivers(year: pd.DataFrame) -> pd.DataFrame:
    top = year.loc[year["overall_shap_rank"].le(20)].copy()
    rows = []
    for (feature,), grp in top.groupby(["feature"], sort=False):
        all_grp = year.loc[year["feature"].eq(feature)]
        fuel    = all_grp["fuel_type"].dropna().mode().iloc[0] if not all_grp["fuel_type"].dropna().empty else "UNKNOWN"
        side    = grp["side"].iloc[0]
        signs   = []
        for yr in [2023, 2024, 2025]:
            match = all_grp.loc[all_grp["year"].eq(yr)]
            if match.empty or pd.isna(match["conditional_mean_shap_when_active"].iloc[0]):
                signs.append("NA")
            else:
                v = match["conditional_mean_shap_when_active"].iloc[0]
                signs.append("+" if v > 0 else ("-" if v < 0 else "0"))
        rows.append(
            {
                "feature":                  feature,
                "fuel_type":                fuel,
                "side":                     side,
                "years_in_top20":           int(grp["year"].nunique()),
                "top20_years":              ", ".join(str(int(y)) for y in sorted(grp["year"].unique())),
                "best_annual_rank":         int(all_grp["overall_shap_rank"].min()),
                "mean_abs_shap_mean":       float(all_grp["mean_abs_shap"].mean()),
                "direction_pattern_2023_2025": "".join(signs),
                "min_active_rows":          int(all_grp["active_rows_in_shap_sample"].min()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["years_in_top20", "mean_abs_shap_mean"], ascending=[False, False])
        .head(50)
    )


def regime_specialists(regime: pd.DataFrame) -> pd.DataFrame:
    regime_order  = ["low_le_p10", "normal_p10_p75", "high_p90_p95", "extreme_gt_p95"]
    regime_labels = {
        "low_le_p10":     "Low <= p10",
        "normal_p10_p75": "Normal p10–p75",
        "high_p90_p95":   "High p90–p95",
        "extreme_gt_p95": "Extreme > p95",
    }
    top = regime.loc[regime["overall_shap_rank"].le(20)].copy()
    rows = []
    for (feature,), grp in top.groupby(["feature"], sort=False):
        all_grp = regime.loc[regime["feature"].eq(feature)]
        fuel    = all_grp["fuel_type"].dropna().mode().iloc[0] if not all_grp["fuel_type"].dropna().empty else "UNKNOWN"
        side    = grp["side"].iloc[0]
        r_list  = grp["regime_label"].tolist()
        signs   = []
        for r in regime_order:
            match = all_grp.loc[all_grp["regime"].eq(r)] if "regime" in all_grp.columns else pd.DataFrame()
            if match.empty or pd.isna(match["conditional_mean_shap_when_active"].iloc[0]):
                signs.append("NA")
            else:
                v = match["conditional_mean_shap_when_active"].iloc[0]
                signs.append("+" if v > 0 else ("-" if v < 0 else "0"))
        rows.append(
            {
                "feature":          feature,
                "fuel_type":        fuel,
                "side":             side,
                "regimes_in_top20": len(set(r_list)),
                "top20_regimes":    ", ".join(r_list),
                "best_regime_rank": int(all_grp["overall_shap_rank"].min()),
                "mean_abs_shap_mean": float(all_grp["mean_abs_shap"].mean()),
                "direction_pattern_low_normal_high_extreme": " | ".join(signs),
                "min_active_rows":  int(all_grp["active_rows_in_shap_sample"].min()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["regimes_in_top20", "mean_abs_shap_mean"], ascending=[False, False])
        .head(55)
    )


def bid_cost_summary(bid_cost: pd.DataFrame) -> pd.DataFrame:
    top = bid_cost.sort_values("mean_abs_shap", ascending=False).head(TOP_N).copy()
    top["raw_bid_price_read"] = np.select(
        [top["weighted_mean_raw_bid_price"].lt(0), top["weighted_mean_raw_bid_price"].gt(0)],
        ["SO usually pays", "SO usually receives"],
        default="zero/mixed",
    )
    cols = [
        "bid_shap_rank", "feature", "fuel_type",
        "mean_abs_shap", "bid_rows", "median_raw_bid_price",
        "weighted_mean_raw_bid_price", "share_so_pays_weighted",
        "share_so_receives_weighted", "raw_bid_price_direction", "raw_bid_price_read",
    ]
    return top[cols]


def key_findings(
    overall: pd.DataFrame,
    persistent: pd.DataFrame,
    regimes: pd.DataFrame,
    bid_cost: pd.DataFrame,
) -> list[str]:
    offer_top = overall.loc[overall["side"].eq("OFFER")].head(3)
    bid_top   = overall.loc[overall["side"].eq("BID")].head(3)

    def fmt(row: pd.Series) -> str:
        return f"{row['feature']} ({row['fuel_type']})"

    top_offer_str = ", ".join(fmt(r) for _, r in offer_top.iterrows())
    top_bid_str   = ", ".join(fmt(r) for _, r in bid_top.iterrows())

    persistent_3y = persistent.loc[persistent["years_in_top20"].ge(3)].head(5)
    persistent_str = ", ".join(
        f"{r['feature']} ({r['fuel_type']})" for _, r in persistent_3y.iterrows()
    ) or "none in top 5 across all 3 years"

    extreme = regimes.loc[
        regimes["top20_regimes"].str.contains("Extreme", na=False)
        & regimes["side"].eq("OFFER")
    ].head(5)
    extreme_str = ", ".join(
        f"{r['feature']} ({r['fuel_type']})" for _, r in extreme.iterrows()
    ) or "none identified"

    so_receives = bid_cost.loc[bid_cost["share_so_receives_weighted"].ge(0.75)].head(4)
    so_pays     = bid_cost.loc[bid_cost["share_so_pays_weighted"].ge(0.75)].head(4)
    receives_str = ", ".join(
        f"{r['feature']} ({r['fuel_type']})" for _, r in so_receives.iterrows()
    ) or "none with ≥75% share"
    pays_str = ", ".join(
        f"{r['feature']} ({r['fuel_type']})" for _, r in so_pays.iterrows()
    ) or "none with ≥75% share"

    return [
        f"The main OFFER-side BMU price drivers are {top_offer_str}; the main BID-side drivers are {top_bid_str}.",
        f"The most persistent BMU drivers across all three years (2023–2025) include {persistent_str}.",
        f"In extreme-price regimes (> p95), the leading OFFER-side BMUs are {extreme_str}.",
        f"Most leading BID-side BMUs are positive raw bids where the SO receives, including {receives_str}.",
        f"A smaller set of important BID-side BMUs are negative raw bids where the SO pays, including {pays_str}.",
    ]


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
p.note { color: #666; font-size: 13px; }
li { margin: 6px 0; }
code { background: #ECEFF1; padding: 1px 4px; border-radius: 4px; }
</style>
"""


def tbl(df: pd.DataFrame, n: int | None = None) -> str:
    block = df if n is None else df.head(n)
    return block.round(4).to_html(index=False, border=0)


def make_html(
    overall: pd.DataFrame,
    sides: pd.DataFrame,
    persistent: pd.DataFrame,
    regimes: pd.DataFrame,
    bid_cost: pd.DataFrame,
    findings: list[str],
) -> str:
    finding_html = "\n".join(f"<li>{f}</li>" for f in findings)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>BMU SHAP Interpretation Summary 2023–2025</title>
{CSS}
</head><body>
<h1>BMU ID SHAP Interpretation Summary: 2023–2025</h1>
<p>Consolidates the main BMU SHAP analysis, annual stability, price-regime stability, and
the BMU bid-side accounting diagnostic into dissertation-ready findings.
This is an interpretation layer, not a new model. Fuel type is annotated throughout — it
does not enter any of the underlying RF models.</p>

<h2>Headline Findings</h2>
<ul>{finding_html}</ul>

<h2>Method Caveats</h2>
<ul>
  <li>One-hot SHAP values are model associations: when a BMU×side dummy is active, predicted
  system price moves above or below the model baseline. They should not be worded as clean
  causal unit-offer experiments.</li>
  <li>Regime SHAP explains variation within low, normal, high, or extreme price regimes; it
  does not explain why the market entered that regime.</li>
  <li>BID-side system-price effects and raw bid-price accounting are different quantities.
  The BID raw bid-price diagnostic gives the accounting interpretation: negative bids mean
  the SO pays, positive bids mean the SO receives.</li>
  <li>Fuel type annotation is derived from the modal value of <code>marginal_fuel_type</code>
  per BMU in the SP summary. A small number of BMUs may have mixed or ambiguous fuel types.</li>
</ul>

<h2>Top Pooled BMU Drivers</h2>
<p class="note">Top whole-period BMU×side drivers by mean absolute SHAP, with fuel type annotation.</p>
{tbl(overall)}

<h2>Offer vs Bid Side Summary</h2>
<p class="note">Compact side-level summary from the whole-period BMU model.</p>
{tbl(sides)}

<h2>Persistent Annual BMU Drivers</h2>
<p class="note">BMUs recurring in annual top-20 SHAP rankings. Direction pattern ordered 2023, 2024, 2025.</p>
{tbl(persistent)}

<h2>Price-Regime Specialists</h2>
<p class="note">BMUs recurring in regime top-20 SHAP rankings. Direction pattern: Low | Normal | High | Extreme.</p>
{tbl(regimes)}

<h2>BID-side BMUs: Raw Bid-Price Check</h2>
<p class="note">Canonical bid-side accounting: starts from the main pooled BMU SHAP BID drivers,
then checks whether those marginal BID rows carry negative or positive raw bid prices.
Negative = SO pays; positive = SO receives.</p>
{tbl(bid_cost)}

<h2>Suggested Dissertation Wording</h2>
<p>The BMU-level SHAP model shows that system price formation is strongly asymmetric by
marginal side. OFFER-side BMUs are associated with higher predicted prices, typically
belonging to CCGT, storage, and flexible generation types. BID-side BMU effects are
associated with lower-price system states, with the leading BID-side drivers tending to
carry positive raw bid prices where the System Operator receives payment.
Annual stability checks confirm that the key BMUs are not artefacts of a single year.
Price-regime analysis shows that extreme-price periods shift toward a narrower set of
high-value OFFER-side BMUs, while low-price periods are dominated by BID-side features.
The bid-price accounting diagnostic then separates price importance from cost accounting:
the most important BID-side BMUs by SHAP are predominantly positive bids where the SO
receives, while a smaller set of negative-bid BMUs represent cost obligations to the SO.</p>
</body></html>"""


def main() -> None:
    print("Loading BMU diagnostic CSVs ...")
    main_df, year_df, regime_df, bid_cost_df = load_inputs()

    overall   = top_overall(main_df)
    sides     = side_summary(main_df)
    persistent = persistent_year_drivers(year_df)
    regimes   = regime_specialists(regime_df)
    bid_cost  = bid_cost_summary(bid_cost_df)
    findings  = key_findings(overall, persistent, regimes, bid_cost)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT_DIR / "bmu_interpretation_top_overall.csv", index=False)
    sides.to_csv(OUT_DIR / "bmu_interpretation_side_summary.csv", index=False)
    persistent.to_csv(OUT_DIR / "bmu_interpretation_persistent_year_drivers.csv", index=False)
    regimes.to_csv(OUT_DIR / "bmu_interpretation_regime_specialists.csv", index=False)
    bid_cost.to_csv(OUT_DIR / "bmu_interpretation_bid_cost_benefit.csv", index=False)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(make_html(overall, sides, persistent, regimes, bid_cost, findings), encoding="utf-8")
    print(f"Saved -> {OUT_HTML}")
    print(f"Saved tables -> {OUT_DIR}")


if __name__ == "__main__":
    main()
