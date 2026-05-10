from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

REPORT_DIR = PROJECT_ROOT / "reports" / "full_2023_2025"
TABLE_DIR = REPORT_DIR / "tables"

MAIN_CSV = TABLE_DIR / "shap_price_importance_2023_2025_sided_updated.csv"
YEAR_CSV = TABLE_DIR / "shap_price_importance_year_stability_2023_2025.csv"
REGIME_CSV = TABLE_DIR / "shap_price_importance_regime_stability_2023_2025.csv"
BID_COST_CSV = TABLE_DIR / "bid_side_price_driver_cost_to_so_diagnostic.csv"

OUT_HTML = REPORT_DIR / "canonical" / "shap_price_importance_interpretation_summary_2023_2025.html"
OUT_DIR = TABLE_DIR / "summary_tables"

TOP_N = 12


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(MAIN_CSV)
    year = pd.read_csv(YEAR_CSV)
    regime = pd.read_csv(REGIME_CSV)
    bid_cost = pd.read_csv(BID_COST_CSV)

    numeric_cols = [
        "mean_abs_shap",
        "conditional_mean_shap_when_active",
        "share_positive_shap_when_active",
        "share_negative_shap_when_active",
        "active_rows_in_shap_sample",
        "overall_shap_rank",
        "side_shap_rank",
        "model_train_r2",
        "median_raw_bid_price",
        "weighted_mean_raw_bid_price",
        "share_so_pays_weighted",
        "share_so_receives_weighted",
        "share_zero_bid_weighted",
        "bid_shap_rank",
        "bid_rows",
    ]
    return (
        numeric(main, numeric_cols),
        numeric(year, numeric_cols + ["year"]),
        numeric(regime, numeric_cols),
        numeric(bid_cost, numeric_cols),
    )


def shap_direction(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value > 0:
        return "raises predicted price"
    if value < 0:
        return "suppresses predicted price"
    return "neutral"


def top_overall(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor, block in main.groupby("predictor", sort=False):
        top = block.sort_values("mean_abs_shap", ascending=False).head(TOP_N).copy()
        top["direction"] = top["conditional_mean_shap_when_active"].map(shap_direction)
        rows.append(top)
    cols = [
        "predictor",
        "overall_shap_rank",
        "feature",
        "side",
        "mean_abs_shap",
        "conditional_mean_shap_when_active",
        "direction",
        "share_positive_shap_when_active",
        "share_negative_shap_when_active",
        "active_rows_in_shap_sample",
    ]
    return pd.concat(rows, ignore_index=True)[cols]


def side_summary(main: pd.DataFrame) -> pd.DataFrame:
    block = main.loc[main["side"].isin(["OFFER", "BID"])].copy()
    return (
        block.groupby(["predictor", "side"], as_index=False)
        .agg(
            top20_mean_abs_shap=("mean_abs_shap", lambda s: s.nlargest(20).mean()),
            top20_directional_mean=("conditional_mean_shap_when_active", lambda s: s.loc[s.abs().nlargest(min(20, len(s))).index].mean()),
            n_features=("feature", "count"),
            n_active_features=("active_rows_in_shap_sample", lambda s: int((s > 0).sum())),
        )
        .sort_values(["predictor", "side"])
    )


def persistent_year_drivers(year: pd.DataFrame) -> pd.DataFrame:
    top = year.loc[year["overall_shap_rank"].le(20)].copy()
    rows = []
    for (predictor, feature), grp in top.groupby(["predictor", "feature"], sort=False):
        all_grp = year.loc[(year["predictor"].eq(predictor)) & (year["feature"].eq(feature))]
        signs = []
        for yr in [2023, 2024, 2025]:
            match = all_grp.loc[all_grp["year"].eq(yr)]
            if match.empty or pd.isna(match["conditional_mean_shap_when_active"].iloc[0]):
                signs.append("NA")
            elif match["conditional_mean_shap_when_active"].iloc[0] > 0:
                signs.append("+")
            elif match["conditional_mean_shap_when_active"].iloc[0] < 0:
                signs.append("-")
            else:
                signs.append("0")
        rows.append(
            {
                "predictor": predictor,
                "feature": feature,
                "side": grp["side"].iloc[0],
                "years_in_top20": int(grp["year"].nunique()),
                "top20_years": ", ".join(str(int(y)) for y in sorted(grp["year"].unique())),
                "best_annual_rank": int(all_grp["overall_shap_rank"].min()),
                "mean_abs_shap_mean": float(all_grp["mean_abs_shap"].mean()),
                "direction_pattern_2023_2025": "".join(signs),
                "min_active_rows": int(all_grp["active_rows_in_shap_sample"].min()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["years_in_top20", "mean_abs_shap_mean"], ascending=[False, False]).head(45)


def regime_specialists(regime: pd.DataFrame) -> pd.DataFrame:
    top = regime.loc[regime["overall_shap_rank"].le(20)].copy()
    rows = []
    for (predictor, feature), grp in top.groupby(["predictor", "feature"], sort=False):
        all_grp = regime.loc[(regime["predictor"].eq(predictor)) & (regime["feature"].eq(feature))]
        regimes = grp["regime_label"].tolist()
        signs = []
        for label in ["Low <= p10", "Normal p10-p75", "High p90-p95", "Extreme > p95"]:
            match = all_grp.loc[all_grp["regime_label"].eq(label)]
            if match.empty or pd.isna(match["conditional_mean_shap_when_active"].iloc[0]):
                signs.append("NA")
            elif match["conditional_mean_shap_when_active"].iloc[0] > 0:
                signs.append("+")
            elif match["conditional_mean_shap_when_active"].iloc[0] < 0:
                signs.append("-")
            else:
                signs.append("0")
        rows.append(
            {
                "predictor": predictor,
                "feature": feature,
                "side": grp["side"].iloc[0],
                "regimes_in_top20": len(set(regimes)),
                "top20_regimes": ", ".join(regimes),
                "best_regime_rank": int(all_grp["overall_shap_rank"].min()),
                "mean_abs_shap_mean": float(all_grp["mean_abs_shap"].mean()),
                "direction_pattern_low_normal_high_extreme": " | ".join(signs),
                "min_active_rows": int(all_grp["active_rows_in_shap_sample"].min()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["regimes_in_top20", "mean_abs_shap_mean"], ascending=[False, False]).head(50)


def bid_cost_summary(bid_cost: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor, block in bid_cost.groupby("predictor", sort=False):
        top = block.sort_values("mean_abs_shap", ascending=False).head(TOP_N).copy()
        top["raw_bid_price_read"] = np.select(
            [
                top["weighted_mean_raw_bid_price"].lt(0),
                top["weighted_mean_raw_bid_price"].gt(0),
            ],
            ["SO usually pays", "SO usually receives"],
            default="zero/mixed",
        )
        rows.append(top)
    cols = [
        "predictor",
        "bid_shap_rank",
        "feature",
        "mean_abs_shap",
        "bid_rows",
        "median_raw_bid_price",
        "weighted_mean_raw_bid_price",
        "share_so_pays_weighted",
        "share_so_receives_weighted",
        "raw_bid_price_direction",
        "raw_bid_price_read",
    ]
    return pd.concat(rows, ignore_index=True)[cols]


def key_findings(
    overall: pd.DataFrame,
    persistent: pd.DataFrame,
    regimes: pd.DataFrame,
    bid_cost: pd.DataFrame,
) -> list[str]:
    gen = overall.loc[overall["predictor"].eq("Generation type")]
    top_offer = gen.loc[gen["side"].eq("OFFER")].head(3)["feature"].tolist()
    top_bid = gen.loc[gen["side"].eq("BID")].head(3)["feature"].tolist()

    persistent_3y = persistent.loc[persistent["years_in_top20"].ge(3)].head(6)["feature"].tolist()
    extreme = regimes.loc[
        regimes["top20_regimes"].str.contains("Extreme", na=False)
        & regimes["side"].eq("OFFER")
    ].head(6)["feature"].tolist()
    bid_receive = bid_cost.loc[bid_cost["share_so_receives_weighted"].ge(0.75)].head(5)["feature"].tolist()
    bid_pay = bid_cost.loc[bid_cost["share_so_pays_weighted"].ge(0.75)].head(5)["feature"].tolist()

    return [
        f"Overall generation-type SHAP is dominated by OFFER-side {', '.join(top_offer)} and BID-side {', '.join(top_bid)}.",
        f"The most persistent annual drivers include {', '.join(persistent_3y)}.",
        f"Extreme-price regimes surface OFFER-side scarcity/flexibility drivers such as {', '.join(extreme)}.",
        f"Most leading BID price drivers are positive raw bids where the SO receives, including {', '.join(bid_receive)}.",
        f"A smaller set of important BID drivers are negative raw bids where the SO pays, including {', '.join(bid_pay)}.",
    ]


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
p.note { color: #666; font-size: 13px; }
li { margin: 6px 0; }
code { background: #ECEFF1; padding: 1px 4px; border-radius: 4px; }
</style>
"""


def table(df: pd.DataFrame, n: int | None = None) -> str:
    block = df if n is None else df.head(n)
    return block.round(4).to_html(index=False, border=0)


def pooled_sections(overall: pd.DataFrame) -> str:
    sections = []
    for predictor, block in overall.groupby("predictor", sort=False):
        sections.append(
            f"""
<h3>{predictor}</h3>
{table(block)}"""
        )
    return "\n".join(sections)


def side_sections(sides: pd.DataFrame) -> str:
    sections = []
    for predictor, block in sides.groupby("predictor", sort=False):
        sections.append(
            f"""
<h3>{predictor}</h3>
{table(block)}"""
        )
    return "\n".join(sections)


def make_html(
    overall: pd.DataFrame,
    sides: pd.DataFrame,
    persistent: pd.DataFrame,
    regimes: pd.DataFrame,
    bid_cost: pd.DataFrame,
    findings: list[str],
) -> str:
    finding_html = "\n".join(f"<li>{finding}</li>" for finding in findings)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SHAP Interpretation Summary 2023-2025</title>
{CSS}
</head><body>
<h1>SHAP Price Importance Interpretation Summary: 2023-2025</h1>
<p>This report consolidates the main SHAP, annual stability, price-regime stability, and the
canonical BID raw bid-price diagnostic into dissertation-ready findings. It is an
interpretation layer, not a new model.</p>

<h2>Headline Findings</h2>
<ul>{finding_html}</ul>

<h2>Method Caveats To Use In The Dissertation</h2>
<ul>
  <li>One-hot SHAP values are model associations: when a label-side dummy is active, predicted
  system price moves above or below the model baseline. They should not be worded as clean
  causal unit-offer experiments.</li>
  <li>Regime SHAP explains variation within low, normal, high, or extreme price regimes; it
  does not by itself explain why the market entered that regime.</li>
  <li>BID-side system-price effects and raw bid-price accounting are different quantities.
  Raw bid prices are used for the accounting interpretation: negative bids mean the SO pays,
  while positive bids mean the SO receives.</li>
</ul>

<h2>Overall Pooled SHAP Drivers</h2>
<p class="note">Top whole-period drivers by mean absolute SHAP. Direction is conditional mean
SHAP when the feature is active.</p>
{pooled_sections(overall)}

<h2>Offer vs Bid Summary</h2>
<p class="note">Compact side-level summary from the whole-period model.</p>
{side_sections(sides)}

<h2>Persistent Annual Drivers</h2>
<p class="note">Features recurring in annual top-20 SHAP rankings. Direction pattern is ordered
2023, 2024, 2025.</p>
{table(persistent)}

<h2>Price-Regime Specialists</h2>
<p class="note">Features recurring in regime top-20 SHAP rankings. Direction pattern is ordered
Low, Normal, High, Extreme.</p>
{table(regimes)}

<h2>BID Drivers: Raw Bid-Price Check</h2>
<p class="note">Canonical accounting reconciliation: starts from the main pooled SHAP BID
drivers, then checks whether those same marginal BID rows are usually negative bids where
the SO pays, or positive bids where the SO receives. This is preferred over split
bid-accounting RF or SHAP models because it keeps price importance and accounting
interpretation separate.</p>
{table(bid_cost)}

<h2>Suggested Dissertation Wording</h2>
<p>The pooled SHAP model indicates that system price formation is strongly asymmetric by marginal
side: OFFER-side CCGT and storage-related features are associated with higher predicted prices,
while leading BID-side features are associated with lower-price system states. Annual stability
checks show that these patterns are not solely driven by one calendar year. Price-regime analysis
adds a more nuanced scarcity interpretation: low-price periods are dominated by BID-side features,
whereas high and extreme regimes shift toward OFFER-side scarcity and flexibility features.
Finally, the BID raw bid-price diagnostic shows that price association should be separated from
bid accounting: many important BID-side price drivers are positive raw bids where the SO receives,
while a smaller set of low-price BID drivers are negative raw bids where the SO pays.</p>
</body></html>"""


def main() -> None:
    print("Loading diagnostic CSVs ...")
    main_df, year_df, regime_df, bid_cost_df = load_inputs()

    overall = top_overall(main_df)
    sides = side_summary(main_df)
    persistent = persistent_year_drivers(year_df)
    regimes = regime_specialists(regime_df)
    bid_cost = bid_cost_summary(bid_cost_df)
    findings = key_findings(overall, persistent, regimes, bid_cost)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT_DIR / "interpretation_top_overall.csv", index=False)
    sides.to_csv(OUT_DIR / "interpretation_side_summary.csv", index=False)
    persistent.to_csv(OUT_DIR / "interpretation_persistent_year_drivers.csv", index=False)
    regimes.to_csv(OUT_DIR / "interpretation_regime_specialists.csv", index=False)
    bid_cost.to_csv(OUT_DIR / "interpretation_bid_cost_benefit.csv", index=False)

    OUT_HTML.write_text(
        make_html(overall, sides, persistent, regimes, bid_cost, findings),
        encoding="utf-8",
    )
    print(f"Saved -> {OUT_HTML}")
    print(f"Saved tables -> {OUT_DIR}")


if __name__ == "__main__":
    main()
