from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

SP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "full_2023_2025" / "bid_offer_stack_2023_2025"
    / "bid_offer_stack_energy_2023_2025_niv_marginal_sp_summary.parquet"
)
REPORT_DIR = PROJECT_ROOT / "reports" / "full_2023_2025"
OUT_HTML = REPORT_DIR / "canonical" / "eda_2023_2025_sp_summary.html"
SHAP_IMPORTANCE_CSV = REPORT_DIR / "tables" / "shap_price_importance_2023_2025_sided_updated.csv"

FUEL_COLOURS = {
    "BATTERY":       "#2196F3",
    "CCGT":          "#F44336",
    "OCGT":          "#FF9800",
    "GAS":           "#E91E63",
    "PS":            "#9C27B0",
    "NPSHYD":        "#00BCD4",
    "WIND":          "#4CAF50",
    "SOLAR":         "#FFEB3B",
    "BIOMASS":       "#795548",
    "COAL":          "#212121",
    "DIESEL":        "#607D8B",
    "LOAD RESPONSE": "#FF5722",
    "INTERCONNECTOR":"#009688",
    "IFA":           "#00897B",
    "IFA2":          "#26A69A",
    "BritNed":       "#00ACC1",
    "NEMO":          "#0097A7",
    "Eleclink":      "#00838F",
    "Viking":        "#006064",
    "Moyle":         "#4DB6AC",
    "EastWest":      "#80CBC4",
    "NSL":           "#B2DFDB",
    "Greenlink":     "#E0F2F1",
    "OTHER":         "#9E9E9E",
    "UNKNOWN_NUMERIC":"#BDBDBD",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:900px;">'


def section(title: str, content: str) -> str:
    return f"""
<section>
  <h2>{title}</h2>
  {content}
</section>
"""


def table_html(df: pd.DataFrame) -> str:
    return df.to_html(border=0, classes="tbl", index=True)


def make_sided_label(sp: pd.DataFrame, col: str) -> pd.Series:
    return sp[col].fillna("MISSING").astype(str) + "_" + sp["niv_active_side"].str.upper()


# ── plot functions ────────────────────────────────────────────────────────────

def plot_fuel_bar(sp: pd.DataFrame) -> str:
    counts = (
        sp.groupby("marginal_fuel_type")
        .size()
        .rename("n_candidates")
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    colours = [FUEL_COLOURS.get(f, "#9E9E9E") for f in counts["marginal_fuel_type"]]
    bars = ax.bar(counts["marginal_fuel_type"], counts["n_candidates"], color=colours, edgecolor="white")
    ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    ax.set_xlabel("Fuel type")
    ax.set_ylabel("Marginal candidate rows")
    ax.set_title("Marginal candidates by fuel type — 2023–2025")
    ax.tick_params(axis="x", rotation=35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_niv_side_fuel(sp: pd.DataFrame) -> str:
    ct = (
        sp.groupby(["niv_active_side", "marginal_fuel_type"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ct.T.plot(kind="bar", ax=ax, color=["#1565C0", "#B71C1C"], edgecolor="white")
    ax.set_xlabel("Fuel type")
    ax.set_ylabel("Marginal candidate rows")
    ax.set_title("Marginal fuel type by NIV active side")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="NIV side")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_tied_dist(sp: pd.DataFrame) -> str:
    counts = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"])["n_tied_marginal_candidates"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(counts.index.astype(int), counts.values, color="#5C6BC0", edgecolor="white")
    ax.set_xlabel("Number of co-marginal candidates in SP")
    ax.set_ylabel("Number of SPs")
    ax.set_title("Co-marginal candidate count distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_price_hist(sp: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, col, label, colour in zip(
        axes,
        ["systemBuyPrice", "systemSellPrice", "marginal_final_price"],
        ["System Buy Price", "System Sell Price", "Raw marginal final price"],
        ["#EF5350", "#42A5F5", "#66BB6A"],
    ):
        data = sp[col].dropna()
        ax.hist(data, bins=60, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1, label=f"Median {data.median():.1f}")
        ax.set_xlabel("£/MWh")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.suptitle("Price distributions (SP-level where applicable)", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_marginal_vs_system_distribution(sp: pd.DataFrame) -> str:
    data = sp.dropna(subset=["systemPrice", "marginal_final_price"]).copy()
    lo = data[["systemPrice", "marginal_final_price"]].quantile(0.005).min()
    hi = data[["systemPrice", "marginal_final_price"]].quantile(0.995).max()
    bins = np.linspace(lo, hi, 90)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, subset, title in [
        (axes[0], data, "All marginal rows"),
        (axes[1], data[data["niv_active_side"] == "offer"], "Offer-active"),
        (axes[2], data[data["niv_active_side"] == "bid"], "Bid-active"),
    ]:
        ax.hist(
            subset["systemPrice"],
            bins=bins,
            histtype="step",
            color="#1E40AF",
            linewidth=1.8,
            label="System price",
            density=True,
        )
        ax.hist(
            subset["marginal_final_price"],
            bins=bins,
            histtype="step",
            color="#E65100",
            linewidth=1.8,
            label="Raw marginal final price",
            density=True,
        )
        ax.axvline(0, color="#424242", linestyle=":", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("£/MWh")
        ax.legend(fontsize=8)
    fig.suptitle("Distribution comparison: published system price vs identified marginal action price", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def _legacy_cost_to_so_marginal_vs_system_scatter(sp: pd.DataFrame) -> str:
    data = sp.dropna(subset=["systemPrice", "marginal_final_price"]).copy()
    sample = data.sample(n=min(25000, len(data)), random_state=0)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    offer = sample["niv_active_side"].eq("offer")
    bid = sample["niv_active_side"].eq("bid")
    ax.scatter(
        sample.loc[offer, "marginal_final_price"],
        sample.loc[offer, "systemPrice"],
        s=5,
        alpha=0.20,
        c="#D32F2F",
        linewidths=0,
        label="Offer-active",
    )
    ax.scatter(
        sample.loc[bid, "marginal_final_price"],
        sample.loc[bid, "systemPrice"],
        s=5,
        alpha=0.20,
        c="#1976D2",
        linewidths=0,
        label="Bid-active",
    )
    lo = float(sample[["marginal_final_price", "systemPrice"]].quantile(0.005).min())
    hi = float(sample[["marginal_final_price", "systemPrice"]].quantile(0.995).max())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Raw marginal final price (£/MWh)")
    ax.set_ylabel("Published system price (£/MWh)")
    ax.set_title("Published system price vs identified marginal action price")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_marginal_vs_system_scatter(sp: pd.DataFrame) -> str:
    data = sp.dropna(subset=["systemPrice", "marginal_final_price"]).copy()
    sample = data.sample(n=min(25000, len(data)), random_state=0)
    lo = float(sample[["marginal_final_price", "systemPrice"]].quantile(0.005).min())
    hi = float(sample[["marginal_final_price", "systemPrice"]].quantile(0.995).max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, side, colour, title in [
        (axes[0], "offer", "#D32F2F", "Offer-active periods"),
        (axes[1], "bid", "#1976D2", "Bid-active periods"),
    ]:
        subset = sample[sample["niv_active_side"].eq(side)]
        ax.scatter(
            subset["marginal_final_price"],
            subset["systemPrice"],
            s=5,
            alpha=0.20,
            c=colour,
            linewidths=0,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("Raw marginal final price (GBP/MWh)")
    axes[0].set_ylabel("Published system price (GBP/MWh)")
    fig.suptitle("Published system price compared with identified marginal action price", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_price_gap_hist(sp: pd.DataFrame) -> str:
    data = sp.dropna(subset=["systemPrice", "marginal_final_price"]).copy()
    data["price_gap"] = data["systemPrice"] - data["marginal_final_price"]
    lo = data["price_gap"].quantile(0.005)
    hi = data["price_gap"].quantile(0.995)
    bins = np.linspace(lo, hi, 90)

    fig, ax = plt.subplots(figsize=(10, 4))
    for side, colour, label in [
        ("offer", "#EF5350", "Offer-active"),
        ("bid", "#42A5F5", "Bid-active"),
    ]:
        subset = data.loc[data["niv_active_side"].eq(side), "price_gap"]
        ax.hist(subset, bins=bins, alpha=0.48, density=True, color=colour, label=label)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("System price - raw marginal final price (£/MWh)")
    ax.set_title("Gap between published system price and identified marginal action price")
    ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_price_box(sp: pd.DataFrame) -> str:
    order = (
        sp.groupby("marginal_fuel_type")["marginal_final_price"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [sp.loc[sp["marginal_fuel_type"] == f, "marginal_final_price"].dropna().values for f in order]
    colours = [FUEL_COLOURS.get(f, "#9E9E9E") for f in order]
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("£/MWh")
    ax.set_title("Raw marginal final price by fuel type — IQR, no outliers")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_monthly_fuel(sp: pd.DataFrame) -> str:
    sp2 = sp.copy()
    sp2["month"] = sp2["settlementDate"].dt.to_period("M").astype(str)
    top_fuels = (
        sp2.groupby("marginal_fuel_type").size()
        .sort_values(ascending=False).head(7).index.tolist()
    )
    sp2["fuel_grp"] = sp2["marginal_fuel_type"].where(sp2["marginal_fuel_type"].isin(top_fuels), "OTHER")
    ct = (
        sp2.groupby(["month", "fuel_grp"])
        .size()
        .unstack(fill_value=0)
    )
    ct = ct[ct.sum().sort_values(ascending=False).index]
    colours = [FUEL_COLOURS.get(f, "#9E9E9E") for f in ct.columns]
    fig, ax = plt.subplots(figsize=(14, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=colours, edgecolor="white", width=0.85)
    ax.set_xlabel("Month")
    ax.set_ylabel("Marginal candidate rows")
    ax.set_title("Monthly marginal candidate mix — top 7 fuel types")
    ax.legend(title="Fuel type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_niv_scatter(sp: pd.DataFrame) -> str:
    sp_unique = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    offer = sp_unique[sp_unique["niv_active_side"] == "offer"]
    bid = sp_unique[sp_unique["niv_active_side"] == "bid"]
    ax.scatter(offer["niv_volume"], offer["systemPrice"], alpha=0.15, s=4, color="#EF5350", label="Offer (short)")
    ax.scatter(bid["niv_volume"], bid["systemPrice"], alpha=0.15, s=4, color="#42A5F5", label="Bid (long)")
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("NIV (MWh)")
    ax.set_ylabel("System Price (£/MWh)")
    ax.set_title("System price vs NIV — 2023–2025")
    ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_stack_rank(sp: pd.DataFrame) -> str:
    offer = sp[sp["niv_active_side"] == "offer"].copy()
    order = (
        offer.groupby("marginal_fuel_type")["marginal_stack_rank"]
        .median().sort_values(ascending=False).index.tolist()
    )
    data = [offer.loc[offer["marginal_fuel_type"] == f, "marginal_stack_rank"].dropna().values for f in order]
    colours = [FUEL_COLOURS.get(f, "#9E9E9E") for f in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Stack rank (offer side)")
    ax.set_title("Stack rank at margin by fuel type — offer-active SPs only")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_battery_timeseries(sp: pd.DataFrame) -> str:
    """Monthly BATTERY share of marginal candidates."""
    sp2 = sp.copy()
    sp2["month"] = sp2["settlementDate"].dt.to_period("M").astype(str)
    monthly = (
        sp2.groupby("month")
        .apply(lambda g: (g["marginal_fuel_type"] == "BATTERY").sum() / len(g))
        .rename("battery_share")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(monthly["month"], monthly["battery_share"] * 100, color="#2196F3", marker="o", markersize=4)
    ax.set_xlabel("Month")
    ax.set_ylabel("BATTERY share of marginal candidates (%)")
    ax.set_title("Monthly BATTERY share at margin — 2023–2025")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()
    return fig_to_b64(fig)


# ── summary stats ─────────────────────────────────────────────────────────────

def summary_stats(sp: pd.DataFrame) -> str:
    sp_unique = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"])
    rows = [
        ("Settlement dates", f"{sp['settlementDate'].dt.date.min()} to {sp['settlementDate'].dt.date.max()}"),
        ("Total SPs covered", f"{len(sp_unique):,}"),
        ("Total marginal candidate rows", f"{len(sp):,}"),
        ("SPs with single marginal (n_tied=1)", f"{(sp_unique['n_tied_marginal_candidates'] == 1).sum():,}"),
        ("SPs with co-marginals (n_tied>1)", f"{(sp_unique['n_tied_marginal_candidates'] > 1).sum():,}"),
        ("Max co-marginals in one SP", f"{sp_unique['n_tied_marginal_candidates'].max()}"),
        ("Offer-active SPs", f"{(sp_unique['niv_active_side'] == 'offer').sum():,}"),
        ("Bid-active SPs", f"{(sp_unique['niv_active_side'] == 'bid').sum():,}"),
        ("Median system price £/MWh", f"{sp_unique['systemPrice'].median():.2f}"),
        ("Median NIV MWh", f"{sp_unique['niv_volume'].median():.1f}"),
        ("Unique fuel types at margin", f"{sp['marginal_fuel_type'].nunique()}"),
        ("Unique BMUs at margin", f"{sp['marginal_bmu_id'].nunique():,}"),
    ]
    html = "<table class='tbl'><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
    for k, v in rows:
        html += f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
    html += "</tbody></table>"
    return html


def _legacy_pooled_fuel_table(sp: pd.DataFrame) -> str:
    ct = sp.groupby("marginal_fuel_type").size().rename("n_candidates")
    sp_ct = sp.groupby("marginal_fuel_type").apply(
        lambda g: g[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    ).rename("n_sps_present")
    price_stats = sp.groupby("marginal_fuel_type")["marginal_final_price"].agg(
        median_price="median", p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9)
    ).round(2)
    tbl = pd.concat([ct, sp_ct, price_stats], axis=1).sort_values("n_candidates", ascending=False)
    tbl.index.name = "fuel_type"
    tbl.columns = ["Candidates", "SPs present", "Median £/MWh", "P10 £/MWh", "P90 £/MWh"]
    return tbl.to_html(border=0, classes="tbl")


def fuel_table(sp: pd.DataFrame) -> str:
    ct = sp.groupby("marginal_fuel_type").size().rename("n_candidates")
    sp_ct = sp.groupby("marginal_fuel_type").apply(
        lambda g: g[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    ).rename("n_sps_present")

    side_counts = (
        sp.pivot_table(
            index="marginal_fuel_type",
            columns="niv_active_side",
            values="systemPrice",
            aggfunc="size",
            fill_value=0,
        )
        .rename(columns={"bid": "bid_count", "offer": "offer_count"})
    )
    side_medians = (
        sp.pivot_table(
            index="marginal_fuel_type",
            columns="niv_active_side",
            values="marginal_final_price",
            aggfunc="median",
        )
        .rename(columns={"bid": "median_raw_bid_price", "offer": "median_raw_offer_price"})
        .round(2)
    )

    tbl = pd.concat([ct, side_counts, sp_ct, side_medians], axis=1).sort_values(
        "n_candidates", ascending=False
    )
    for col in ["bid_count", "offer_count"]:
        if col not in tbl.columns:
            tbl[col] = 0
    for col in ["median_raw_bid_price", "median_raw_offer_price"]:
        if col not in tbl.columns:
            tbl[col] = np.nan
    tbl = tbl[
        [
            "n_candidates",
            "bid_count",
            "offer_count",
            "n_sps_present",
            "median_raw_bid_price",
            "median_raw_offer_price",
        ]
    ]
    tbl.index.name = "fuel_type"
    tbl.columns = [
        "Total candidates",
        "Bid count",
        "Offer count",
        "SPs present",
        "Median raw bid price GBP/MWh",
        "Median raw offer price GBP/MWh",
    ]
    return tbl.to_html(border=0, classes="tbl")


def key_findings_callout() -> str:
    return """
<div class="callout">
  <p>CCGT is the dominant marginal candidate on both sides (~43% of rows).</p>
  <p>BATTERY is second by frequency (~16%) but its SHAP impact is notably lower than its frequency, consistent with price-taker behaviour.</p>
  <p>WIND_BID is rare (rank 5 by count) but highly impactful (SHAP rank 2, mean SHAP when active = -56.84).</p>
  <p>Raw marginal final price differs substantially from published SBP, motivating SBP as the regression target.</p>
</div>
"""


def coverage_note() -> str:
    return (
        '<p class="note">Theoretical SP count for 2023-2025 is 52,608 '
        '(3 years x 17,520). Coverage of 50,684 represents 96.3%; the ~1,924 '
        'missing SPs are NIV=0 periods and SPs with no energy-only accepted actions, '
        'excluded by design.</p>'
    )


def load_shap_importance() -> pd.DataFrame:
    if not SHAP_IMPORTANCE_CSV.exists():
        return pd.DataFrame()
    imp = pd.read_csv(SHAP_IMPORTANCE_CSV)
    keep = [
        "source_column",
        "feature",
        "mean_abs_shap",
        "conditional_mean_shap_when_active",
        "mdi",
        "overall_shap_rank",
        "side_shap_rank",
        "active_rows_in_shap_sample",
    ]
    imp = imp[[c for c in keep if c in imp.columns]].copy()
    for col in [
        "mean_abs_shap",
        "conditional_mean_shap_when_active",
        "mdi",
        "overall_shap_rank",
        "side_shap_rank",
        "active_rows_in_shap_sample",
    ]:
        if col in imp.columns:
            imp[col] = pd.to_numeric(imp[col], errors="coerce")
    return imp


def classify_frequency_importance(freq_rank: int, shap_rank: float) -> str:
    shap_rank_is_top = pd.notna(shap_rank) and shap_rank <= 20
    if freq_rank <= 20 and shap_rank_is_top:
        return "frequent and important"
    if freq_rank <= 20:
        return "frequent but lower-importance"
    if shap_rank_is_top:
        return "less frequent but important"
    return "lower frequency / lower importance"


def marginal_setter_frequency_table(
    sp: pd.DataFrame,
    source_column: str,
    label: str,
    shap_imp: pd.DataFrame,
    top_n: int = 25,
) -> str:
    feature_col = f"{source_column}_sided"
    block = sp.copy()
    block[feature_col] = make_sided_label(block, source_column)
    total_sps = block[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]

    freq = (
        block.groupby(feature_col)
        .agg(
            count=("systemPrice", "size"),
            median_system_price=("systemPrice", "median"),
            median_raw_marginal_price=("marginal_final_price", "median"),
        )
        .reset_index()
        .rename(columns={feature_col: "label_x_side"})
    )
    unique_sps = (
        block.groupby(feature_col)[["settlementDate", "settlementPeriod"]]
        .apply(lambda g: g.drop_duplicates().shape[0])
        .rename("unique_marginal_periods")
        .reset_index()
        .rename(columns={feature_col: "label_x_side"})
    )
    freq = freq.merge(unique_sps, on="label_x_side", how="left")
    freq["share_of_marginal_periods"] = freq["unique_marginal_periods"] / total_sps
    freq["frequency_rank"] = freq["count"].rank(method="first", ascending=False).astype(int)

    if not shap_imp.empty:
        imp = shap_imp.loc[shap_imp["source_column"].eq(source_column)].rename(
            columns={"feature": "label_x_side"}
        )
        freq = freq.merge(imp, on="label_x_side", how="left")
    else:
        freq["mean_abs_shap"] = np.nan
        freq["conditional_mean_shap_when_active"] = np.nan
        freq["mdi"] = np.nan
        freq["overall_shap_rank"] = np.nan

    freq["frequency_importance_read"] = [
        classify_frequency_importance(rank, shap_rank)
        for rank, shap_rank in zip(freq["frequency_rank"], freq["overall_shap_rank"])
    ]
    freq = freq.sort_values("frequency_rank").head(top_n)
    show_cols = [
        "frequency_rank",
        "label_x_side",
        "count",
        "unique_marginal_periods",
        "share_of_marginal_periods",
        "median_system_price",
        "median_raw_marginal_price",
        "mdi",
        "mean_abs_shap",
        "conditional_mean_shap_when_active",
        "overall_shap_rank",
        "frequency_importance_read",
    ]
    freq = freq[[c for c in show_cols if c in freq.columns]].copy()
    freq = freq.rename(
        columns={
            "frequency_rank": "Rank",
            "label_x_side": "Label x side",
            "count": "Count",
            "unique_marginal_periods": "Unique marginal periods",
            "share_of_marginal_periods": "Share of marginal periods",
            "median_system_price": "Median system price",
            "median_raw_marginal_price": "Median raw marginal price",
            "mdi": "RF importance (MDI)",
            "mean_abs_shap": "Mean |SHAP|",
            "conditional_mean_shap_when_active": "Mean SHAP when active",
            "overall_shap_rank": "SHAP rank",
            "frequency_importance_read": "Frequency / importance read",
        }
    )
    return f"<h3>{label}</h3>" + freq.round(4).to_html(index=False, border=0, classes="tbl")


def marginal_setter_frequency_section(sp: pd.DataFrame) -> str:
    shap_imp = load_shap_importance()
    note = (
        "<p class='note'>Frequency is not importance. Count and share show how often a "
        "label-side combination is marginal; RF importance and mean |SHAP| come from the main "
        "sided SHAP/RF model where available. This separates frequent but ordinary setters "
        "from rarer high-impact setters.</p>"
    )
    metric_key = (
        "<p class='note'>_BID / _OFFER suffixes denote the side the unit was acting on "
        "(bid = long-system action; offer = short-system action).<br>"
        "\"RF importance (MDI)\" = Mean Decrease in Impurity from the Random Forest.<br>"
        "\"Mean |SHAP|\" = unsigned magnitude of contribution to predicted SBP.<br>"
        "\"Mean SHAP when active\" = signed direction: negative pulls SBP down, positive pulls SBP up.</p>"
    )
    tables = [
        marginal_setter_frequency_table(sp, "marginal_fuel_type", "Generation type", shap_imp),
        marginal_setter_frequency_table(sp, "marginal_family_id", "Family ID", shap_imp),
        marginal_setter_frequency_table(sp, "marginal_bmu_id", "BMU ID", shap_imp),
    ]
    return note + metric_key + "\n".join(tables)


def marginal_vs_system_gap_table(sp: pd.DataFrame) -> str:
    data = sp.dropna(subset=["systemPrice", "marginal_final_price"]).copy()
    data["price_gap"] = data["systemPrice"] - data["marginal_final_price"]
    data["abs_price_gap"] = data["price_gap"].abs()

    rows = []
    for label, subset in [("All", data)] + [
        (side.title(), data[data["niv_active_side"].eq(side)])
        for side in ["offer", "bid"]
    ]:
        rows.append(
            {
                "Side": label,
                "Rows": len(subset),
                "Median SBP": subset["systemPrice"].median(),
                "Median raw marginal price": subset["marginal_final_price"].median(),
                "Median |gap|": subset["abs_price_gap"].median(),
                "Share within £1": subset["abs_price_gap"].le(1).mean(),
                "Share within £5": subset["abs_price_gap"].le(5).mean(),
                "Share within £10": subset["abs_price_gap"].le(10).mean(),
            }
        )
    tbl = pd.DataFrame(rows)
    share_10_col = [c for c in tbl.columns if c.startswith("Share within") and c.endswith("10")][0]
    tbl = tbl[
        ["Side", "Rows", "Median SBP", "Median raw marginal price", "Median |gap|", share_10_col]
    ].rename(columns={share_10_col: "Share within &pound;10"})
    return tbl.round(4).to_html(index=False, border=0, classes="tbl")


def marginal_vs_system_interpretation() -> str:
    return (
        "<p>The raw marginal final price is not the same thing as the published SBP. "
        "Bid-active periods are much less likely "
        "to sit close to SBP than offer-active periods. This supports using SBP as the model "
        "target rather than treating the raw marginal price as the settled price.</p>"
    )


# ── HTML assembly ─────────────────────────────────────────────────────────────

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #212121; }
  h1   { border-bottom: 3px solid #1565C0; padding-bottom: 8px; color: #1565C0; }
  h2   { margin-top: 40px; color: #283593; border-left: 4px solid #5C6BC0;
         padding-left: 10px; }
  h3   { margin-top: 24px; color: #37474F; }
  section { margin-bottom: 40px; }
  .tbl { border-collapse: collapse; font-size: 13px; width: 100%; }
  .tbl th { background: #1565C0; color: white; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
  .tbl tr:nth-child(even) td { background: #f5f5f5; }
  img  { display: block; margin: 12px 0; border-radius: 6px;
         box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  p.note { font-size: 12px; color: #757575; font-style: italic; }
  .callout { background: #f5f7fa; border-left: 4px solid #1565C0;
             padding: 12px 14px; margin: 18px 0 24px 0; }
  .callout p { margin: 6px 0; }
</style>
"""


def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    print(f"  {len(sp):,} rows, {sp[['settlementDate','settlementPeriod']].drop_duplicates().shape[0]:,} unique SPs")

    print("Building charts ...")
    b64_fuel_bar   = plot_fuel_bar(sp)
    b64_niv_fuel   = plot_niv_side_fuel(sp)
    b64_tied       = plot_tied_dist(sp)
    b64_price_hist = plot_price_hist(sp)
    b64_marginal_scatter = plot_marginal_vs_system_scatter(sp)
    b64_price_box  = plot_price_box(sp)
    b64_monthly    = plot_monthly_fuel(sp)
    b64_scatter    = plot_niv_scatter(sp)
    b64_rank       = plot_stack_rank(sp)
    b64_battery_ts = plot_battery_timeseries(sp)

    print("Assembling HTML ...")
    body = f"""
{CSS}
<h1>EDA — 2023–2025 NIV-Marginal SP Summary</h1>
<p>Source: <code>bid_offer_stack_energy_2023_2025_niv_marginal_sp_summary.parquet</code><br>
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>

{key_findings_callout()}

{section("1. Overview", summary_stats(sp) + coverage_note())}

{section("2. Marginal fuel type breakdown",
    fuel_table(sp)
    + img_tag(b64_fuel_bar)
    + '<p class="note">One row per (SP, BMU) at the top accepted price. SPs with co-marginals contribute multiple rows.</p>'
)}

{section("3. Fuel type by NIV active side",
    img_tag(b64_niv_fuel)
    + '<p class="note">Offer-active = short system (NIV &gt; 0); bid-active = long system (NIV &lt; 0).</p>'
)}

{section("4. Top marginal setter frequency",
    marginal_setter_frequency_section(sp)
)}

{section("5. Co-marginal distribution",
    img_tag(b64_tied)
    + '<p class="note">Count of distinct BMUs sharing the top accepted price in each SP.</p>'
)}

{section("6. Price distributions",
    img_tag(b64_price_hist)
)}

{section("7. Marginal action price vs published system price",
    img_tag(b64_marginal_scatter)
)}

{section("8. Marginal price by fuel type",
    img_tag(b64_price_box)
    + '<p class="note">Raw marginal final price: bid-side prices retain their original signs. Outliers hidden.</p>'
)}

{section("9. Monthly marginal mix",
    img_tag(b64_monthly)
)}

{section("10. BATTERY share at margin over time",
    img_tag(b64_battery_ts)
    + '<p class="note">Monthly BATTERY share as % of all marginal candidate rows. Captures growing BESS penetration.</p>'
)}

{section("11. System price vs NIV",
    img_tag(b64_scatter)
    + '<p class="note">Each point is a unique SP. Offer = short system (NIV &gt; 0); Bid = long system (NIV &lt; 0).</p>'
)}

{section("12. Stack rank at margin (offer-active SPs)",
    img_tag(b64_rank)
    + '<p class="note">Higher stack rank = deeper into the merit order = more accepted offers above the marginal unit.</p>'
)}
"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>EDA 2023-2025</title></head><body>{body}</body></html>", encoding="utf-8")
    print(f"Saved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
