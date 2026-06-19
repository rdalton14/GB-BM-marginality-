from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
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
UNIT_CONDITIONAL_CSV = REPORT_DIR / "tables" / "bmu_side_conditional_system_price_distribution_2023_2025.csv"

FUEL_COLOURS = {
    "BATTERY":        "#2196F3",
    "CCGT":           "#F44336",
    "OCGT":           "#FF9800",
    "GAS":            "#E91E63",
    "PS":             "#9C27B0",
    "NPSHYD":         "#00BCD4",
    "WIND":           "#4CAF50",
    "SOLAR":          "#FFEB3B",
    "BIOMASS":        "#795548",
    "COAL":           "#212121",
    "DIESEL":         "#607D8B",
    "LOAD RESPONSE":  "#FF5722",
    "INTERCONNECTOR": "#009688",
    "IFA":            "#00897B",
    "IFA2":           "#26A69A",
    "BritNed":        "#00ACC1",
    "NEMO":           "#0097A7",
    "Eleclink":       "#00838F",
    "Viking":         "#006064",
    "Moyle":          "#4DB6AC",
    "EastWest":       "#80CBC4",
    "NSL":            "#B2DFDB",
    "Greenlink":      "#E0F2F1",
    "OTHER":          "#9E9E9E",
    "UNKNOWN_NUMERIC":"#BDBDBD",
}

IC_FUELS = ["BritNed", "NEMO", "IFA", "IFA2", "Eleclink", "Viking", "Moyle", "EastWest", "NSL", "Greenlink", "INTERCONNECTOR"]


# ── helpers ──────────────────────────────────────────────────────────────────

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str, width: str = "100%", max_width: str = "900px") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:{max_width};">'


def section(title: str, content: str) -> str:
    return f"""
<section>
  <h2>{title}</h2>
  {content}
</section>"""


def details_block(summary: str, content: str, open_: bool = False) -> str:
    open_attr = " open" if open_ else ""
    return f"<details{open_attr}><summary>{html.escape(summary)}</summary>{content}</details>"


def table_html(df: pd.DataFrame) -> str:
    return df.to_html(border=0, classes="tbl", index=True)


def make_sided_label(sp: pd.DataFrame, col: str) -> pd.Series:
    return sp[col].fillna("MISSING").astype(str) + "_" + sp["niv_active_side"].str.upper()


def _fuel_colour_list(fuels: list[str]) -> list[str]:
    return [FUEL_COLOURS.get(f, "#9E9E9E") for f in fuels]


# ── existing plot functions ───────────────────────────────────────────────────

def plot_fuel_bar(sp: pd.DataFrame) -> str:
    counts = (
        sp.groupby("marginal_fuel_type")
        .size()
        .rename("n_candidates")
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(13, 4))
    colours = _fuel_colour_list(counts["marginal_fuel_type"].tolist())
    bars = ax.bar(counts["marginal_fuel_type"], counts["n_candidates"], color=colours,
                  edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    ax.set_xlabel("Fuel type")
    ax.set_ylabel("Marginal candidate rows")
    ax.set_title("Marginal candidates by fuel type — 2023–2025")
    ax.tick_params(axis="x", rotation=40)
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
    counts = (
        sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"])["n_tied_marginal_candidates"]
        .value_counts()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(counts.index.astype(int), counts.values, color="#5C6BC0", edgecolor="white")
    ax.set_xlabel("Number of co-marginal candidates in SP")
    ax.set_ylabel("Number of SPs")
    ax.set_title("Co-marginal candidate count distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_price_hist(sp: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, col, label, colour in zip(
        axes,
        ["systemPrice", "marginal_final_price"],
        ["System Price", "Raw marginal final price"],
        ["#EF5350", "#66BB6A"],
    ):
        data = sp[col].dropna()
        ax.hist(data, bins=60, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1,
                   label=f"Median {data.median():.1f}")
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
        ax.hist(subset["systemPrice"], bins=bins, histtype="step", color="#1E40AF",
                linewidth=1.8, label="System price", density=True)
        ax.hist(subset["marginal_final_price"], bins=bins, histtype="step", color="#E65100",
                linewidth=1.8, label="Raw marginal final price", density=True)
        ax.axvline(0, color="#424242", linestyle=":", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("£/MWh")
        ax.legend(fontsize=8)
    fig.suptitle("Distribution comparison: published system price vs identified marginal action price", fontsize=11)
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
        (axes[1], "bid",   "#1976D2", "Bid-active periods"),
    ]:
        subset = sample[sample["niv_active_side"].eq(side)]
        ax.scatter(subset["marginal_final_price"], subset["systemPrice"],
                   s=5, alpha=0.20, c=colour, linewidths=0)
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
        ("bid",   "#42A5F5", "Bid-active"),
    ]:
        subset = data.loc[data["niv_active_side"].eq(side), "price_gap"]
        ax.hist(subset, bins=bins, alpha=0.48, density=True, color=colour, label=label)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("System price − raw marginal final price (£/MWh)")
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
    colours = _fuel_colour_list(order)
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


def plot_niv_scatter(sp: pd.DataFrame) -> str:
    sp_unique = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    offer = sp_unique[sp_unique["niv_active_side"] == "offer"]
    bid   = sp_unique[sp_unique["niv_active_side"] == "bid"]
    ax.scatter(offer["niv_volume"], offer["systemPrice"], alpha=0.15, s=4,
               color="#EF5350", label="Offer (short)")
    ax.scatter(bid["niv_volume"],   bid["systemPrice"],   alpha=0.15, s=4,
               color="#42A5F5", label="Bid (long)")
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
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data    = [offer.loc[offer["marginal_fuel_type"] == f, "marginal_stack_rank"].dropna().values for f in order]
    colours = _fuel_colour_list(order)
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


# ── new plot functions ────────────────────────────────────────────────────────

def plot_temporal_trajectory_sided(sp: pd.DataFrame) -> str:
    """Two-panel monthly stacked % bar (offer / bid), normalised to share within each side."""
    sp2 = sp.copy()
    sp2["month"] = sp2["settlementDate"].dt.to_period("M").astype(str)

    TOP_FUELS = ["CCGT", "BATTERY", "WIND", "PS", "BIOMASS", "NPSHYD", "OCGT"]
    sp2["fuel_grp"] = sp2["marginal_fuel_type"].where(sp2["marginal_fuel_type"].isin(TOP_FUELS), "OTHER")

    # Build consistent fuel order from offer side (dominant side for ordering)
    offer_ct = sp2[sp2["niv_active_side"] == "offer"].groupby("fuel_grp").size()
    fuel_order = offer_ct.sort_values(ascending=False).index.tolist()
    colours = _fuel_colour_list(fuel_order)

    panels = []
    for side in ["offer", "bid"]:
        ct = (
            sp2[sp2["niv_active_side"] == side]
            .groupby(["month", "fuel_grp"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=fuel_order, fill_value=0)
        )
        ct_share = ct.div(ct.sum(axis=1), axis=0) * 100
        panels.append(ct_share)

    months = panels[0].index.tolist()
    tick_labels = [m if i % 3 == 0 else "" for i, m in enumerate(months)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, ct_share, title in [
        (axes[0], panels[0], "Offer-active SPs (system short — NIV > 0)"),
        (axes[1], panels[1], "Bid-active SPs (system long — NIV < 0)"),
    ]:
        bottom = np.zeros(len(ct_share))
        xs = np.arange(len(ct_share))
        for fuel, colour in zip(fuel_order, colours):
            vals = ct_share[fuel].values if fuel in ct_share.columns else np.zeros(len(ct_share))
            ax.bar(xs, vals, bottom=bottom, color=colour, edgecolor="white", width=0.85, label=fuel)
            bottom += vals
        ax.set_ylabel("Share (%)")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    handles = [mpatches.Patch(color=c, label=f) for c, f in zip(colours, fuel_order)]
    axes[0].legend(handles=handles, title="Fuel type",
                   bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.suptitle("Monthly marginal candidate mix by NIV active side — 2023–2025\n"
                 "(normalised within each side; quarterly x-axis labels)",
                 fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_technology_share_trends(sp: pd.DataFrame) -> str:
    """Monthly share of CCGT and BATTERY at the margin, offer and bid shown separately."""
    sp2 = sp.copy()
    sp2["month"] = sp2["settlementDate"].dt.to_period("M")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, side, title in [
        (axes[0], "offer", "Offer-active SPs (short system)"),
        (axes[1], "bid",   "Bid-active SPs (long system)"),
    ]:
        subset = sp2[sp2["niv_active_side"] == side]
        monthly_share = (
            subset.groupby("month")["marginal_fuel_type"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        months = [str(m) for m in monthly_share.index]
        xs = range(len(months))

        for fuel, lw in [("CCGT", 2.2), ("BATTERY", 2.2)]:
            if fuel in monthly_share.columns:
                ax.plot(xs, monthly_share[fuel] * 100,
                        color=FUEL_COLOURS[fuel], marker="o", markersize=3,
                        linewidth=lw, label=fuel)

        ax.set_xlabel("Month")
        ax.set_ylabel("Share of marginal candidates (%)")
        ax.set_title(title)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([m if i % 3 == 0 else "" for i, m in enumerate(months)],
                           rotation=45, ha="right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(fontsize=9)

    fig.suptitle("CCGT vs BATTERY: monthly share at the margin — 2023–2025", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_freq_vs_shap(sp: pd.DataFrame, shap_imp: pd.DataFrame) -> str:
    """Frequency vs mean|SHAP| scatter — one point per technology×side label."""
    if shap_imp.empty:
        return ""

    sp2 = sp.copy()
    sp2["label_x_side"] = make_sided_label(sp2, "marginal_fuel_type")
    freq = sp2.groupby("label_x_side").size().rename("count").reset_index()

    shap_fuel = (
        shap_imp[shap_imp["source_column"] == "marginal_fuel_type"]
        .rename(columns={"feature": "label_x_side"})
        [["label_x_side", "mean_abs_shap", "conditional_mean_shap_when_active", "active_rows_in_shap_sample"]]
    )
    merged = freq.merge(shap_fuel, on="label_x_side", how="inner")

    count_med = merged["count"].median()
    shap_med  = merged["mean_abs_shap"].median()

    merged["fuel"] = merged["label_x_side"].str.rsplit("_", n=1).str[0]
    merged["side"] = merged["label_x_side"].str.rsplit("_", n=1).str[1]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for side, marker, alpha in [("OFFER", "o", 0.78), ("BID", "s", 0.72)]:
        subset = merged[merged["side"].eq(side)]
        colours = [FUEL_COLOURS.get(f, "#9E9E9E") for f in subset["fuel"]]
        ax.scatter(
            subset["count"],
            subset["mean_abs_shap"],
            s=95,
            marker=marker,
            color=colours,
            edgecolors="white",
            linewidth=0.9,
            alpha=alpha,
            label=side.title(),
            zorder=3,
        )

    focus_labels = {
        "CCGT_OFFER",
        "CCGT_BID",
        "BATTERY_OFFER",
        "BATTERY_BID",
        "WIND_BID",
        "WIND_OFFER",
        "PS_OFFER",
        "PS_BID",
    }
    label_set = set(merged.nlargest(5, "mean_abs_shap")["label_x_side"])
    label_set |= set(merged.nlargest(5, "count")["label_x_side"])
    label_set |= focus_labels

    offsets = {
        "CCGT_OFFER": (8, 8),
        "CCGT_BID": (8, -10),
        "BATTERY_OFFER": (8, -12),
        "BATTERY_BID": (-72, 8),
        "WIND_BID": (8, 8),
        "WIND_OFFER": (8, -10),
        "PS_OFFER": (8, 8),
        "PS_BID": (8, -10),
    }
    for _, row in merged[merged["label_x_side"].isin(label_set)].iterrows():
        xytext = offsets.get(row["label_x_side"], (7, 4))
        ax.annotate(
            row["label_x_side"],
            (row["count"], row["mean_abs_shap"]),
            fontsize=8,
            xytext=xytext,
            textcoords="offset points",
            color="#263238",
            arrowprops=dict(arrowstyle="-", color="#B0BEC5", linewidth=0.6, shrinkA=0, shrinkB=3),
        )

    ax.axhline(shap_med,  color="#9E9E9E", linestyle="--", linewidth=0.8)
    ax.axvline(count_med, color="#9E9E9E", linestyle=":", linewidth=0.8)

    # Light quadrant hints, without labelling every point.
    for (xp, yp, txt) in [
        (0.70, 0.92, "frequent & high-impact"),
        (0.07, 0.92, "rarer & high-impact"),
        (0.72, 0.03, "frequent & low-impact\n(price taker ←)"),
        (0.07, 0.07, "rarer & low-impact"),
    ]:
        ax.text(xp, yp, txt, transform=ax.transAxes, fontsize=8,
                color="#757575", va="top" if yp > 0.5 else "bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Marginal candidate count (frequency at margin)")
    ax.set_ylabel("Mean |SHAP| — contribution to predicted SBP (£/MWh)")
    ax.set_title("Marginal frequency vs SHAP impact by technology × side\n"
                 "Log x-axis; labels limited to high-frequency, high-impact, and thesis-relevant fuels")
    ax.legend(title="NIV side", fontsize=8)
    ax.grid(axis="both", color="#E0E0E0", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_sbp_conditional_fuel(sp: pd.DataFrame) -> str:
    """SBP distribution conditional on marginal fuel type, offer and bid separately."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, side, title in [
        (axes[0], "offer", "Offer-active SPs (short system)"),
        (axes[1], "bid",   "Bid-active SPs (long system)"),
    ]:
        subset = sp[sp["niv_active_side"] == side]
        order = (
            subset.groupby("marginal_fuel_type")["systemPrice"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        data    = [subset.loc[subset["marginal_fuel_type"] == f, "systemPrice"].dropna().values for f in order]
        colours = _fuel_colour_list(order)
        bp = ax.boxplot(data, patch_artist=True, notch=False, showfliers=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_title(title)
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Published SBP (£/MWh)")
    fig.suptitle("System Buy Price conditional on marginal technology — IQR, no outliers\n"
                 "Hypothesis: BATTERY marginal in lower-price / longer-system periods",
                 fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_gap_by_fuel(sp: pd.DataFrame) -> str:
    """SBP − raw marginal price gap by fuel type, offer and bid separately."""
    data = sp.dropna(subset=["systemPrice", "marginal_final_price"]).copy()
    data["price_gap"] = data["systemPrice"] - data["marginal_final_price"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, side, title in [
        (axes[0], "offer", "Offer-active"),
        (axes[1], "bid",   "Bid-active"),
    ]:
        subset = data[data["niv_active_side"] == side]
        order = (
            subset.groupby("marginal_fuel_type")["price_gap"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        plot_data = [subset.loc[subset["marginal_fuel_type"] == f, "price_gap"].dropna().values for f in order]
        colours   = _fuel_colour_list(order)
        bp = ax.boxplot(plot_data, patch_artist=True, notch=False, showfliers=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_title(title)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("SBP − raw marginal price (£/MWh)")
    fig.suptitle("Price gap (SBP − raw marginal price) by marginal technology — IQR, no outliers",
                 fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── table / section functions ─────────────────────────────────────────────────

def _mode_or_missing(s: pd.Series) -> str:
    values = s.dropna().astype(str)
    if values.empty:
        return "MISSING"
    mode = values.mode()
    if mode.empty:
        return values.iloc[0]
    return mode.iloc[0]


def marginal_bmu_side_rows(sp: pd.DataFrame) -> pd.DataFrame:
    """One observed target row per SP/unit/side where that BMU is marginal."""
    keep_cols = [
        "settlementDate",
        "settlementPeriod",
        "marginal_bmu_id",
        "niv_active_side",
        "marginal_fuel_type",
        "marginal_family_id",
        "systemPrice",
        "marginal_final_price",
        "niv_volume",
    ]
    cols = [c for c in keep_cols if c in sp.columns]
    data = sp.loc[:, cols].dropna(
        subset=["settlementDate", "settlementPeriod", "marginal_bmu_id", "niv_active_side", "systemPrice"]
    ).copy()
    data = data.drop_duplicates(
        subset=["settlementDate", "settlementPeriod", "marginal_bmu_id", "niv_active_side"]
    )
    data["bmu_side_label"] = make_sided_label(data, "marginal_bmu_id")
    data["systemPrice"] = pd.to_numeric(data["systemPrice"], errors="coerce")
    if "marginal_final_price" in data.columns:
        data["marginal_final_price"] = pd.to_numeric(data["marginal_final_price"], errors="coerce")
    if "niv_volume" in data.columns:
        data["niv_volume"] = pd.to_numeric(data["niv_volume"], errors="coerce")
    return data.dropna(subset=["systemPrice"])


def unit_conditional_distribution_stats(sp: pd.DataFrame) -> pd.DataFrame:
    """Empirical target distribution conditional on BMU x side being at the margin."""
    data = marginal_bmu_side_rows(sp)
    if data.empty:
        return pd.DataFrame()

    agg_kwargs = dict(
        marginal_bmu_id=("marginal_bmu_id", _mode_or_missing),
        niv_active_side=("niv_active_side", _mode_or_missing),
        marginal_fuel_type=("marginal_fuel_type", _mode_or_missing),
        marginal_family_id=("marginal_family_id", _mode_or_missing),
        n_rows=("systemPrice", "size"),
        mean_system_price=("systemPrice", "mean"),
        sd_system_price=("systemPrice", "std"),
        min_system_price=("systemPrice", "min"),
        p05_system_price=("systemPrice", lambda s: s.quantile(0.05)),
        p25_system_price=("systemPrice", lambda s: s.quantile(0.25)),
        median_system_price=("systemPrice", "median"),
        p75_system_price=("systemPrice", lambda s: s.quantile(0.75)),
        p95_system_price=("systemPrice", lambda s: s.quantile(0.95)),
        max_system_price=("systemPrice", "max"),
    )
    if "marginal_final_price" in data.columns:
        agg_kwargs.update(
            median_raw_marginal_price=("marginal_final_price", "median"),
            p05_raw_marginal_price=("marginal_final_price", lambda s: s.quantile(0.05)),
            p95_raw_marginal_price=("marginal_final_price", lambda s: s.quantile(0.95)),
        )
    if "niv_volume" in data.columns:
        agg_kwargs.update(
            median_niv_mwh=("niv_volume", "median"),
        )

    stats = (
        data.groupby("bmu_side_label", dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
        .rename(columns={"bmu_side_label": "label_x_side"})
        .sort_values("n_rows", ascending=False)
    )
    stats["share_of_unit_margin_rows"] = stats["n_rows"] / len(data)
    stats["rank_by_frequency"] = np.arange(1, len(stats) + 1)
    return stats


def plot_bmu_side_conditional_distribution(sp: pd.DataFrame, top_n_each_side: int = 12) -> str:
    """Full observed SBP distributions for frequent BMU x side marginal setters."""
    data = marginal_bmu_side_rows(sp)
    if data.empty:
        return ""

    selected_labels: list[str] = []
    for side in ["offer", "bid"]:
        labels = (
            data.loc[data["niv_active_side"].eq(side), "bmu_side_label"]
            .value_counts()
            .head(top_n_each_side)
            .index.tolist()
        )
        selected_labels.extend(labels)
    selected = data[data["bmu_side_label"].isin(selected_labels)].copy()
    if selected.empty:
        return ""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, max(6, 0.43 * top_n_each_side + 2)),
        sharex=True,
    )
    rng = np.random.default_rng(4)
    side_specs = [
        ("offer", "#C62828", "Offer-active BMUs"),
        ("bid", "#1565C0", "Bid-active BMUs"),
    ]

    for ax, (side, colour, title) in zip(axes, side_specs):
        side_data = selected[selected["niv_active_side"].eq(side)].copy()
        if side_data.empty:
            ax.text(0.5, 0.5, f"No {side} rows", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        order = (
            side_data.groupby("bmu_side_label")["systemPrice"]
            .median()
            .sort_values()
            .index.tolist()
        )
        y_positions = np.arange(len(order))
        for y, label in zip(y_positions, order):
            values = side_data.loc[side_data["bmu_side_label"].eq(label), "systemPrice"].dropna()
            jitter = rng.uniform(-0.24, 0.24, size=len(values))
            ax.scatter(values, y + jitter, s=9, alpha=0.17, color=colour, linewidths=0)

            q10, q25, q50, q75, q90 = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            ax.hlines(y, q10, q90, color="#263238", linewidth=1.0, alpha=0.85)
            ax.hlines(y, q25, q75, color=colour, linewidth=4.0, alpha=0.90)
            ax.plot(q50, y, marker="|", color="black", markersize=13, markeredgewidth=2)

        meta = (
            side_data.groupby("bmu_side_label")
            .agg(
                n=("systemPrice", "size"),
                fuel=("marginal_fuel_type", _mode_or_missing),
            )
            .reindex(order)
        )
        labels = [f"{label} [{row.fuel}], n={int(row.n):,}" for label, row in meta.iterrows()]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.axvline(0, color="#616161", linestyle=":", linewidth=0.8)
        ax.axvline(data["systemPrice"].median(), color="#212121", linestyle="--", linewidth=0.8)
        ax.grid(axis="x", color="#E0E0E0", linewidth=0.6, alpha=0.9)
        ax.set_xscale("symlog", linthresh=50)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    axes[0].set_ylabel("BMU x NIV side, ordered by median observed SBP")
    for ax in axes:
        ax.set_xlabel("Observed systemPrice (GBP/MWh), symlog scale")
    fig.suptitle(
        "Direct conditional distribution: observed SBP when each BMU is at the margin\n"
        "Dots are individual SP observations; thick bar = IQR; thin bar = P10-P90; black tick = median",
        fontsize=11,
    )
    fig.tight_layout()
    return fig_to_b64(fig)


def unit_conditional_distribution_table(stats: pd.DataFrame, n: int = 30) -> str:
    if stats.empty:
        return "<p class='note'>No BMU-level conditional distribution rows available.</p>"
    show = stats.head(n).copy()
    show["share_of_unit_margin_rows"] = show["share_of_unit_margin_rows"].map("{:.2%}".format)
    keep = [
        "rank_by_frequency",
        "label_x_side",
        "marginal_fuel_type",
        "marginal_family_id",
        "n_rows",
        "share_of_unit_margin_rows",
        "median_system_price",
        "p05_system_price",
        "p95_system_price",
        "min_system_price",
        "max_system_price",
        "median_raw_marginal_price",
        "median_niv_mwh",
    ]
    show = show[[c for c in keep if c in show.columns]].rename(columns={
        "rank_by_frequency": "Rank",
        "label_x_side": "BMU x side",
        "marginal_fuel_type": "Fuel",
        "marginal_family_id": "Family",
        "n_rows": "Rows",
        "share_of_unit_margin_rows": "Share",
        "median_system_price": "Median SBP",
        "p05_system_price": "P5 SBP",
        "p95_system_price": "P95 SBP",
        "min_system_price": "Min SBP",
        "max_system_price": "Max SBP",
        "median_raw_marginal_price": "Median raw marginal",
        "median_niv_mwh": "Median NIV MWh",
    })
    return show.round(2).to_html(index=False, border=0, classes="tbl")


def distributional_summary_table(sp: pd.DataFrame) -> str:
    sp_unique = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"])

    def _row(s: pd.Series, name: str) -> dict:
        s = s.dropna()
        return {
            "Series": name,
            "N": f"{len(s):,}",
            "Mean": f"{s.mean():.2f}",
            "SD": f"{s.std():.2f}",
            "Min": f"{s.min():.2f}",
            "P5": f"{s.quantile(.05):.2f}",
            "P25": f"{s.quantile(.25):.2f}",
            "P50": f"{s.median():.2f}",
            "P75": f"{s.quantile(.75):.2f}",
            "P95": f"{s.quantile(.95):.2f}",
            "Max": f"{s.max():.2f}",
            "% < 0": f"{(s < 0).mean():.1%}",
            "% > £500": f"{(s > 500).mean():.1%}",
        }

    rows = [
        _row(sp_unique["systemPrice"],              "System Price (SP-unique)"),
        _row(sp["marginal_final_price"],            "Marginal final price (all candidate rows)"),
        _row(sp_unique["niv_volume"],               "NIV volume MWh (SP-unique)"),
        _row(sp["n_tied_marginal_candidates"].astype(float), "n tied co-marginals (SP-unique duplicated)"),
    ]
    return pd.DataFrame(rows).to_html(index=False, border=0, classes="tbl")


def class_balance_table(sp: pd.DataFrame) -> str:
    sp2 = sp.copy()
    sp2["label"] = make_sided_label(sp2, "marginal_fuel_type")
    counts = sp2["label"].value_counts().rename("Count").reset_index()
    counts.columns = ["Label (fuel×side)", "Count"]
    counts["Share"] = (counts["Count"] / len(sp2)).map("{:.2%}".format)
    return counts.to_html(index=False, border=0, classes="tbl")


def summary_stats(sp: pd.DataFrame) -> str:
    sp_unique = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"])
    rows = [
        ("Settlement dates",                 f"{sp['settlementDate'].dt.date.min()} to {sp['settlementDate'].dt.date.max()}"),
        ("Total SPs covered",                f"{len(sp_unique):,}"),
        ("Total marginal candidate rows",    f"{len(sp):,}"),
        ("SPs with single marginal (n=1)",   f"{(sp_unique['n_tied_marginal_candidates'] == 1).sum():,}"),
        ("SPs with co-marginals (n>1)",      f"{(sp_unique['n_tied_marginal_candidates'] > 1).sum():,}"),
        ("Max co-marginals in one SP",       f"{sp_unique['n_tied_marginal_candidates'].max()}"),
        ("Offer-active SPs",                 f"{(sp_unique['niv_active_side'] == 'offer').sum():,}"),
        ("Bid-active SPs",                   f"{(sp_unique['niv_active_side'] == 'bid').sum():,}"),
        ("Median SBP £/MWh",                 f"{sp_unique['systemBuyPrice'].median():.2f}"),
        ("Median NIV MWh",                   f"{sp_unique['niv_volume'].median():.1f}"),
        ("Unique fuel types at margin",      f"{sp['marginal_fuel_type'].nunique()}"),
        ("Unique BMUs at margin",            f"{sp['marginal_bmu_id'].nunique():,}"),
        ("Unique plant families at margin",  f"{sp['marginal_family_id'].nunique():,}"),
        ("Rows with unresolved numeric BMU", f"{sp['marginal_is_numeric_id'].sum():,}"),
    ]
    html = "<table class='tbl'><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
    for k, v in rows:
        html += f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
    html += "</tbody></table>"
    return html


def fuel_table(sp: pd.DataFrame) -> str:
    ct     = sp.groupby("marginal_fuel_type").size().rename("n_candidates")
    sp_ct  = sp.groupby("marginal_fuel_type").apply(
        lambda g: g[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    ).rename("n_sps_present")
    side_counts = (
        sp.pivot_table(index="marginal_fuel_type", columns="niv_active_side",
                       values="systemPrice", aggfunc="size", fill_value=0)
        .rename(columns={"bid": "bid_count", "offer": "offer_count"})
    )
    side_medians = (
        sp.pivot_table(index="marginal_fuel_type", columns="niv_active_side",
                       values="marginal_final_price", aggfunc="median")
        .rename(columns={"bid": "median_raw_bid_price", "offer": "median_raw_offer_price"})
        .round(2)
    )
    tbl = pd.concat([ct, side_counts, sp_ct, side_medians], axis=1).sort_values("n_candidates", ascending=False)
    for col in ["bid_count", "offer_count"]:
        if col not in tbl.columns:
            tbl[col] = 0
    for col in ["median_raw_bid_price", "median_raw_offer_price"]:
        if col not in tbl.columns:
            tbl[col] = np.nan
    tbl = tbl[["n_candidates", "bid_count", "offer_count", "n_sps_present",
               "median_raw_bid_price", "median_raw_offer_price"]]
    tbl.index.name = "fuel_type"
    tbl.columns = ["Total candidates", "Bid count", "Offer count", "SPs present",
                   "Median raw bid price GBP/MWh", "Median raw offer price GBP/MWh"]
    return tbl.to_html(border=0, classes="tbl")


def top_bmu_table(sp: pd.DataFrame, n: int = 20) -> str:
    tbl = (
        sp.groupby(["marginal_bmu_id", "marginal_fuel_type", "marginal_family_id"])
        .agg(
            total        =("systemPrice", "size"),
            offer_rows   =("niv_active_side", lambda x: (x == "offer").sum()),
            bid_rows     =("niv_active_side", lambda x: (x == "bid").sum()),
            median_sbp   =("systemPrice", "median"),
            median_raw   =("marginal_final_price", "median"),
        )
        .reset_index()
        .sort_values("total", ascending=False)
        .head(n)
    )
    tbl["share"] = (tbl["total"] / len(sp)).map("{:.2%}".format)
    tbl = tbl.rename(columns={
        "marginal_bmu_id":    "BMU",
        "marginal_fuel_type": "Fuel",
        "marginal_family_id": "Family",
        "total":              "Rows",
        "offer_rows":         "Offer",
        "bid_rows":           "Bid",
        "share":              "Share",
        "median_sbp":         "Median SBP £/MWh",
        "median_raw":         "Median raw £/MWh",
    })
    return tbl.round(2).to_html(index=False, border=0, classes="tbl")


def family_rollup_table(sp: pd.DataFrame, n: int = 20) -> str:
    tbl = (
        sp.groupby(["marginal_family_id", "marginal_fuel_type"])
        .agg(
            total      =("systemPrice", "size"),
            offer_rows =("niv_active_side", lambda x: (x == "offer").sum()),
            bid_rows   =("niv_active_side", lambda x: (x == "bid").sum()),
            n_bmus     =("marginal_bmu_id", "nunique"),
            median_sbp =("systemPrice", "median"),
        )
        .reset_index()
        .sort_values("total", ascending=False)
        .head(n)
    )
    tbl["share"] = (tbl["total"] / len(sp)).map("{:.2%}".format)
    tbl = tbl.rename(columns={
        "marginal_family_id": "Family",
        "marginal_fuel_type": "Fuel",
        "total":              "Rows",
        "offer_rows":         "Offer",
        "bid_rows":           "Bid",
        "n_bmus":             "BMUs",
        "share":              "Share",
        "median_sbp":         "Median SBP £/MWh",
    })
    return tbl.round(2).to_html(index=False, border=0, classes="tbl")


def other_bucket_section(sp: pd.DataFrame) -> str:
    other = sp[sp["marginal_fuel_type"] == "OTHER"].copy()
    other_bmu = (
        other.groupby(["marginal_bmu_id", "marginal_bmu_type", "marginal_family_id"])
        .agg(n=("systemPrice", "size"),
             offer=("niv_active_side", lambda x: (x == "offer").sum()),
             bid  =("niv_active_side", lambda x: (x == "bid").sum()))
        .reset_index()
        .sort_values("n", ascending=False)
        .head(20)
    )
    other_bmu["share_of_other"] = (other_bmu["n"] / len(other)).map("{:.1%}".format)

    gas = sp[sp["marginal_fuel_type"] == "GAS"].copy()
    gas_bmu = (
        gas.groupby(["marginal_bmu_id", "marginal_bmu_type"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
        .head(15)
    )

    ic = sp[sp["marginal_fuel_type"].isin(IC_FUELS)].copy()
    per_link = (
        ic.groupby("marginal_fuel_type")
        .agg(n=("systemPrice", "size"),
             bid  =("niv_active_side", lambda x: (x == "bid").sum()),
             offer=("niv_active_side", lambda x: (x == "offer").sum()),
             median_sbp=("systemPrice", "median"),
             median_raw=("marginal_final_price", "median"))
        .reset_index()
        .rename(columns={"marginal_fuel_type": "Link", "n": "Rows",
                         "median_sbp": "Median SBP £/MWh", "median_raw": "Median raw £/MWh",
                         "bid": "Bid", "offer": "Offer"})
        .sort_values("Rows", ascending=False)
    )
    total_row = pd.DataFrame([{
        "Link": "ALL INTERCONNECTORS",
        "Rows": len(ic),
        "Bid":  (ic["niv_active_side"] == "bid").sum(),
        "Offer":(ic["niv_active_side"] == "offer").sum(),
        "Median SBP £/MWh": round(ic["systemPrice"].median(), 2),
        "Median raw £/MWh": round(ic["marginal_final_price"].median(), 2),
    }])
    ic_tbl = pd.concat([per_link, total_row], ignore_index=True)

    bmu_type_note = (
        "BMU type codes: <strong>T</strong> = transmission-connected generator, "
        "<strong>E</strong> = embedded generator, <strong>S</strong> = supplier / trader, "
        "<strong>V</strong> = virtual / aggregator / demand."
    )

    return f"""
<h3>OTHER bucket — top 20 constituent BMUs
  ({len(other):,} rows, {len(other)/len(sp):.1%} of total marginal candidates)</h3>
<p class="note">{bmu_type_note} OTHER captures named BMUs not resolved to a specific generation technology
in the fuel-type register.</p>
{other_bmu.to_html(index=False, border=0, classes="tbl")}

<h3>GAS vs CCGT / OCGT disambiguation
  ({len(gas):,} rows, {len(gas)/len(sp):.1%} of total)</h3>
<p class="note">GAS captures residual gas-labelled BMUs that do not resolve to CCGT or OCGT in the
register. The {len(gas):,} GAS rows are predominantly supplier (S) and embedded (E) BMU types —
non-scheduled or embedded gas plant, not transmission-connected CCGT.</p>
{gas_bmu.to_html(index=False, border=0, classes="tbl")}

<h3>Interconnector aggregate view
  ({len(ic):,} rows, {len(ic)/len(sp):.1%} of total)</h3>
<p class="note">Rows with unresolved numeric BMU IDs are predominantly interconnector references
({sp['marginal_is_numeric_id'].sum():,} rows total). Each named link is shown separately below,
with an aggregated total.</p>
{ic_tbl.round(2).to_html(index=False, border=0, classes="tbl")}
"""


def load_shap_importance() -> pd.DataFrame:
    if not SHAP_IMPORTANCE_CSV.exists():
        return pd.DataFrame()
    imp = pd.read_csv(SHAP_IMPORTANCE_CSV)
    keep = ["source_column", "feature", "mean_abs_shap", "conditional_mean_shap_when_active",
            "mdi", "overall_shap_rank", "side_shap_rank", "active_rows_in_shap_sample"]
    imp = imp[[c for c in keep if c in imp.columns]].copy()
    for col in ["mean_abs_shap", "conditional_mean_shap_when_active", "mdi",
                "overall_shap_rank", "side_shap_rank", "active_rows_in_shap_sample"]:
        if col in imp.columns:
            imp[col] = pd.to_numeric(imp[col], errors="coerce")
    return imp


def classify_frequency_importance(freq_rank: int, shap_rank: float) -> str:
    shap_top = pd.notna(shap_rank) and shap_rank <= 20
    if freq_rank <= 20 and shap_top:
        return "frequent and important"
    if freq_rank <= 20:
        return "frequent but lower-importance"
    if shap_top:
        return "less frequent but important"
    return "lower frequency / lower importance"


def marginal_setter_frequency_table(
    sp: pd.DataFrame,
    source_column: str,
    label: str,
    shap_imp: pd.DataFrame,
    top_n: int = 25,
    include_heading: bool = True,
) -> str:
    feature_col = f"{source_column}_sided"
    block = sp.copy()
    block[feature_col] = make_sided_label(block, source_column)
    total_sps = block[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]

    freq = (
        block.groupby(feature_col)
        .agg(count=("systemPrice", "size"),
             median_system_price=("systemPrice", "median"),
             median_raw_marginal_price=("marginal_final_price", "median"))
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
            columns={"feature": "label_x_side"})
        freq = freq.merge(imp, on="label_x_side", how="left")
    else:
        for col in ["mean_abs_shap", "conditional_mean_shap_when_active", "mdi", "overall_shap_rank"]:
            freq[col] = np.nan

    freq["frequency_importance_read"] = [
        classify_frequency_importance(r, s)
        for r, s in zip(freq["frequency_rank"], freq["overall_shap_rank"])
    ]
    freq = freq.sort_values("frequency_rank").head(top_n)
    show_cols = ["frequency_rank", "label_x_side", "count", "unique_marginal_periods",
                 "share_of_marginal_periods", "median_system_price", "median_raw_marginal_price",
                 "mdi", "mean_abs_shap", "conditional_mean_shap_when_active",
                 "overall_shap_rank", "frequency_importance_read"]
    freq = freq[[c for c in show_cols if c in freq.columns]].rename(columns={
        "frequency_rank":                 "Rank",
        "label_x_side":                   "Label × side",
        "count":                          "Count",
        "unique_marginal_periods":        "Unique marginal periods",
        "share_of_marginal_periods":      "Share of marginal periods",
        "median_system_price":            "Median SBP",
        "median_raw_marginal_price":      "Median raw price",
        "mdi":                            "RF importance (MDI)",
        "mean_abs_shap":                  "Mean |SHAP|",
        "conditional_mean_shap_when_active": "Mean SHAP when active",
        "overall_shap_rank":              "SHAP rank",
        "frequency_importance_read":      "Freq / impact read",
    })
    heading = f"<h3>{label}</h3>" if include_heading else ""
    return heading + freq.round(4).to_html(index=False, border=0, classes="tbl")


def marginal_setter_frequency_section(sp: pd.DataFrame, shap_imp: pd.DataFrame) -> str:
    note = (
        "<p class='note'>Frequency is not importance. Count and share show how often a "
        "label×side combination is marginal; RF importance and mean |SHAP| come from the main "
        "sided SHAP/RF model. This separates frequent-but-ordinary setters from rarer "
        "high-impact setters. Technology is shown inline; family and BMU detail are collapsed "
        "below so the evidence is available without dominating the report.</p>"
    )
    metric_key = (
        "<p class='note'><strong>_BID / _OFFER</strong> = NIV active side "
        "(bid = long system; offer = short system).<br>"
        "<strong>RF importance (MDI)</strong> = Mean Decrease in Impurity.<br>"
        "<strong>Mean |SHAP|</strong> = unsigned magnitude of contribution to predicted SBP.<br>"
        "<strong>Mean SHAP when active</strong> = signed direction: negative pulls SBP down, "
        "positive pulls SBP up.</p>"
    )
    gen_table = marginal_setter_frequency_table(
        sp,
        "marginal_fuel_type",
        "Generation type",
        shap_imp,
        top_n=18,
    )
    family_table = marginal_setter_frequency_table(
        sp,
        "marginal_family_id",
        "Family ID",
        shap_imp,
        top_n=20,
        include_heading=False,
    )
    bmu_table = marginal_setter_frequency_table(
        sp,
        "marginal_bmu_id",
        "BMU ID",
        shap_imp,
        top_n=25,
        include_heading=False,
    )
    return (
        note
        + metric_key
        + gen_table
        + details_block("Family ID frequency and SHAP table", family_table)
        + details_block("BMU ID frequency and SHAP table", bmu_table)
    )


def marginal_vs_system_gap_table(sp: pd.DataFrame) -> str:
    data = sp.dropna(subset=["systemPrice", "marginal_final_price"]).copy()
    data["price_gap"]     = data["systemPrice"] - data["marginal_final_price"]
    data["abs_price_gap"] = data["price_gap"].abs()

    rows = []
    for label, subset in [("All", data)] + [
        (s.title(), data[data["niv_active_side"].eq(s)]) for s in ["offer", "bid"]
    ]:
        rows.append({
            "Side":                  label,
            "Rows":                  len(subset),
            "Median SBP":            subset["systemPrice"].median(),
            "Median raw marginal":   subset["marginal_final_price"].median(),
            "Median |gap|":          subset["abs_price_gap"].median(),
            "Share within £1":       subset["abs_price_gap"].le(1).mean(),
            "Share within £5":       subset["abs_price_gap"].le(5).mean(),
            "Share within £10":      subset["abs_price_gap"].le(10).mean(),
        })
    return pd.DataFrame(rows).round(4).to_html(index=False, border=0, classes="tbl")


def key_findings_callout() -> str:
    return """
<div class="callout">
  <p><strong>Central finding:</strong> CCGT (gas) dominates marginal price <em>formation</em>
     in the GB BM across 2023–2025, on both the offer and bid side.
     Battery storage, despite reaching significant scale by volume, exhibits price-<em>taker</em>
     behaviour: it is the second-most-frequent marginal candidate yet its SHAP contribution to SBP
     is disproportionately low, and it is concentrated in lower-price, long-system periods.</p>
  <p><strong>Bid/offer separation matters:</strong> WIND is almost exclusively a bid-side marginal
     setter (4,684 bid vs 50 offer rows) with a large negative SHAP (mean −56.8 £/MWh when active);
     treating bids and offers as pooled would obscure this entirely. This is the empirical
     justification for the Atherton et al. extension.</p>
  <p><strong>Unit-level identity matters:</strong> fuel-type shares mask sharp concentration in a
     small number of CCGT plant families (e.g. RYHPS, CNQPS, DRAXX). The BMU-level lens is the
     core methodological contribution.</p>
  <p><strong>Raw marginal final price differs substantially from published SBP</strong>, supporting
     SBP as the regression target rather than the stack-identified marginal price.</p>
</div>
"""


def coverage_note() -> str:
    return (
        '<p class="note">Theoretical SP count for 2023–2025 is 52,608 '
        '(3 years × 17,520). Coverage of 50,684 represents 96.3%; the ~1,924 '
        'missing SPs are NIV=0 periods and SPs with no energy-only accepted actions, '
        'excluded by design.</p>'
    )


# ── CSS ───────────────────────────────────────────────────────────────────────

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
  .hook { background: #fff8e1; border-left: 4px solid #F9A825;
          padding: 10px 14px; margin: 12px 0; font-size: 13px; color: #5D4037; }
  details { margin: 14px 0 18px 0; border: 1px solid #e0e0e0; border-radius: 6px;
            padding: 8px 12px 12px 12px; background: #fafafa; }
  summary { cursor: pointer; font-weight: 600; color: #283593; }
  details .tbl { margin-top: 10px; }
</style>
"""


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading SP summary ...")
    sp = pd.read_parquet(SP_SUMMARY_PATH)
    sp["settlementDate"] = pd.to_datetime(sp["settlementDate"])
    n_unique = sp[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    print(f"  {len(sp):,} rows, {n_unique:,} unique SPs")

    shap_imp = load_shap_importance()
    unit_cond_stats = unit_conditional_distribution_stats(sp)
    unit_conditional_csv_rel = str(UNIT_CONDITIONAL_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/")

    print("Building charts ...")
    b64_fuel_bar        = plot_fuel_bar(sp)
    b64_niv_fuel        = plot_niv_side_fuel(sp)
    b64_temporal        = plot_temporal_trajectory_sided(sp)
    b64_tech_trends     = plot_technology_share_trends(sp)
    b64_freq_shap       = plot_freq_vs_shap(sp, shap_imp)
    b64_unit_cond       = plot_bmu_side_conditional_distribution(sp)
    b64_sbp_cond        = plot_sbp_conditional_fuel(sp)
    b64_price_box       = plot_price_box(sp)
    b64_niv_scatter     = plot_niv_scatter(sp)
    b64_stack_rank      = plot_stack_rank(sp)
    b64_marg_dist       = plot_marginal_vs_system_distribution(sp)
    b64_marg_scatter    = plot_marginal_vs_system_scatter(sp)
    b64_gap_hist        = plot_price_gap_hist(sp)
    b64_gap_by_fuel     = plot_gap_by_fuel(sp)
    b64_price_hist      = plot_price_hist(sp)
    b64_tied            = plot_tied_dist(sp)

    print("Assembling HTML ...")
    body = f"""
{CSS}
<h1>EDA — 2023–2025 NIV-Marginal SP Summary</h1>
<p>Source: <code>bid_offer_stack_energy_2023_2025_niv_marginal_sp_summary.parquet</code><br>
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>

{key_findings_callout()}

{section("1. Overview",
    summary_stats(sp)
    + coverage_note()
    + "<h3>Distributional summary — key series</h3>"
    + distributional_summary_table(sp)
    + "<h3>Class balance — marginal technology × side label</h3>"
    + "<p class='note'>These are the model input labels. Rare classes (e.g. WIND_BID) "
      "may carry high SHAP despite few rows — treat with caution.</p>"
    + class_balance_table(sp)
)}

{section("2. CCGT dominates at the margin",
    "<p>CCGT is the dominant marginal setter by candidate count and by SHAP contribution, "
    "on both the offer side (price-up pressure) and the bid side (price-down). "
    "BATTERY ranks second by frequency but its SHAP rank is fifth or sixth.</p>"
    + fuel_table(sp)
    + img_tag(b64_fuel_bar)
    + img_tag(b64_niv_fuel)
    + "<p class='note'>Offer-active = short system (NIV > 0); bid-active = long system (NIV < 0).</p>"
)}

{section("3. Temporal trajectory: batteries entering at scale, CCGT retaining price formation",
    "<p>Monthly share of marginal candidates by technology, normalised within each NIV side. "
    "The two-panel stacked chart separates offer-side (price-up) and bid-side (price-down) "
    "dynamics. Follow CCGT (red) and BATTERY (blue) across 2023–2025.</p>"
    + img_tag(b64_temporal)
    + "<h3>CCGT vs BATTERY — share trends by side</h3>"
    + img_tag(b64_tech_trends)
    + ""
)}

{section("4. BMU-level view — why unit identity matters",
    "<p>Fuel-type shares mask how marginal setting is distributed across individual BMUs and plant families. "
    "Price-setting power is not evenly distributed across CCGT BMUs. This is the "
    "empirical motivation for going to unit level rather than stopping at technology.</p>"
    + "<h3>Top 20 BMUs by marginal frequency</h3>"
    + top_bmu_table(sp)
    + "<h3>Top 20 plant families by marginal frequency</h3>"
    + family_rollup_table(sp)
)}

{section("5. Frequency vs impact: the price-taker fingerprint",
    "<p>The scatter below places each technology×side label at its marginal frequency (x) "
    "against its mean |SHAP| (y). The bottom-right quadrant — frequent but low-impact — "
    "is the price-taker fingerprint. BATTERY sits there; CCGT_OFFER and WIND_BID do not.</p>"
    + img_tag(b64_freq_shap)
    + "<h3>Frequency and SHAP tables by granularity</h3>"
    + marginal_setter_frequency_section(sp, shap_imp)
)}

{section("6. Direct unit-conditional price distributions",
    "<p>This is the direct empirical check: for each BMU x side, take every SP where that "
    "unit is marginal and plot the observed <code>systemPrice</code>. The plot shows the "
    "most frequent offer-side and bid-side BMU setters; the CSV contains the full "
    "BMU x side percentile table for all units.</p>"
    + img_tag(b64_unit_cond, max_width="1100px")
    + "<p class='note'>The x-axis uses a symlog scale so negative prices and high-price tails "
      "remain visible. Dots are individual SP observations, not model effects.</p>"
    + "<h3>Top BMU x side conditional distribution stats</h3>"
    + unit_conditional_distribution_table(unit_cond_stats)
    + f"<p class='note'>Full table saved to <code>{unit_conditional_csv_rel}</code>.</p>"
)}

{section("7. Price conditional on marginal technology",
    "<p>Hypothesis: BATTERY is predominantly marginal in lower-price, long-system periods — "
    "the mechanism behind its low SHAP. Box plots of published SBP conditional on the marginal "
    "technology, shown separately for offer-active and bid-active SPs.</p>"
    + img_tag(b64_sbp_cond)
    + "<h3>Raw marginal final price by fuel type (pooled)</h3>"
    + img_tag(b64_price_box)
    + "<p class='note'>Raw marginal final price: bid-side prices retain their original signs. "
      "Outliers hidden (IQR only).</p>"
)}

{section("8. Bid/offer asymmetry — why separating bids and offers matters",
    "<p>WIND_BID has a mean SHAP of −56.8 £/MWh when active — the second-most-impactful "
    "feature overall — yet it appears almost exclusively on the bid side (4,684 bid rows vs "
    "50 offer rows). Pooling bids and offers would bury this signal entirely. "
    "Stack rank and NIV scatter confirm the structural difference between the two sides.</p>"
    + img_tag(b64_niv_scatter)
    + "<p class='note'>Each point is a unique SP. Offer = short system (NIV > 0); "
      "Bid = long system (NIV < 0).</p>"
    + img_tag(b64_stack_rank)
    + "<p class='note'>Higher stack rank = deeper into the merit order = more accepted offers "
      "above the marginal unit. Offer-active SPs only.</p>"
)}

{section("9. Marginal action price vs published SBP — target justification",
    "<p>The raw marginal final price identified from the stack is not the same as the "
    "published System Buy Price. The gap is systematic: bid-active periods are far less "
    "likely than offer-active periods to have their marginal price close to SBP. "
    "This motivates SBP — not the raw stack price — as the regression target.</p>"
    + img_tag(b64_marg_dist)
    + img_tag(b64_marg_scatter)
    + img_tag(b64_gap_hist)
    + "<h3>Gap distribution conditional on marginal technology</h3>"
    + img_tag(b64_gap_by_fuel)
    + "<h3>Gap summary table</h3>"
    + marginal_vs_system_gap_table(sp)
)}

{section("10. Supporting distributions",
    "<h3>Price distributions</h3>"
    + img_tag(b64_price_hist)
    + "<h3>Co-marginal candidate count distribution</h3>"
    + img_tag(b64_tied)
    + "<p class='note'>Count of distinct BMUs sharing the top accepted price in each SP. "
      "SPs with n_tied > 1 contribute multiple rows to the candidate dataset.</p>"
)}

{section("11. Data quality and taxonomy",
    other_bucket_section(sp)
)}
"""

    UNIT_CONDITIONAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    unit_cond_stats.to_csv(UNIT_CONDITIONAL_CSV, index=False)
    print(f"Saved -> {UNIT_CONDITIONAL_CSV}")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>EDA 2023–2025</title></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    print(f"Saved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
