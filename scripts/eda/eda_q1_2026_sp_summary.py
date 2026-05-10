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
    / "data" / "processed" / "q1_2026" / "bid_offer_stack_q1_2026"
    / "bid_offer_stack_energy_q1_2026_niv_marginal_sp_summary.parquet"
)
OUT_HTML = PROJECT_ROOT / "reports" / "q1_2026" / "canonical" / "eda_q1_2026_sp_summary.html"

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
    "DIESEL":        "#607D8B",
    "LOAD RESPONSE": "#FF5722",
    "INTERCONNECTOR":"#009688",
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
    ax.set_title("Marginal candidates by fuel type — Q1 2026")
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
    fuels = ct.columns.tolist()
    colours = [FUEL_COLOURS.get(f, "#9E9E9E") for f in fuels]
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
        ["systemBuyPrice", "systemSellPrice", "marginal_price_cost_to_so"],
        ["System Buy Price", "System Sell Price", "Marginal Price (cost to SO)"],
        ["#EF5350", "#42A5F5", "#66BB6A"],
    ):
        data = sp[col].dropna()
        ax.hist(data, bins=60, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1, label=f"Median {data.median():.1f}")
        ax.set_xlabel("£/MWh")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sp_unique = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"])
    fig.suptitle("Price distributions (SP-level where applicable)", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_price_box(sp: pd.DataFrame) -> str:
    order = (
        sp.groupby("marginal_fuel_type")["marginal_price_cost_to_so"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [sp.loc[sp["marginal_fuel_type"] == f, "marginal_price_cost_to_so"].dropna().values for f in order]
    colours = [FUEL_COLOURS.get(f, "#9E9E9E") for f in order]
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("£/MWh")
    ax.set_title("Marginal price (cost to SO) by fuel type — IQR, no outliers")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_weekly_fuel(sp: pd.DataFrame) -> str:
    sp2 = sp.copy()
    sp2["week"] = sp2["settlementDate"].dt.isocalendar().week.astype(int)
    top_fuels = (
        sp2.groupby("marginal_fuel_type").size()
        .sort_values(ascending=False).head(7).index.tolist()
    )
    sp2["fuel_grp"] = sp2["marginal_fuel_type"].where(sp2["marginal_fuel_type"].isin(top_fuels), "OTHER")
    ct = (
        sp2.groupby(["week", "fuel_grp"])
        .size()
        .unstack(fill_value=0)
    )
    # reorder columns by total
    ct = ct[ct.sum().sort_values(ascending=False).index]
    colours = [FUEL_COLOURS.get(f, "#9E9E9E") for f in ct.columns]
    fig, ax = plt.subplots(figsize=(11, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=colours, edgecolor="white", width=0.8)
    ax.set_xlabel("ISO week number")
    ax.set_ylabel("Marginal candidate rows")
    ax.set_title("Weekly marginal candidate mix — top 7 fuel types")
    ax.legend(title="Fuel type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_niv_scatter(sp: pd.DataFrame) -> str:
    sp_unique = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    offer = sp_unique[sp_unique["niv_active_side"] == "offer"]
    bid = sp_unique[sp_unique["niv_active_side"] == "bid"]
    ax.scatter(offer["niv_volume"], offer["systemPrice"], alpha=0.25, s=6, color="#EF5350", label="Offer (short)")
    ax.scatter(bid["niv_volume"], bid["systemPrice"], alpha=0.25, s=6, color="#42A5F5", label="Bid (long)")
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("NIV (MWh)")
    ax.set_ylabel("System Price (£/MWh)")
    ax.set_title("System price vs NIV — Q1 2026")
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


def fuel_table(sp: pd.DataFrame) -> str:
    sp_unique = sp.drop_duplicates(subset=["settlementDate", "settlementPeriod"])
    ct = sp.groupby("marginal_fuel_type").size().rename("n_candidates")
    sp_ct = sp.groupby("marginal_fuel_type").apply(
        lambda g: g[["settlementDate", "settlementPeriod"]].drop_duplicates().shape[0]
    ).rename("n_sps_present")
    price_stats = sp.groupby("marginal_fuel_type")["marginal_price_cost_to_so"].agg(
        median_price="median", p10="quantile", p90=lambda x: x.quantile(0.9)
    ).round(2)
    tbl = pd.concat([ct, sp_ct, price_stats], axis=1).sort_values("n_candidates", ascending=False)
    tbl.index.name = "fuel_type"
    tbl.columns = ["Candidates", "SPs present", "Median £/MWh", "P10 £/MWh", "P90 £/MWh"]
    return tbl.to_html(border=0, classes="tbl")


# ── HTML assembly ─────────────────────────────────────────────────────────────

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #212121; }
  h1   { border-bottom: 3px solid #1565C0; padding-bottom: 8px; color: #1565C0; }
  h2   { margin-top: 40px; color: #283593; border-left: 4px solid #5C6BC0;
         padding-left: 10px; }
  section { margin-bottom: 40px; }
  .tbl { border-collapse: collapse; font-size: 13px; width: 100%; }
  .tbl th { background: #1565C0; color: white; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
  .tbl tr:nth-child(even) td { background: #f5f5f5; }
  img  { display: block; margin: 12px 0; border-radius: 6px;
         box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  p.note { font-size: 12px; color: #757575; font-style: italic; }
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
    b64_price_box  = plot_price_box(sp)
    b64_weekly     = plot_weekly_fuel(sp)
    b64_scatter    = plot_niv_scatter(sp)
    b64_rank       = plot_stack_rank(sp)

    print("Assembling HTML ...")
    body = f"""
{CSS}
<h1>EDA — Q1 2026 NIV-Marginal SP Summary</h1>
<p>Source: <code>bid_offer_stack_energy_q1_2026_niv_marginal_sp_summary.parquet</code><br>
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>

{section("1. Overview", summary_stats(sp))}

{section("2. Marginal fuel type breakdown",
    fuel_table(sp)
    + img_tag(b64_fuel_bar)
    + '<p class="note">One row per (SP, BMU) at the top accepted price. SPs with co-marginals contribute multiple rows.</p>'
)}

{section("3. Fuel type by NIV active side",
    img_tag(b64_niv_fuel)
    + '<p class="note">Offer-active = short system (NIV &gt; 0); bid-active = long system (NIV &lt; 0).</p>'
)}

{section("4. Co-marginal distribution",
    img_tag(b64_tied)
    + '<p class="note">Count of distinct BMUs sharing the top accepted price in each SP.</p>'
)}

{section("5. Price distributions",
    img_tag(b64_price_hist)
)}

{section("6. Marginal price by fuel type",
    img_tag(b64_price_box)
    + '<p class="note">Marginal price cost to SO: offer price unchanged; bid price negated. Outliers hidden.</p>'
)}

{section("7. Weekly marginal mix",
    img_tag(b64_weekly)
)}

{section("8. System price vs NIV",
    img_tag(b64_scatter)
    + '<p class="note">Each point is a unique SP. SBP/SSP spread drives the system price colour-coded by NIV direction.</p>'
)}

{section("9. Stack rank at margin (offer-active SPs)",
    img_tag(b64_rank)
    + '<p class="note">Higher stack rank = deeper into the merit order = more accepted offers above the marginal unit.</p>'
)}
"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>EDA Q1 2026</title></head><body>{body}</body></html>", encoding="utf-8")
    print(f"Saved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
