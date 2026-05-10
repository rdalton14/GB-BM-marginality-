from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "marginal_action_sp_2023_2025_identity_unknown_numeric.parquet"
)
SUMMARY_PATH = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_id_resolution_2023_2025_summary.json"
OUT_HTML = PROJECT_ROOT / "outputs" / "reports" / "eda_report_2023_2025_ispstack_identity_unknown_numeric.html"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F8F8",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linewidth": 0.8,
    }
)

C_OFFER = "#E45756"
C_BID = "#4C78A8"
C_UNKNOWN = "#9C755F"
C_ACCENT = "#72B7B2"


def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(fig: plt.Figure, width: str = "100%") -> str:
    return (
        f'<img src="data:image/png;base64,{fig_to_b64(fig)}" '
        f'style="width:{width};max-width:1100px;display:block;margin:0 auto;" />'
    )


def subsection(num: str, title: str, note_above: str, chart_html: str, note_below: str) -> str:
    return (
        f'<div class="subsection" id="{num}">'
        f"<h3>{num} {title}</h3>"
        f'<p class="note-above">{note_above}</p>'
        f"{chart_html}"
        f'<p class="obs">{note_below}</p>'
        f"</div>"
    )


def section(sid: str, title: str, content: str) -> str:
    return f'<section id="{sid}"><h2>{title}</h2>{content}</section>'


def html_table(df_t: pd.DataFrame) -> str:
    headers = "".join(f"<th>{c}</th>" for c in df_t.columns)
    rows = []
    for _, row in df_t.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in df_t.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="dtable"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def load_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(PANEL_PATH)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    df["timestamp"] = df["settlementDate"] + pd.to_timedelta((df["settlementPeriod"] - 1) * 30, unit="m")
    df["hour"] = ((df["settlementPeriod"] - 1) // 2).astype(int)
    df["month_name"] = df["settlementDate"].dt.strftime("%b")
    df["year_month"] = df["settlementDate"].dt.to_period("M").astype(str)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return df, summary


def main() -> None:
    sns.set_theme(style="whitegrid")
    df, summary = load_data()

    n_total = len(df)
    date_min = df["settlementDate"].min().strftime("%Y-%m-%d")
    date_max = df["settlementDate"].max().strftime("%Y-%m-%d")

    tech_counts = df["marginal_tech_final"].fillna("UNKNOWN").value_counts()
    gen_counts = df["marginal_generator_label_final"].fillna("UNKNOWN").value_counts()
    top_tech = tech_counts.head(12)
    top_generators = gen_counts.head(20)
    tech_palette = sns.color_palette("tab20", n_colors=max(len(top_tech), 8))
    tech_colors = {tech: tech_palette[i] for i, tech in enumerate(top_tech.index)}
    if "UNKNOWN_NUMERIC" in tech_counts.index:
        tech_colors["UNKNOWN_NUMERIC"] = C_UNKNOWN

    sec1 = ""
    overview = pd.DataFrame(
        {
            "Metric": [
                "Settlement periods",
                "Date range",
                "Offer winners",
                "Bid winners",
                "Mixed/ambiguous winners",
                "Identity-substituted rows",
                "Unknown-numeric rows",
                "Unique final generators",
                "Unique final technologies",
            ],
            "Value": [
                f"{n_total:,}",
                f"{date_min} to {date_max}",
                f"{summary['winner_side_counts'].get('offer', 0):,} ({100 * summary['winner_side_counts'].get('offer', 0) / n_total:.2f}%)",
                f"{summary['winner_side_counts'].get('bid', 0):,} ({100 * summary['winner_side_counts'].get('bid', 0) / n_total:.2f}%)",
                f"{summary['winner_side_counts'].get('mixed_or_ambiguous', 0):,}",
                f"{summary['winner_rows_identity_substituted']:,}",
                f"{summary['numeric_winner_rows_unknown_numeric']:,}",
                f"{df['marginal_generator_label_final'].nunique(dropna=True):,}",
                f"{df['marginal_tech_final'].nunique(dropna=True):,}",
            ],
        }
    )
    sec1 += subsection(
        "1.1",
        "Panel Snapshot",
        "This is the full 2023-2025 ISPSTACK winner-action panel after the same identity cleanup we used on Q1 2026.",
        html_table(overview),
        "The nice result here is that the fallback logic fully resolved the numeric-winner issue in the final panel: there are no remaining `UNKNOWN_NUMERIC` rows in this full-history version.",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    side_counts = pd.Series(summary["winner_side_counts"])
    side_order = [s for s in ["offer", "bid", "mixed_or_ambiguous"] if s in side_counts.index]
    colors = [C_OFFER if s == "offer" else C_BID if s == "bid" else C_UNKNOWN for s in side_order]
    ax.bar(side_order, side_counts.reindex(side_order).values, color=colors, edgecolor="white")
    ax.set_title("Winner Side Split, 2023-2025", fontsize=11, fontweight="bold")
    ax.set_ylabel("Settlement periods")
    fig.tight_layout()
    sec1 += subsection(
        "1.2",
        "Offer vs Bid Winners",
        "The winner-takes-price rule is still strongly offer-led at full-history scale, but bid-led periods are no longer tiny once we move beyond Q1 2026.",
        img_tag(fig),
        ", ".join([f"{side}: {int(side_counts[side]):,} ({100 * side_counts[side] / n_total:.2f}%)" for side in side_order]),
    )

    sec2 = ""
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(top_tech.index, top_tech.values, color=[tech_colors.get(t, C_ACCENT) for t in top_tech.index], edgecolor="white")
    for b, v in zip(bars, top_tech.values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(5, top_tech.max() * 0.005), f"{v:,}", ha="center", fontsize=8)
    ax.set_title("Top Final Marginal Technologies", fontsize=11, fontweight="bold")
    ax.set_ylabel("Settlement periods")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    sec2 += subsection(
        "2.1",
        "Technology Mix",
        "The final marginal technology mix after generator-label cleanup and numeric-ID fallback.",
        img_tag(fig),
        ", ".join([f"{tech}: {count:,}" for tech, count in top_tech.items()]),
    )

    monthly_tech = (
        df.groupby(["year_month", "marginal_tech_final"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_tech.index[:6], fill_value=0)
    )
    monthly_share = monthly_tech.div(monthly_tech.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 4))
    for tech in monthly_share.columns:
        ax.plot(monthly_share.index, monthly_share[tech], linewidth=1.8, label=tech, color=tech_colors.get(tech, C_ACCENT))
    tick_step = max(1, len(monthly_share) // 12)
    ax.set_xticks(range(0, len(monthly_share), tick_step))
    ax.set_xticklabels(monthly_share.index[::tick_step], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Share of monthly winner rows")
    ax.set_title("Monthly Technology Share of Winner Rows", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    sec2 += subsection(
        "2.2",
        "Technology Share Over Time",
        "A rolling view of who sits at the margin month by month, which is useful for spotting structural change rather than quarter-specific noise.",
        img_tag(fig),
        "This is one of the most useful plots for seeing whether battery, CCGT, PS, and peaker technologies are gaining or losing marginal share over the full study window.",
    )

    sec3 = ""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["marginal_price_winner"].dropna(), bins=100, ax=ax, color=C_OFFER)
    ax.set_title("Winner Marginal Price Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("GBP/MWh")
    fig.tight_layout()
    sec3 += subsection(
        "3.1",
        "Winner Price Distribution",
        "Distribution of selected winner prices across the full 2023-2025 sample.",
        img_tag(fig),
        f"Median winner price is {df['marginal_price_winner'].median():.2f} GBP/MWh; the 95th percentile is {df['marginal_price_winner'].quantile(0.95):.2f}; the 99th percentile is {df['marginal_price_winner'].quantile(0.99):.2f}.",
    )

    price_by_side = (
        df.groupby("marginal_side_winner")["marginal_price_winner"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
        .reset_index()
    )
    sec3 += subsection(
        "3.2",
        "Winner Price by Side",
        "A compact summary of whether bid-led periods price differently from offer-led periods.",
        html_table(price_by_side),
        "This is a helpful grounding table before we move into any modelling, because it tells us whether the side split is economically meaningful rather than just a bookkeeping flag.",
    )

    hourly_price = (
        df.groupby(["hour", "marginal_side_winner"])["marginal_price_winner"]
        .median()
        .unstack(fill_value=np.nan)
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    for side, color in [("offer", C_OFFER), ("bid", C_BID)]:
        if side in hourly_price.columns:
            ax.plot(hourly_price.index, hourly_price[side], linewidth=2, marker="o", markersize=3, label=side, color=color)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Median winner price (GBP/MWh)")
    ax.set_title("Intraday Winner Price by Side", fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    sec3 += subsection(
        "3.3",
        "Intraday Winner Price by Side",
        "Median winner prices across the day, split by whether the selected winner sits on the offer or bid side.",
        img_tag(fig),
        "If the bid and offer curves diverge at certain hours, that gives us a first hint about when side-specific pricing behavior really matters.",
    )

    sec4 = ""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top_generators)), top_generators.values, color=C_ACCENT, edgecolor="white")
    ax.set_yticks(range(len(top_generators)))
    ax.set_yticklabels(top_generators.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Settlement periods")
    ax.set_title("Top 20 Final Generator Labels", fontsize=11, fontweight="bold")
    fig.tight_layout()
    sec4 += subsection(
        "4.1",
        "Generator Concentration",
        "The most frequent final generator labels after the family-register and identity-resolution layer.",
        img_tag(fig),
        f"The top generator appears {int(top_generators.iloc[0]):,} times, and the top 20 together account for {100 * top_generators.sum() / n_total:.1f}% of all winner rows.",
    )

    top50_share = gen_counts.head(50).sum() / n_total
    fig, ax = plt.subplots(figsize=(10, 4))
    cum_share = gen_counts.head(50).cumsum() / n_total * 100
    ax.plot(range(1, len(cum_share) + 1), cum_share.values, color=C_BID, linewidth=2, marker="o", markersize=3)
    ax.axhline(80, color="red", linestyle="--", linewidth=1)
    idx_80 = np.searchsorted(cum_share.values, 80)
    if idx_80 < len(cum_share):
        ax.axvline(idx_80 + 1, color="red", linestyle=":", linewidth=1)
        ax.text(idx_80 + 1.5, 55, f"{idx_80 + 1} generators -> 80%", color="red", fontsize=9)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative % of winner rows")
    ax.set_title("Cumulative Winner Concentration, Top 50 Generators", fontsize=11, fontweight="bold")
    fig.tight_layout()
    sec4 += subsection(
        "4.2",
        "Concentration Curve",
        "A quick view of how concentrated the marginal generator distribution is once we work at the cleaned generator-label level.",
        img_tag(fig),
        f"The top 50 generators account for {100 * top50_share:.1f}% of all winner rows.",
    )

    nav_items = [
        ("s1", "1. Overview"),
        ("s2", "2. Technology Mix"),
        ("s3", "3. Winner Pricing"),
        ("s4", "4. Generator Concentration"),
    ]
    nav_html = "\n".join(f'<li><a href="#{sid}">{label}</a></li>' for sid, label in nav_items)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EDA Report - 2023-2025 ISPSTACK identity final</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;font-size:13px;color:#222;background:#F5F5F5;display:flex}}
#sidebar{{width:230px;min-width:230px;height:100vh;position:sticky;top:0;background:#1E293B;overflow-y:auto;padding:20px 0;flex-shrink:0}}
#sidebar h1{{color:#94A3B8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;padding:0 16px 12px}}
#sidebar ul{{list-style:none;padding:0}}
#sidebar li a{{display:block;padding:8px 16px;color:#CBD5E1;text-decoration:none;font-size:12px;border-left:3px solid transparent;transition:all .15s}}
#sidebar li a:hover{{background:#334155;color:#F8FAFC;border-left-color:#72B7B2}}
#content{{flex:1;min-width:0;padding:24px 32px;max-width:1320px}}
h2{{font-size:18px;font-weight:700;color:#1E293B;margin:32px 0 12px;padding-bottom:8px;border-bottom:2px solid #E2E8F0}}
h3{{font-size:14px;font-weight:600;color:#334155;margin:24px 0 8px}}
.subsection{{background:white;border-radius:8px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.07)}}
.note-above{{color:#475569;font-size:12px;line-height:1.6;margin-bottom:14px;padding:10px 14px;background:#F8FAFC;border-left:3px solid #72B7B2;border-radius:4px}}
.obs{{color:#475569;font-size:12px;line-height:1.6;margin-top:12px;padding:10px 14px;background:#FFF7ED;border-left:3px solid #E45756;border-radius:4px}}
table.dtable{{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}}
table.dtable th{{background:#1E293B;color:#F8FAFC;padding:7px 12px;text-align:left;font-weight:600}}
table.dtable td{{padding:6px 12px;border-bottom:1px solid #E2E8F0}}
table.dtable tr:nth-child(even){{background:#F8FAFC}}
section{{margin-bottom:40px}}
</style>
</head>
<body>
<nav id="sidebar">
  <h1>2023-2025 ISPSTACK EDA</h1>
  <ul>{nav_html}</ul>
</nav>
<main id="content">
  <h1 style="font-size:22px;font-weight:800;color:#1E293B;margin-bottom:4px">EDA - 2023-2025 ISPSTACK Winner Panel</h1>
  <p style="color:#64748B;font-size:12px;margin-bottom:8px">
    Final identity-cleaned panel | {date_min} to {date_max} | {n_total:,} settlement periods
  </p>
  {section("s1", "1. Overview", sec1)}
  {section("s2", "2. Technology Mix", sec2)}
  {section("s3", "3. Winner Pricing", sec3)}
  {section("s4", "4. Generator Concentration", sec4)}
</main>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved -> {OUT_HTML}")


if __name__ == "__main__":
    main()
