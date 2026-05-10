from __future__ import annotations

import base64
import io
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_CSV = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "master_panel_2023_2025_cleaned_refined_stack_manual_register.csv"
OUT_HTML = PROJECT_ROOT / "outputs" / "reports" / "eda_report_master_panel_2023_2025_original.html"

C_GAS = "#E45756"
C_BESS = "#4C78A8"
C_OTHER = "#72B7B2"
C_ACCENT = "#F58518"

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


def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(fig: plt.Figure, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{fig_to_b64(fig)}" style="width:{width};max-width:1100px;display:block;margin:0 auto;" />'


def section(sid: str, title: str, content: str) -> str:
    return f'<section id="{sid}"><h2>{title}</h2>{content}</section>'


def subsection(num: str, title: str, note_above: str, chart_html: str, note_below: str) -> str:
    return (
        f'<div class="subsection" id="{num}">'
        f"<h3>{num} {title}</h3>"
        f'<p class="note-above">{note_above}</p>'
        f"{chart_html}"
        f'<p class="obs">{note_below}</p>'
        f"</div>"
    )


def html_table(df_t: pd.DataFrame) -> str:
    headers = "".join(f"<th>{c}</th>" for c in df_t.columns)
    rows = []
    for _, row in df_t.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in df_t.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="dtable"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


TECH_COL = "marginal_tech_final"


print("Loading cleaned original stack panel with manual register labels ...")
df = pd.read_csv(PANEL_CSV, parse_dates=["settlementDate"])
df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
df["hour"] = ((df["settlementPeriod"] - 1) // 2).astype(int)
df["month_name"] = df["settlementDate"].dt.strftime("%b")
df["day_of_week"] = df["settlementDate"].dt.dayofweek
df["dow_name"] = df["settlementDate"].dt.strftime("%a")
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_peak"] = df["hour"].between(7, 22).astype(int)
df["timestamp"] = df["settlementDate"] + pd.to_timedelta((df["settlementPeriod"] - 1) * 30, unit="m")
df["bess_bid_volume_filled"] = df["bess_bid_volume"].fillna(0)
df["bess_net_volume"] = df["bess_offer_volume"].fillna(0) - df["bess_bid_volume_filled"]
df[TECH_COL] = df[TECH_COL].fillna("UNKNOWN")

n_total = len(df)
date_min = df["settlementDate"].min().strftime("%Y-%m-%d")
date_max = df["settlementDate"].max().strftime("%Y-%m-%d")
n_imp = int((df["price_imputed"] == 1).sum())
energy_missing_but_marginal = int((df["offer_volume_energy"].isna() & df[TECH_COL].notna()).sum())
TECH_ORDER = df[TECH_COL].value_counts().index.tolist()
palette = sns.color_palette("tab20", n_colors=max(len(TECH_ORDER), 3))
TECH_COLORS = {tech: palette[i] for i, tech in enumerate(TECH_ORDER)}
top_tech = df[TECH_COL].value_counts().reindex(TECH_ORDER, fill_value=0)
TOP_TECH_ORDER = TECH_ORDER[:8]


print("Building Section 1 ...")
sec1 = ""

overview = pd.DataFrame(
    {
        "Metric": [
            "Rows",
            "Date range",
            "Unique dates",
            "Price-imputed SPs",
            "Rows with any missing value",
            "Energy-missing but marginal-labelled SPs",
            "Unique marginal BMUs",
            "Median marginal price (GBP/MWh)",
        ],
        "Value": [
            f"{n_total:,}",
            f"{date_min} to {date_max}",
            f"{df['settlementDate'].nunique():,}",
            f"{n_imp:,} ({100*n_imp/n_total:.2f}%)",
            f"{int(df.isna().any(axis=1).sum()):,} ({100*df.isna().any(axis=1).mean():.2f}%)",
            f"{energy_missing_but_marginal:,}",
            f"{df['marginal_bmu'].nunique():,}",
            f"{df['marginal_price'].median():.2f}",
        ],
    }
)
sec1 += subsection(
    "1.1",
    "Dataset Snapshot",
    "High-level overview of the cleaned original stack panel, now displayed using your manually curated marginal technology register.",
    html_table(overview),
    "This EDA stays on the cleaned original stack panel and its original within-day marginal imputation logic, "
    "but the technology labels now come from your manually curated supplementary register.",
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
bars = ax1.bar(top_tech.index, top_tech.values, color=[TECH_COLORS[t] for t in top_tech.index], edgecolor="white", width=0.55)
for b, v in zip(bars, top_tech.values):
    ax1.text(b.get_x() + b.get_width() / 2, v + 120, f"{v:,}", ha="center", fontsize=8)
ax1.set_title("Marginal Technology Frequency", fontsize=11, fontweight="bold")
ax1.set_ylabel("Settlement Periods")
ax1.set_ylim(0, top_tech.max() * 1.12)
ax2.pie(
    top_tech.values,
    labels=[f"{k}\n{v:,} ({100*v/n_total:.1f}%)" for k, v in top_tech.items()],
    colors=[TECH_COLORS[t] for t in top_tech.index],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
ax2.set_title("Share of Marginal Events", fontsize=11, fontweight="bold")
fig.tight_layout()
sec1 += subsection(
    "1.2",
    "Marginal Technology Mix",
    "Full marginal technology mix using `marginal_tech_final`, i.e. your manually curated marginal register labels.",
    img_tag(fig),
    ", ".join([f"{tech}: {top_tech[tech]:,} ({100*top_tech[tech]/n_total:.1f}%)" for tech in TOP_TECH_ORDER]),
)


print("Building Section 2 ...")
sec2 = ""

fig, ax = plt.subplots(figsize=(12, 4))
imp_by_date = df.groupby("settlementDate")["price_imputed"].sum()
ax.bar(range(len(imp_by_date)), imp_by_date.values, color=C_ACCENT, edgecolor="white", width=0.85)
tick_step = max(1, len(imp_by_date) // 10)
ax.set_xticks(range(0, len(imp_by_date), tick_step))
ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in imp_by_date.index[::tick_step]], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Imputed SP count")
ax.set_title("Within-Day Marginal/Price Imputation by Date", fontsize=11, fontweight="bold")
fig.tight_layout()
peak_imp_date = imp_by_date.idxmax().strftime("%Y-%m-%d")
sec2 += subsection(
    "2.1",
    "Imputation Footprint",
    "How often the original pipeline had to carry marginal labels/price across the day.",
    img_tag(fig),
    f"`price_imputed == 1` in {n_imp:,} SPs ({100*n_imp/n_total:.2f}%). "
    f"The highest imputation day is {peak_imp_date} with {int(imp_by_date.max())} SPs filled.",
)

fig, ax = plt.subplots(figsize=(9, 4))
counts = pd.Series(
    {
        "All rows": n_total,
        "price_imputed == 1": n_imp,
        "offer_volume_energy missing\nbut marginal label present": energy_missing_but_marginal,
    }
)
bars = ax.bar(counts.index, counts.values, color=["#9ecae9", C_ACCENT, "#fdae6b"], edgecolor="white")
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + n_total * 0.01, f"{v:,}", ha="center", fontsize=8)
ax.set_title("Observed vs Filled Marginal Coverage", fontsize=11, fontweight="bold")
fig.tight_layout()
sec2 += subsection(
    "2.2",
    "Observed vs Filled Marginals",
    "This isolates the part of the original panel where marginal labels survive despite no observed energy-offer volume in the SP.",
    img_tag(fig),
    f"{energy_missing_but_marginal:,} SPs ({100*energy_missing_but_marginal/n_total:.2f}%) have `offer_volume_energy` missing but still carry a marginal label, "
    "which is the clearest signature of the original within-day filling rule.",
)

missing_table = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .head(12)
    .rename("missingValues")
    .reset_index()
    .rename(columns={"index": "column"})
)
missing_table["pctOfRows"] = (100 * missing_table["missingValues"] / n_total).round(2)
sec2 += subsection(
    "2.3",
    "Top Missingness Table",
    "Largest missingness blocks in the cleaned original panel.",
    html_table(missing_table),
    "These NaNs are not all equal: some reflect no bid/system activity, while others are denominator-driven shares.",
)


print("Building Section 3 ...")
sec3 = ""

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(df["marginal_price"], bins=80, kde=True, ax=axes[0, 0], color=C_GAS)
axes[0, 0].set_title("Marginal Price Distribution")
axes[0, 0].set_xlabel("GBP/MWh")

sns.histplot(df["offer_max_price"], bins=80, kde=False, ax=axes[0, 1], color=C_OTHER)
axes[0, 1].set_title("Offer Max Price Distribution")
axes[0, 1].set_xlabel("GBP/MWh")
axes[0, 1].set_xlim(df["offer_max_price"].clip(upper=df["offer_max_price"].quantile(0.99)).min(), df["offer_max_price"].quantile(0.99))

sns.boxplot(data=df, x=TECH_COL, y="marginal_price", order=TOP_TECH_ORDER, palette=TECH_COLORS, ax=axes[1, 0])
axes[1, 0].set_title("Marginal Price by Technology")
axes[1, 0].set_xlabel("")
axes[1, 0].set_ylabel("GBP/MWh")
axes[1, 0].tick_params(axis="x", rotation=35)

hourly = df.groupby("hour")["marginal_price"].mean()
axes[1, 1].plot(hourly.index, hourly.values, color=C_ACCENT, linewidth=2, marker="o", markersize=3)
axes[1, 1].set_title("Average Marginal Price by Hour")
axes[1, 1].set_xlabel("Hour")
axes[1, 1].set_ylabel("GBP/MWh")
axes[1, 1].set_xticks(range(0, 24, 2))

fig.tight_layout()
sec3 += subsection(
    "3.1",
    "Price Formation Overview",
    "Distributions and intraday shape of stack pricing in the original panel.",
    img_tag(fig),
    f"Median marginal price is {df['marginal_price'].median():.2f} GBP/MWh, "
    f"with a 99th percentile of {df['marginal_price'].quantile(0.99):.2f}.",
)

fig, ax = plt.subplots(figsize=(10, 5))
for tech in TOP_TECH_ORDER:
    sub = df[df[TECH_COL] == tech]
    ax.scatter(sub["offer_max_price"], sub["marginal_price"], alpha=0.18, s=8, label=tech, color=TECH_COLORS[tech], rasterized=True)
lim = max(df["offer_max_price"].quantile(0.99), df["marginal_price"].quantile(0.99)) * 1.02
ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
ax.set_xlim(-20, lim)
ax.set_ylim(-20, lim)
ax.set_xlabel("offer_max_price")
ax.set_ylabel("marginal_price")
ax.set_title("offer_max_price vs marginal_price", fontsize=11, fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout()
gap_gt_100 = int(((df["offer_max_price"] - df["marginal_price"]).abs() > 100).sum())
sec3 += subsection(
    "3.2",
    "Offer Max vs Marginal Price",
    "The marginal price should typically sit on or below the highest qualifying offer in the SP.",
    img_tag(fig),
    f"{gap_gt_100:,} SPs show an absolute gap greater than 100 GBP between `offer_max_price` and `marginal_price`, "
    "which is a useful set to spot-check for repricing or edge-case stack behavior.",
)


print("Building Section 4 ...")
sec4 = ""

top5_tech = TECH_ORDER[:5]
daily_top5 = (
    df.groupby(["settlementDate", TECH_COL])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=top5_tech, fill_value=0)
)
daily_top5_share = daily_top5.div(daily_top5.sum(axis=1), axis=0).rolling(7, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(13, 4))
for tech in top5_tech:
    ax.plot(daily_top5_share.index, daily_top5_share[tech], linewidth=1.8, label=tech, color=TECH_COLORS[tech])
ax.set_title("7-Day Rolling Marginal Share of Top Technologies", fontsize=11, fontweight="bold")
ax.set_ylabel("Share of daily marginal rows")
ax.set_ylim(0, min(1.0, daily_top5_share.max().max() * 1.1))
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
sec4 += subsection(
    "4.1",
    "Technology Leadership Over Time",
    "A rolling view of which technologies are most often setting the marginal label through the sample.",
    img_tag(fig),
    "This helps us see whether leadership at the margin is steady, seasonal, or switches sharply across the 2023-2025 period.",
)

price_state = (
    df[df[TECH_COL].isin(TOP_TECH_ORDER)]
    .assign(
        system_state=lambda x: np.where(
            x["niv"] < 0, "Short", np.where(x["niv"] > 0, "Long", "Balanced")
        )
    )
    .groupby([TECH_COL, "system_state"])["marginal_price"]
    .median()
    .unstack(fill_value=np.nan)
    .reindex(TOP_TECH_ORDER)
)
state_cols = [c for c in ["Short", "Long", "Balanced"] if c in price_state.columns]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(price_state.index))
width = 0.24 if len(state_cols) >= 3 else 0.35
state_palette = {"Short": "#E45756", "Long": "#4C78A8", "Balanced": "#72B7B2"}
for i, state in enumerate(state_cols):
    ax.bar(x + (i - (len(state_cols) - 1) / 2) * width, price_state[state].values, width=width, label=state, color=state_palette[state], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(price_state.index, rotation=35, ha="right")
ax.set_ylabel("Median marginal price (GBP/MWh)")
ax.set_title("Median Marginal Price by Technology and System State", fontsize=11, fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout()
sec4 += subsection(
    "4.2",
    "Technology Pricing by System State",
    "Median marginal prices split by whether NIV indicates a short, long, or balanced system.",
    img_tag(fig),
    "This is a compact way to compare whether technologies look systematically more expensive when they appear under tighter system conditions.",
)

top_bmu = df["marginal_bmu"].value_counts().head(20)
top_bmu_tech = [df.loc[df["marginal_bmu"] == bmu, TECH_COL].mode().iat[0] for bmu in top_bmu.index]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(top_bmu)), top_bmu.values, color=[TECH_COLORS.get(t, C_OTHER) for t in top_bmu_tech], edgecolor="white")
ax.set_yticks(range(len(top_bmu)))
ax.set_yticklabels(top_bmu.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Marginal SP count")
ax.set_title("Top 20 Marginal BMUs", fontsize=11, fontweight="bold")
for bar, v in zip(bars, top_bmu.values):
    ax.text(v + 15, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=8)
fig.tight_layout()
sec4 += subsection(
    "4.3",
    "Marginal BMU Concentration",
    "Which individual units dominate marginal events under the original file's logic.",
    img_tag(fig),
    f"The most frequent marginal BMU appears {int(top_bmu.iloc[0]):,} times, and the top 20 units account for {100*top_bmu.sum()/n_total:.1f}% of all rows.",
)


print("Building Section 5 ...")
sec5 = ""

monthly_share = df.groupby(["month_name", TECH_COL]).size().unstack(fill_value=0).reindex(columns=TOP_TECH_ORDER, fill_value=0)
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_share = monthly_share.reindex([m for m in month_order if m in monthly_share.index])
monthly_pct = monthly_share.div(monthly_share.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(12, 4))
bottom = np.zeros(len(monthly_pct))
for tech in TOP_TECH_ORDER:
    vals = monthly_pct[tech].values
    ax.bar(monthly_pct.index, vals, bottom=bottom, color=TECH_COLORS[tech], edgecolor="white", label=tech)
    bottom += vals
ax.set_ylim(0, 1)
ax.set_ylabel("Share of monthly marginal rows")
ax.set_title("Monthly Marginal Technology Share", fontsize=11, fontweight="bold")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
sec5 += subsection(
    "5.1",
    "Monthly Technology Share",
    "How the main marginal technologies move across the full 2023-2025 window.",
    img_tag(fig),
    "This is useful for spotting whether technologies like PS, BATTERY, CCGT, OCGT, and other manually curated labels gain share gradually, seasonally, or in bursts.",
)

fig, ax = plt.subplots(figsize=(12, 5))
for tech in TOP_TECH_ORDER:
    hr = df[df[TECH_COL] == tech].groupby("hour").size()
    ax.plot(hr.index, hr.values, linewidth=2, marker="o", markersize=3, color=TECH_COLORS[tech], label=tech)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour")
ax.set_ylabel("Marginal event count")
ax.set_title("Intraday Marginal Frequency by Technology", fontsize=11, fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout()
sec5 += subsection(
    "5.2",
    "Intraday Technology Timing",
    "When different technologies show up at the margin across the day.",
    img_tag(fig),
    "This is especially handy for seeing whether different marginal technologies show distinct intraday timing rather than being flattened into a single residual category.",
)

fig, ax = plt.subplots(figsize=(11, 4))
short_long = pd.DataFrame(
    {
        "Short system": [int((df.loc[df[TECH_COL] == tech, "niv"] < 0).sum()) for tech in TOP_TECH_ORDER],
        "Long system": [int((df.loc[df[TECH_COL] == tech, "niv"] > 0).sum()) for tech in TOP_TECH_ORDER],
    },
    index=TOP_TECH_ORDER,
)
short_long.plot(kind="bar", stacked=True, color=[C_GAS, C_BESS], edgecolor="white", ax=ax)
ax.set_ylabel("Settlement periods")
ax.set_title("Marginal Technology by System State Sign of NIV", fontsize=11, fontweight="bold")
ax.legend(frameon=False)
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
sec5 += subsection(
    "5.3",
    "Technology and System State",
    "How often each marginal technology appears when the system is short versus long.",
    img_tag(fig),
    "This is a simple but important selection-effect check before you compare technologies in regressions.",
)


print("Assembling HTML ...")
nav_items = [
    ("s1", "1. Overview"),
    ("s2", "2. Imputation & Missingness"),
    ("s3", "3. Price Formation"),
    ("s4", "4. Technology Behavior"),
    ("s5", "5. Seasonality & System State"),
]
nav_html = "\n".join(f'<li><a href="#{sid}">{label}</a></li>' for sid, label in nav_items)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EDA Report - master_panel_2023_2025 original</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;font-size:13px;color:#222;background:#F5F5F5;display:flex}}
#sidebar{{width:230px;min-width:230px;height:100vh;position:sticky;top:0;background:#1E293B;overflow-y:auto;padding:20px 0;flex-shrink:0}}
#sidebar h1{{color:#94A3B8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;padding:0 16px 12px}}
#sidebar ul{{list-style:none;padding:0}}
#sidebar li a{{display:block;padding:8px 16px;color:#CBD5E1;text-decoration:none;font-size:12px;border-left:3px solid transparent;transition:all .15s}}
#sidebar li a:hover{{background:#334155;color:#F8FAFC;border-left-color:#F58518}}
#content{{flex:1;min-width:0;padding:24px 32px;max-width:1320px}}
h2{{font-size:18px;font-weight:700;color:#1E293B;margin:32px 0 12px;padding-bottom:8px;border-bottom:2px solid #E2E8F0}}
h3{{font-size:14px;font-weight:600;color:#334155;margin:24px 0 8px}}
.subsection{{background:white;border-radius:8px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.07)}}
.note-above{{color:#475569;font-size:12px;line-height:1.6;margin-bottom:14px;padding:10px 14px;background:#F8FAFC;border-left:3px solid #F58518;border-radius:4px}}
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
  <h1>Original Stack EDA</h1>
  <ul>{nav_html}</ul>
</nav>
<main id="content">
  <h1 style="font-size:22px;font-weight:800;color:#1E293B;margin-bottom:4px">EDA - Original `master_panel_2023_2025`</h1>
  <p style="color:#64748B;font-size:12px;margin-bottom:8px">
    Cleaned original stack with manual marginal register applied | {date_min} to {date_max} | {n_total:,} settlement periods | full marginal technology breakdown
  </p>
  {section("s1", "1. Overview", sec1)}
  {section("s2", "2. Imputation and Missingness", sec2)}
  {section("s3", "3. Price Formation", sec3)}
  {section("s4", "4. Technology-Specific Behavior", sec4)}
  {section("s5", "5. Seasonality and System State", sec5)}
</main>
</body>
</html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"Saved -> {OUT_HTML}")
