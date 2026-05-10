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
INPUT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "q1_2026"
    / "ispstack_marginal_action_q1_2026"
    / "marginal_action_sp_q1_2026_identity_final.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "q1_2026" / "ispstack_marginal_action_q1_2026"
OUT_PANEL = OUTPUT_DIR / "marginal_action_sp_q1_2026_identity_unknown_numeric.parquet"
OUT_SUMMARY = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "q1_2026_ispstack_identity_unknown_numeric_summary.json"
OUT_HTML = PROJECT_ROOT / "outputs" / "reports" / "eda_report_q1_2026_ispstack_identity_unknown_numeric.html"


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


def load_and_finalize_panel() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(INPUT_PANEL).copy()
    unresolved_mask = df["marginal_identity_resolution_rule_final"].eq("numeric_winner_unresolved_kept")

    df["marginal_identity_unknown_numeric"] = unresolved_mask
    df.loc[unresolved_mask, "marginal_id_final"] = "UNKNOWN_NUMERIC"
    df.loc[unresolved_mask, "marginal_bmu_final"] = "UNKNOWN_NUMERIC"
    df.loc[unresolved_mask, "marginal_generator_id_final"] = "GEN_UNKNOWN_NUMERIC"
    df.loc[unresolved_mask, "marginal_generator_label_final"] = "UNKNOWN_NUMERIC"
    df.loc[unresolved_mask, "marginal_tech_final"] = "UNKNOWN_NUMERIC"
    df.loc[unresolved_mask, "marginal_identity_resolution_rule_final"] = "numeric_winner_labeled_unknown_numeric"

    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    df["timestamp"] = df["settlementDate"] + pd.to_timedelta((df["settlementPeriod"] - 1) * 30, unit="m")
    df["hour"] = ((df["settlementPeriod"] - 1) // 2).astype(int)
    df["month_name"] = df["settlementDate"].dt.strftime("%b")
    df["day_of_week"] = df["settlementDate"].dt.dayofweek
    df["dow_name"] = df["settlementDate"].dt.strftime("%a")

    summary = {
        "rows": int(len(df)),
        "date_min": df["settlementDate"].min().strftime("%Y-%m-%d"),
        "date_max": df["settlementDate"].max().strftime("%Y-%m-%d"),
        "winner_side_counts": {str(k): int(v) for k, v in df["marginal_side_winner"].value_counts(dropna=False).to_dict().items()},
        "identity_substituted_rows": int(df["marginal_identity_substituted"].fillna(False).sum()),
        "unknown_numeric_rows": int(df["marginal_identity_unknown_numeric"].sum()),
        "unknown_numeric_share": float(df["marginal_identity_unknown_numeric"].mean()),
        "unique_final_generators": int(df["marginal_generator_label_final"].nunique(dropna=True)),
        "unique_final_techs": int(df["marginal_tech_final"].nunique(dropna=True)),
    }
    return df, summary


def save_outputs(df: pd.DataFrame, summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PANEL, index=False)
    df.to_csv(OUT_PANEL.with_suffix(".csv"), index=False)
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def build_report(df: pd.DataFrame, summary: dict) -> str:
    n_total = len(df)
    tech_counts = df["marginal_tech_final"].fillna("UNKNOWN").value_counts()
    top_tech = tech_counts.head(10)
    tech_palette = sns.color_palette("tab20", n_colors=max(10, len(top_tech)))
    tech_colors = {tech: tech_palette[i] for i, tech in enumerate(top_tech.index)}
    tech_colors["UNKNOWN_NUMERIC"] = C_UNKNOWN

    sec1 = ""
    overview = pd.DataFrame(
        {
            "Metric": [
                "Settlement periods",
                "Date range",
                "Winner-side offer share",
                "Winner-side bid share",
                "Identity-substituted rows",
                "Unknown-numeric rows",
                "Unique final generator labels",
                "Unique final tech labels",
            ],
            "Value": [
                f"{n_total:,}",
                f"{summary['date_min']} to {summary['date_max']}",
                f"{100 * summary['winner_side_counts'].get('offer', 0) / n_total:.2f}%",
                f"{100 * summary['winner_side_counts'].get('bid', 0) / n_total:.2f}%",
                f"{summary['identity_substituted_rows']:,}",
                f"{summary['unknown_numeric_rows']:,} ({100 * summary['unknown_numeric_share']:.2f}%)",
                f"{summary['unique_final_generators']:,}",
                f"{summary['unique_final_techs']:,}",
            ],
        }
    )
    sec1 += subsection(
        "1.1",
        "Panel Snapshot",
        "This is the Q1 2026 ISPSTACK winner-action panel after numeric-ID cleanup. The last unresolved numeric winners are now explicitly labeled `UNKNOWN_NUMERIC` rather than left as raw digits.",
        html_table(overview),
        f"We rescued most of the identity issue through same-side substitution, and only {summary['unknown_numeric_rows']} rows remain as explicit unknown numeric winners.",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(top_tech.index, top_tech.values, color=[tech_colors.get(t, C_ACCENT) for t in top_tech.index], edgecolor="white")
    for b, v in zip(bars, top_tech.values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(5, top_tech.max() * 0.01), f"{v:,}", ha="center", fontsize=8)
    ax.set_title("Top Final Marginal Technologies", fontsize=11, fontweight="bold")
    ax.set_ylabel("Settlement periods")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    sec1 += subsection(
        "1.2",
        "Final Technology Mix",
        "Top marginal technologies after applying the winner rule, generator/technology mapping, numeric-ID substitution, and final `UNKNOWN_NUMERIC` stamping.",
        img_tag(fig),
        ", ".join([f"{tech}: {count:,}" for tech, count in top_tech.items()]),
    )

    sec2 = ""
    winner_counts = df["marginal_side_winner"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    side_order = [s for s in ["offer", "bid", "mixed_or_ambiguous"] if s in winner_counts.index]
    colors = [C_OFFER if s == "offer" else C_BID if s == "bid" else C_UNKNOWN for s in side_order]
    ax.bar(side_order, winner_counts.reindex(side_order).values, color=colors, edgecolor="white")
    ax.set_title("Winner Side Split", fontsize=11, fontweight="bold")
    ax.set_ylabel("Settlement periods")
    fig.tight_layout()
    sec2 += subsection(
        "2.1",
        "Offer vs Bid Winners",
        "The winner-takes-price rule remains strongly offer-led in Q1, but the bid-led periods are still worth keeping an eye on.",
        img_tag(fig),
        ", ".join([f"{side}: {int(winner_counts[side]):,} ({100 * winner_counts[side] / n_total:.2f}%)" for side in side_order]),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["marginal_price_winner"].dropna(), bins=80, ax=ax, color=C_OFFER)
    ax.set_title("Winner Marginal Price Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("GBP/MWh")
    fig.tight_layout()
    sec2 += subsection(
        "2.2",
        "Winner Price Distribution",
        "Distribution of the selected winner price before any external fundamentals are brought in.",
        img_tag(fig),
        f"Median winner price is {df['marginal_price_winner'].median():.2f} GBP/MWh and the 99th percentile is {df['marginal_price_winner'].quantile(0.99):.2f}.",
    )

    price_by_side = (
        df.groupby("marginal_side_winner")["marginal_price_winner"]
        .agg(["median", "mean", "count"])
        .round(2)
        .reset_index()
    )
    sec2 += subsection(
        "2.3",
        "Winner Price by Side",
        "Compact summary of how winner prices differ between offer-led and bid-led periods.",
        html_table(price_by_side),
        "This is a good early check on whether bid-led periods look mechanically weird or just economically different.",
    )

    sec3 = ""
    gen_counts = df["marginal_generator_label_final"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(gen_counts)), gen_counts.values, color=C_ACCENT, edgecolor="white")
    ax.set_yticks(range(len(gen_counts)))
    ax.set_yticklabels(gen_counts.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Settlement periods")
    ax.set_title("Top 20 Final Generator Labels", fontsize=11, fontweight="bold")
    fig.tight_layout()
    sec3 += subsection(
        "3.1",
        "Generator Concentration",
        "The most frequent final generator labels after the numeric-ID cleanup.",
        img_tag(fig),
        f"`UNKNOWN_NUMERIC` appears {int(gen_counts.get('UNKNOWN_NUMERIC', 0)):,} times in the top-20 table." if "UNKNOWN_NUMERIC" in gen_counts.index else "`UNKNOWN_NUMERIC` does not appear in the top-20 final generator labels.",
    )

    unknown_daily = (
        df.groupby("settlementDate")["marginal_identity_unknown_numeric"]
        .sum()
        .reset_index(name="unknownRows")
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(unknown_daily["settlementDate"].astype(str), unknown_daily["unknownRows"], color=C_UNKNOWN, edgecolor="white")
    tick_step = max(1, len(unknown_daily) // 10)
    ax.set_xticks(range(0, len(unknown_daily), tick_step))
    ax.set_xticklabels(unknown_daily["settlementDate"].dt.strftime("%Y-%m-%d").iloc[::tick_step], rotation=35, ha="right", fontsize=8)
    ax.set_title("Remaining UNKNOWN_NUMERIC Rows by Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    sec3 += subsection(
        "3.2",
        "Residual Unknowns",
        "Only a very small set of winner rows still have no named substitute and are now explicitly labeled `UNKNOWN_NUMERIC`.",
        img_tag(fig),
        f"There are {summary['unknown_numeric_rows']} such rows in total, so the residual identity problem is now tiny and visible rather than hidden.",
    )

    nav_items = [
        ("s1", "1. Overview"),
        ("s2", "2. Winner Pricing"),
        ("s3", "3. Final Labels"),
    ]
    nav_html = "\n".join(f'<li><a href="#{sid}">{label}</a></li>' for sid, label in nav_items)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EDA Report - Q1 2026 ISPSTACK identity final</title>
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
  <h1>Q1 ISPSTACK EDA</h1>
  <ul>{nav_html}</ul>
</nav>
<main id="content">
  <h1 style="font-size:22px;font-weight:800;color:#1E293B;margin-bottom:4px">EDA - Q1 2026 ISPSTACK Winner Panel</h1>
  <p style="color:#64748B;font-size:12px;margin-bottom:8px">
    Identity-final panel with `UNKNOWN_NUMERIC` fallback | {summary['date_min']} to {summary['date_max']} | {n_total:,} settlement periods
  </p>
  {section("s1", "1. Overview", sec1)}
  {section("s2", "2. Winner Pricing", sec2)}
  {section("s3", "3. Final Labels", sec3)}
</main>
</body>
</html>"""
    return html


def main() -> None:
    sns.set_theme(style="whitegrid")
    df, summary = load_and_finalize_panel()
    save_outputs(df, summary)
    html = build_report(df, summary)
    OUT_HTML.write_text(html, encoding="utf-8")

    print("=" * 72)
    print("Q1 ISPSTACK Identity Finalization + EDA Complete")
    print("=" * 72)
    print(f"Rows                      : {summary['rows']:,}")
    print(f"Unknown numeric rows      : {summary['unknown_numeric_rows']:,}")
    print(f"Identity-substituted rows : {summary['identity_substituted_rows']:,}")
    print(f"Final panel               : {OUT_PANEL}")
    print(f"EDA report                : {OUT_HTML}")


if __name__ == "__main__":
    main()
