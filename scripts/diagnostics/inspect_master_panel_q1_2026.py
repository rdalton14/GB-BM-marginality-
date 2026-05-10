import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_PARQUET = PROJECT_ROOT / "data" / "processed" / "q1_2026" / "master_panel_q1_2026_prototype.parquet"
DEFAULT_PANEL_CSV = PROJECT_ROOT / "data" / "processed" / "q1_2026" / "master_panel_q1_2026_prototype.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "q1_2026_prototype_checks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Optional path to CSV or parquet panel file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for saved figures.")
    return parser.parse_args()


def load_panel(panel_path: Path | None) -> pd.DataFrame:
    if panel_path is None:
        panel_path = DEFAULT_PANEL_PARQUET if DEFAULT_PANEL_PARQUET.exists() else DEFAULT_PANEL_CSV

    if panel_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(panel_path)
    else:
        df = pd.read_csv(panel_path)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    df["timestamp"] = df["settlementDate"] + pd.to_timedelta((df["settlementPeriod"] - 1) * 30, unit="m")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("Q1 2026 master panel prototype inspection")
    print(f"shape = {df.shape}")
    print(f"date range = {df['settlementDate'].min().date()} to {df['settlementDate'].max().date()}")
    dup_count = df.duplicated(["settlementDate", "settlementPeriod"]).sum()
    print(f"duplicate SP keys = {dup_count}")
    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    rows_with_missing = df[df.isna().any(axis=1)].copy()
    print(f"\nrows with any missing values = {len(rows_with_missing)}")
    if not rows_with_missing.empty:
        missing_by_date = rows_with_missing.groupby(rows_with_missing["settlementDate"].dt.date).size()
        print("\nDates with any missing values:")
        print(missing_by_date.to_string())

    corr_cols = [
        "systemPrice",
        "netImbalanceVolume",
        "dayAheadPrice",
        "demandForecastError",
        "windForecastError",
        "solarForecastError",
        "lossOfLoadProbability",
        "deratedMargin",
    ]
    print("\nCorrelation preview:")
    print(df[corr_cols].corr().round(3).to_string())


def save_missingness_plot(df: pd.DataFrame, output_dir: Path) -> None:
    daily_missing = (
        df.assign(any_missing=df.isna().any(axis=1))
        .groupby("settlementDate", as_index=False)["any_missing"]
        .sum()
        .rename(columns={"any_missing": "rowsWithAnyMissing"})
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=daily_missing, x="settlementDate", y="rowsWithAnyMissing", color="#4C78A8", ax=ax)
    ax.set_title("Rows With Any Missing Values by Date")
    ax.set_xlabel("settlementDate")
    ax.set_ylabel("rows with any missing")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "missingness_by_date.png", dpi=150)
    plt.close(fig)


def save_distribution_plots(df: pd.DataFrame, output_dir: Path) -> None:
    cols = [
        "systemPrice",
        "netImbalanceVolume",
        "demandForecastError",
        "windForecastError",
        "solarForecastError",
        "lossOfLoadProbability",
        "deratedMargin",
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), bins=40, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(col)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "core_distributions.png", dpi=150)
    plt.close(fig)


def save_time_series_plots(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    sns.lineplot(data=df, x="timestamp", y="systemPrice", ax=axes[0], color="#E45756", linewidth=0.8)
    axes[0].set_title("System Price Over Q1 2026")
    axes[0].set_ylabel("GBP/MWh")

    sns.lineplot(data=df, x="timestamp", y="netImbalanceVolume", ax=axes[1], color="#4C78A8", linewidth=0.8)
    axes[1].set_title("Net Imbalance Volume Over Q1 2026")
    axes[1].set_ylabel("MWh")

    daily = (
        df.groupby("settlementDate", as_index=False)[
            ["demandOutturn", "dayAheadNationalDemandForecast", "dayAheadPrice"]
        ]
        .mean()
    )
    sns.lineplot(data=daily, x="settlementDate", y="demandOutturn", ax=axes[2], label="demandOutturn", color="#54A24B")
    sns.lineplot(
        data=daily,
        x="settlementDate",
        y="dayAheadNationalDemandForecast",
        ax=axes[2],
        label="dayAheadNationalDemandForecast",
        color="#F58518",
    )
    axes[2].set_title("Daily Mean Demand: Actual vs Forecast")
    axes[2].set_ylabel("MW")
    axes[2].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_dir / "time_series_overview.png", dpi=150)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    corr_cols = [
        "systemPrice",
        "netImbalanceVolume",
        "dayAheadPrice",
        "demandForecastError",
        "windForecastError",
        "solarForecastError",
        "lossOfLoadProbability",
        "deratedMargin",
    ]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_panel(args.input)
    print_summary(df)
    save_missingness_plot(df, output_dir)
    save_distribution_plots(df, output_dir)
    save_time_series_plots(df, output_dir)
    save_correlation_heatmap(df, output_dir)
    print(f"\nSaved figures to {output_dir}")


if __name__ == "__main__":
    main()
