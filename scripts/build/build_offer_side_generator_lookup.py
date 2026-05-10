from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "marginal_action_sp_2023_2025_offer_side_with_system_price.csv"
)
REGISTER_CSV = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ispstack"
    / "generator_family_register_2023_2025.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_2023_2025"
    / "ispstack_marginal_action_2023_2025"
    / "generator_lookup"
)
LOOKUP_CSV = OUTPUT_DIR / "offer_side_generator_label_lookup.csv"
SUMMARY_JSON = OUTPUT_DIR / "offer_side_generator_label_lookup_summary.json"

PANEL_COLS = [
    "marginal_generator_label_final",
    "marginal_generator_id_final",
    "marginal_bmu_final",
    "marginal_offer_plant",
    "marginal_tech_final",
]

REGISTER_PREFERRED_COLS = [
    "generator_label",
    "generator_id",
    "generator_family",
    "generator_family_label",
    "plant_name",
    "station_name",
    "display_name",
    "technology",
    "tech",
    "bmu",
]


def join_unique(values: pd.Series, sep: str = " | ", max_items: int = 12) -> str:
    items = [str(v).strip() for v in values.dropna().astype(str) if str(v).strip()]
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    if len(seen) > max_items:
        shown = seen[:max_items]
        shown.append(f"... (+{len(seen) - max_items} more)")
        return sep.join(shown)
    return sep.join(seen)


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV, usecols=PANEL_COLS, low_memory=False)
    df["marginal_generator_label_final"] = df["marginal_generator_label_final"].astype("string").str.strip()
    df = df[df["marginal_generator_label_final"].notna() & (df["marginal_generator_label_final"] != "")]
    return df


def build_panel_lookup(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("marginal_generator_label_final", dropna=False)
    lookup = grouped.agg(
        marginal_rows=("marginal_generator_label_final", "size"),
        generator_ids=("marginal_generator_id_final", join_unique),
        bmus=("marginal_bmu_final", join_unique),
        offer_plants=("marginal_offer_plant", join_unique),
        techs=("marginal_tech_final", join_unique),
        n_unique_bmus=("marginal_bmu_final", lambda s: s.dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
    ).reset_index()
    lookup = lookup.rename(columns={"marginal_generator_label_final": "generator_label_final"})
    lookup["row_share_pct"] = 100 * lookup["marginal_rows"] / len(df)
    return lookup.sort_values("marginal_rows", ascending=False).reset_index(drop=True)


def load_register() -> pd.DataFrame | None:
    if not REGISTER_CSV.exists():
        return None
    register = pd.read_csv(REGISTER_CSV, low_memory=False)
    available = [c for c in REGISTER_PREFERRED_COLS if c in register.columns]
    if not available:
        return None
    return register[available].copy()


def build_register_lookup(register: pd.DataFrame) -> pd.DataFrame:
    label_col = None
    for candidate in ["generator_label", "generator_family_label", "generator_family"]:
        if candidate in register.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("No usable generator label column found in register.")

    agg_spec: dict[str, tuple[str, object]] = {}
    if "generator_id" in register.columns:
        agg_spec["register_generator_ids"] = ("generator_id", join_unique)
    if "bmu" in register.columns:
        agg_spec["register_bmus"] = ("bmu", join_unique)

    plant_col = next((c for c in ["plant_name", "station_name", "display_name"] if c in register.columns), None)
    if plant_col is not None:
        agg_spec["register_plant_names"] = (plant_col, join_unique)

    tech_col = next((c for c in ["technology", "tech"] if c in register.columns), None)
    if tech_col is not None:
        agg_spec["register_techs"] = (tech_col, join_unique)

    grouped = register.groupby(label_col, dropna=False)
    lookup = grouped.agg(**agg_spec).reset_index()
    lookup = lookup.rename(columns={label_col: "generator_label_final"})
    return lookup


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    panel_lookup = build_panel_lookup(panel)

    register = load_register()
    if register is not None:
        register_lookup = build_register_lookup(register)
        final_lookup = panel_lookup.merge(register_lookup, on="generator_label_final", how="left")
    else:
        final_lookup = panel_lookup.copy()

    final_lookup.to_csv(LOOKUP_CSV, index=False)

    summary = {
        "input_panel_csv": str(PANEL_CSV),
        "input_register_csv": str(REGISTER_CSV),
        "rows_in_panel": int(len(panel)),
        "unique_generator_labels": int(panel_lookup["generator_label_final"].nunique()),
        "lookup_csv": str(LOOKUP_CSV),
        "top_10_labels": final_lookup.head(10)["generator_label_final"].tolist(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
