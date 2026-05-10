from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

BASE_DIR = PROJECT_ROOT / "data" / "processed" / "full_2023_2025" / "ispstack_marginal_action_2023_2025"
RAW_PATH = BASE_DIR / "accepted_actions_long_raw_2023_2025.parquet"
RANKED_PATH = BASE_DIR / "accepted_actions_ranked_2023_2025.parquet"
CANDIDATES_PATH = BASE_DIR / "marginal_candidates_sp_2023_2025.parquet"
ACTION_PATH = BASE_DIR / "marginal_action_sp_2023_2025.parquet"

BMU_REF_PATH = PROJECT_ROOT / "data" / "interim" / "ispstack" / "bmu_reference_canonical_2023_2025.csv"
GEN_REG_PATH = PROJECT_ROOT / "data" / "interim" / "ispstack" / "generator_family_register_2023_2025.csv"
MARGINAL_REG_PATH = PROJECT_ROOT / "data" / "interim" / "ispstack" / "marginal_bmu_supplementary_register_2023_2025.csv"

OUT_REGISTER = PROJECT_ROOT / "data" / "interim" / "ispstack" / "ispstack_id_resolution_2023_2025.csv"
OUT_REGISTER_PARQUET = OUT_REGISTER.with_suffix(".parquet")
OUT_REVIEW = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_id_resolution_2023_2025_review_queue.csv"
OUT_AUDIT = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_id_resolution_2023_2025_audit.csv"
OUT_FINAL_AUDIT = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_id_resolution_2023_2025_identity_final_audit.csv"
OUT_SUMMARY = PROJECT_ROOT / "data" / "diagnostics" / "audits" / "ispstack_id_resolution_2023_2025_summary.json"
OUT_ACTION_APPLIED = BASE_DIR / "marginal_action_sp_2023_2025_resolved.parquet"
OUT_CANDIDATES_APPLIED = BASE_DIR / "marginal_candidates_sp_2023_2025_resolved.parquet"
OUT_ACTION_FINAL = BASE_DIR / "marginal_action_sp_2023_2025_identity_final.parquet"
OUT_ACTION_UNKNOWN = BASE_DIR / "marginal_action_sp_2023_2025_identity_unknown_numeric.parquet"


def classify_id(raw_id: str) -> str:
    value = str(raw_id)
    if value.isdigit():
        return "numeric_only"
    if value.startswith(("2__", "V__", "E_", "T_", "I_", "M_", "B_", "W_")):
        return "structured_stack_id"
    if "-" in value or "_" in value:
        return "structured_other"
    return "unstructured_other"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(RAW_PATH)
    ranked = pd.read_parquet(RANKED_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)
    action = pd.read_parquet(ACTION_PATH)
    return raw, ranked, candidates, action


def build_usage_tables(raw: pd.DataFrame, candidates: pd.DataFrame, action: pd.DataFrame) -> pd.DataFrame:
    raw_usage = (
        raw.groupby("id")
        .agg(
            raw_row_count=("id", "size"),
            raw_abs_volume_total=("abs_volume", "sum"),
            raw_offer_rows=("is_offer", "sum"),
            raw_bid_rows=("is_bid", "sum"),
        )
        .reset_index()
        .rename(columns={"id": "raw_id"})
    )

    offer_cand = (
        candidates["marginal_offer_id"]
        .dropna()
        .value_counts()
        .rename_axis("raw_id")
        .reset_index(name="offer_candidate_sp_count")
    )
    bid_cand = (
        candidates["marginal_bid_id"]
        .dropna()
        .value_counts()
        .rename_axis("raw_id")
        .reset_index(name="bid_candidate_sp_count")
    )
    winner_usage = (
        action["marginal_id_winner"]
        .dropna()
        .value_counts()
        .rename_axis("raw_id")
        .reset_index(name="winner_sp_count")
    )

    usage = raw_usage.merge(offer_cand, on="raw_id", how="left")
    usage = usage.merge(bid_cand, on="raw_id", how="left")
    usage = usage.merge(winner_usage, on="raw_id", how="left")
    for col in ["offer_candidate_sp_count", "bid_candidate_sp_count", "winner_sp_count"]:
        usage[col] = usage[col].fillna(0).astype(int)
    usage["raw_id"] = usage["raw_id"].astype(str)
    return usage


def load_reference_maps() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bmu_ref = pd.read_csv(BMU_REF_PATH)
    gen_reg = pd.read_csv(GEN_REG_PATH)
    marginal_reg = pd.read_csv(MARGINAL_REG_PATH)
    return bmu_ref, gen_reg, marginal_reg


def build_resolution_register(
    usage: pd.DataFrame, bmu_ref: pd.DataFrame, gen_reg: pd.DataFrame, marginal_reg: pd.DataFrame
) -> pd.DataFrame:
    tech_map = (
        marginal_reg[["elexon_bmu_id", "reg_fuel_type"]]
        .drop_duplicates()
        .rename(columns={"elexon_bmu_id": "raw_id", "reg_fuel_type": "resolved_tech_from_marginal"})
    )
    bmu_map = (
        bmu_ref[["id", "tech_final", "mapping_source", "mapping_confidence", "is_unmatched"]]
        .drop_duplicates()
        .rename(
            columns={
                "id": "raw_id",
                "tech_final": "resolved_tech_from_bmu_ref",
                "mapping_source": "bmu_mapping_source",
                "mapping_confidence": "bmu_mapping_confidence",
                "is_unmatched": "bmu_is_unmatched",
            }
        )
    )
    gen_map = (
        gen_reg[["elexon_bmu_id", "generator_id_final", "generator_label_final", "collapse_decision", "collapse_rule"]]
        .drop_duplicates()
        .rename(
            columns={
                "elexon_bmu_id": "raw_id",
                "generator_id_final": "resolved_generator_id",
                "generator_label_final": "resolved_generator_label",
                "collapse_decision": "generator_collapse_decision",
                "collapse_rule": "generator_collapse_rule",
            }
        )
    )

    reg = usage.merge(tech_map, on="raw_id", how="left")
    reg = reg.merge(bmu_map, on="raw_id", how="left")
    reg = reg.merge(gen_map, on="raw_id", how="left")

    reg["id_type"] = reg["raw_id"].map(classify_id)
    reg["resolved_bmu"] = reg["raw_id"].where(reg["resolved_tech_from_bmu_ref"].notna() | reg["resolved_tech_from_marginal"].notna())
    reg["resolved_tech"] = reg["resolved_tech_from_marginal"].fillna(reg["resolved_tech_from_bmu_ref"])

    reg["resolution_rule"] = "unresolved"
    reg.loc[reg["resolved_bmu"].notna() & reg["resolved_generator_id"].notna(), "resolution_rule"] = "direct_bmu_plus_generator_register"
    reg.loc[reg["resolved_bmu"].notna() & reg["resolved_generator_id"].isna(), "resolution_rule"] = "direct_bmu_only"
    reg.loc[reg["id_type"] == "numeric_only", "resolution_rule"] = "numeric_raw_id_manual_review"
    reg.loc[reg["resolved_bmu"].isna() & reg["id_type"].isin(["structured_stack_id", "structured_other"]), "resolution_rule"] = "structured_unresolved_manual_review"

    reg["needs_manual_review"] = reg["resolution_rule"].isin(
        ["numeric_raw_id_manual_review", "structured_unresolved_manual_review", "unresolved"]
    )

    reg["resolved_bmu_final"] = reg["resolved_bmu"]
    reg.loc[reg["id_type"] == "numeric_only", "resolved_bmu_final"] = reg["raw_id"].map(lambda x: f"UNKNOWN_STACK_ID_{x}")
    reg["resolved_generator_id_final"] = reg["resolved_generator_id"]
    reg.loc[reg["id_type"] == "numeric_only", "resolved_generator_id_final"] = reg["raw_id"].map(lambda x: f"GEN_UNKNOWN_STACK_ID_{x}")
    reg["resolved_generator_label_final"] = reg["resolved_generator_label"]
    reg.loc[reg["id_type"] == "numeric_only", "resolved_generator_label_final"] = reg["raw_id"].map(lambda x: f"UNKNOWN_STACK_ID_{x}")
    reg["resolved_tech_final"] = reg["resolved_tech"]

    return reg.sort_values(["needs_manual_review", "winner_sp_count", "raw_row_count"], ascending=[True, False, False])


def build_review_queue(register: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "raw_id",
        "id_type",
        "resolution_rule",
        "resolved_bmu_final",
        "resolved_generator_id_final",
        "resolved_tech_final",
        "raw_row_count",
        "winner_sp_count",
        "offer_candidate_sp_count",
        "bid_candidate_sp_count",
    ]
    return register[register["needs_manual_review"]][cols].copy()


def apply_resolution(candidates: pd.DataFrame, action: pd.DataFrame, register: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = register[
        [
            "raw_id",
            "resolved_bmu_final",
            "resolved_generator_id_final",
            "resolved_generator_label_final",
            "resolved_tech_final",
            "resolution_rule",
            "needs_manual_review",
        ]
    ].copy()

    cand = candidates.copy()
    act = action.copy()

    for side in ["offer", "bid"]:
        renamed = lookup.rename(
            columns={
                "raw_id": f"marginal_{side}_id",
                "resolved_bmu_final": f"marginal_{side}_bmu_resolved",
                "resolved_generator_id_final": f"marginal_{side}_generator_id_resolved",
                "resolved_generator_label_final": f"marginal_{side}_generator_label_resolved",
                "resolved_tech_final": f"marginal_{side}_tech_resolved",
                "resolution_rule": f"marginal_{side}_resolution_rule",
                "needs_manual_review": f"marginal_{side}_needs_manual_review",
            }
        )
        cand = cand.merge(renamed, on=f"marginal_{side}_id", how="left")

    winner_renamed = lookup.rename(
        columns={
            "raw_id": "marginal_id_winner",
            "resolved_bmu_final": "marginal_bmu_winner_resolved",
            "resolved_generator_id_final": "marginal_generator_id_winner_resolved",
            "resolved_generator_label_final": "marginal_generator_label_winner_resolved",
            "resolved_tech_final": "marginal_tech_winner_resolved",
            "resolution_rule": "marginal_winner_resolution_rule",
            "needs_manual_review": "marginal_winner_needs_manual_review",
        }
    )
    act = act.merge(winner_renamed, on="marginal_id_winner", how="left")
    return cand, act


def build_identity_substitute_candidates(ranked: pd.DataFrame, register: pd.DataFrame) -> pd.DataFrame:
    lookup = register[
        [
            "raw_id",
            "id_type",
            "resolved_bmu_final",
            "resolved_generator_id_final",
            "resolved_generator_label_final",
            "resolved_tech_final",
            "resolution_rule",
            "needs_manual_review",
        ]
    ].copy()

    stack = ranked.copy()
    stack["id"] = stack["id"].astype(str)
    stack = stack.merge(lookup, left_on="id", right_on="raw_id", how="left")

    valid = stack[
        stack["stack_price_valid"]
        & stack["stack_price"].notna()
        & stack["volume"].notna()
        & ~stack["needs_manual_review"].fillna(True)
        & stack["resolved_bmu_final"].notna()
    ].copy()

    valid = valid.sort_values(
        by=["settlementDate", "settlementPeriod", "direction", "stack_price", "abs_volume", "sequenceNumber"],
        ascending=[True, True, True, False, False, False],
        kind="stable",
    )
    valid["identity_substitute_choice_rank"] = valid.groupby(["settlementDate", "settlementPeriod", "direction"]).cumcount() + 1
    valid["identity_substitute_price_rank_offset"] = valid["identity_substitute_choice_rank"] - 1
    return valid[
        [
            "settlementDate",
            "settlementPeriod",
            "direction",
            "identity_substitute_choice_rank",
            "identity_substitute_price_rank_offset",
            "id",
            "stack_price",
            "volume",
            "abs_volume",
            "sequenceNumber",
            "rank_in_stack_seq",
            "rank_in_stack_price",
            "price_gap_prev_seq",
            "price_gap_next_seq",
            "resolved_bmu_final",
            "resolved_generator_id_final",
            "resolved_generator_label_final",
            "resolved_tech_final",
            "resolution_rule",
        ]
    ].copy()


def apply_identity_fallback(
    action_resolved: pd.DataFrame, ranked: pd.DataFrame, register: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    substitutes = build_identity_substitute_candidates(ranked, register)
    top_substitutes = substitutes[substitutes["identity_substitute_choice_rank"] == 1].copy()
    top_substitutes = top_substitutes.rename(
        columns={
            "direction": "marginal_side_winner",
            "id": "marginal_identity_substitute_raw_id",
            "identity_substitute_price_rank_offset": "marginal_identity_substitute_price_rank_offset",
            "stack_price": "marginal_identity_substitute_price",
            "volume": "marginal_identity_substitute_volume",
            "abs_volume": "marginal_identity_substitute_abs_volume",
            "sequenceNumber": "marginal_identity_substitute_sequenceNumber",
            "rank_in_stack_seq": "marginal_identity_substitute_rank_in_stack_seq",
            "rank_in_stack_price": "marginal_identity_substitute_rank_in_stack_price",
            "price_gap_prev_seq": "marginal_identity_substitute_price_gap_prev",
            "price_gap_next_seq": "marginal_identity_substitute_price_gap_next",
            "resolved_bmu_final": "marginal_identity_substitute_bmu",
            "resolved_generator_id_final": "marginal_identity_substitute_generator_id",
            "resolved_generator_label_final": "marginal_identity_substitute_generator_label",
            "resolved_tech_final": "marginal_identity_substitute_tech",
            "resolution_rule": "marginal_identity_substitute_resolution_rule",
        }
    )

    out = action_resolved.copy()
    out["marginal_id_winner"] = out["marginal_id_winner"].astype(str)
    out = out.merge(top_substitutes, on=["settlementDate", "settlementPeriod", "marginal_side_winner"], how="left")

    numeric_unresolved = (
        out["marginal_winner_needs_manual_review"].fillna(False)
        & out["marginal_winner_resolution_rule"].eq("numeric_raw_id_manual_review")
        & out["marginal_side_winner"].isin(["offer", "bid"])
    )
    has_substitute = out["marginal_identity_substitute_bmu"].notna()
    use_substitute = numeric_unresolved & has_substitute

    out["marginal_identity_raw_winner_id"] = out["marginal_id_winner"]
    out["marginal_identity_substituted"] = use_substitute

    out["marginal_id_final"] = out["marginal_id_winner"]
    out.loc[use_substitute, "marginal_id_final"] = out.loc[use_substitute, "marginal_identity_substitute_raw_id"]

    out["marginal_bmu_final"] = out["marginal_bmu_winner_resolved"]
    out.loc[use_substitute, "marginal_bmu_final"] = out.loc[use_substitute, "marginal_identity_substitute_bmu"]

    out["marginal_generator_id_final"] = out["marginal_generator_id_winner_resolved"]
    out.loc[use_substitute, "marginal_generator_id_final"] = out.loc[use_substitute, "marginal_identity_substitute_generator_id"]

    out["marginal_generator_label_final"] = out["marginal_generator_label_winner_resolved"]
    out.loc[use_substitute, "marginal_generator_label_final"] = out.loc[use_substitute, "marginal_identity_substitute_generator_label"]

    out["marginal_tech_final"] = out["marginal_tech_winner_resolved"]
    out.loc[use_substitute, "marginal_tech_final"] = out.loc[use_substitute, "marginal_identity_substitute_tech"]

    out["marginal_identity_resolution_rule_final"] = out["marginal_winner_resolution_rule"]
    out.loc[use_substitute, "marginal_identity_resolution_rule_final"] = "numeric_winner_replaced_with_next_named_same_side"
    out.loc[numeric_unresolved & ~has_substitute, "marginal_identity_resolution_rule_final"] = "numeric_winner_unresolved_kept"

    out["marginal_identity_price_basis"] = "raw_winner_price"
    out["marginal_identity_substitute_price_delta"] = out["marginal_price_winner"] - out["marginal_identity_substitute_price"]

    audit = out[
        [
            "settlementDate",
            "settlementPeriod",
            "marginal_side_winner",
            "marginal_identity_raw_winner_id",
            "marginal_price_winner",
            "marginal_volume_winner",
            "marginal_winner_resolution_rule",
            "marginal_winner_needs_manual_review",
            "marginal_identity_substituted",
            "marginal_id_final",
            "marginal_bmu_final",
            "marginal_generator_id_final",
            "marginal_generator_label_final",
            "marginal_tech_final",
            "marginal_identity_resolution_rule_final",
            "marginal_identity_substitute_raw_id",
            "marginal_identity_substitute_price",
            "marginal_identity_substitute_volume",
            "marginal_identity_substitute_sequenceNumber",
            "marginal_identity_substitute_rank_in_stack_price",
            "marginal_identity_substitute_price_rank_offset",
            "marginal_identity_substitute_price_delta",
        ]
    ].copy()
    audit = audit[
        audit["marginal_winner_resolution_rule"].eq("numeric_raw_id_manual_review")
        | audit["marginal_identity_substituted"]
    ].sort_values(["marginal_identity_substituted", "settlementDate", "settlementPeriod"], ascending=[False, True, True])
    return out, audit


def apply_unknown_numeric_label(action_final: pd.DataFrame) -> pd.DataFrame:
    out = action_final.copy()
    unresolved_mask = out["marginal_identity_resolution_rule_final"].eq("numeric_winner_unresolved_kept")
    out["marginal_identity_unknown_numeric"] = unresolved_mask
    out.loc[unresolved_mask, "marginal_id_final"] = "UNKNOWN_NUMERIC"
    out.loc[unresolved_mask, "marginal_bmu_final"] = "UNKNOWN_NUMERIC"
    out.loc[unresolved_mask, "marginal_generator_id_final"] = "GEN_UNKNOWN_NUMERIC"
    out.loc[unresolved_mask, "marginal_generator_label_final"] = "UNKNOWN_NUMERIC"
    out.loc[unresolved_mask, "marginal_tech_final"] = "UNKNOWN_NUMERIC"
    out.loc[unresolved_mask, "marginal_identity_resolution_rule_final"] = "numeric_winner_labeled_unknown_numeric"
    return out


def build_audit(register: pd.DataFrame) -> pd.DataFrame:
    return (
        register.groupby(["id_type", "resolution_rule", "needs_manual_review"])
        .agg(n_ids=("raw_id", "size"), raw_rows=("raw_row_count", "sum"), winner_rows=("winner_sp_count", "sum"))
        .reset_index()
        .sort_values(["winner_rows", "raw_rows"], ascending=False)
    )


def build_summary(register: pd.DataFrame, review: pd.DataFrame, audit: pd.DataFrame, action_unknown: pd.DataFrame) -> dict:
    return {
        "n_unique_raw_ids": int(len(register)),
        "n_manual_review_ids": int(len(review)),
        "winner_rows_total": int(register["winner_sp_count"].sum()),
        "winner_rows_manual_review": int(register.loc[register["needs_manual_review"], "winner_sp_count"].sum()),
        "winner_rows_manual_review_share": float(register.loc[register["needs_manual_review"], "winner_sp_count"].sum() / register["winner_sp_count"].sum()) if register["winner_sp_count"].sum() else 0.0,
        "id_type_counts": register["id_type"].value_counts(dropna=False).to_dict(),
        "resolution_rule_counts": register["resolution_rule"].value_counts(dropna=False).to_dict(),
        "winner_side_counts": action_unknown["marginal_side_winner"].value_counts(dropna=False).to_dict(),
        "winner_rows_identity_substituted": int(action_unknown["marginal_identity_substituted"].fillna(False).sum()),
        "winner_rows_identity_substituted_share": float(action_unknown["marginal_identity_substituted"].fillna(False).mean()),
        "numeric_winner_rows_unknown_numeric": int(action_unknown["marginal_identity_unknown_numeric"].sum()),
        "numeric_winner_rows_unknown_numeric_share": float(action_unknown["marginal_identity_unknown_numeric"].mean()),
        "top_unresolved_winner_ids": (
            register.loc[register["needs_manual_review"]]
            .sort_values(["winner_sp_count", "raw_row_count"], ascending=False)[["raw_id", "id_type", "winner_sp_count", "raw_row_count"]]
            .head(25)
            .to_dict(orient="records")
        ),
    }


def main() -> None:
    raw, ranked, candidates, action = load_inputs()
    usage = build_usage_tables(raw, candidates, action)
    bmu_ref, gen_reg, marginal_reg = load_reference_maps()
    register = build_resolution_register(usage, bmu_ref, gen_reg, marginal_reg)
    review = build_review_queue(register)
    audit = build_audit(register)
    candidates_resolved, action_resolved = apply_resolution(candidates, action, register)
    action_final, final_audit = apply_identity_fallback(action_resolved, ranked, register)
    action_unknown = apply_unknown_numeric_label(action_final)
    summary = build_summary(register, review, audit, action_unknown)

    register.to_csv(OUT_REGISTER, index=False)
    register.to_parquet(OUT_REGISTER_PARQUET, index=False)
    review.to_csv(OUT_REVIEW, index=False)
    audit.to_csv(OUT_AUDIT, index=False)
    final_audit.to_csv(OUT_FINAL_AUDIT, index=False)

    candidates_resolved.to_parquet(OUT_CANDIDATES_APPLIED, index=False)
    candidates_resolved.to_csv(OUT_CANDIDATES_APPLIED.with_suffix(".csv"), index=False)
    action_resolved.to_parquet(OUT_ACTION_APPLIED, index=False)
    action_resolved.to_csv(OUT_ACTION_APPLIED.with_suffix(".csv"), index=False)
    action_final.to_parquet(OUT_ACTION_FINAL, index=False)
    action_final.to_csv(OUT_ACTION_FINAL.with_suffix(".csv"), index=False)
    action_unknown.to_parquet(OUT_ACTION_UNKNOWN, index=False)
    action_unknown.to_csv(OUT_ACTION_UNKNOWN.with_suffix(".csv"), index=False)

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("2023-2025 ISPSTACK ID Resolution Built")
    print("=" * 72)
    print(f"Unique raw ids            : {len(register):,}")
    print(f"Manual-review ids         : {len(review):,}")
    print(f"Winner rows manual review : {summary['winner_rows_manual_review']} / {summary['winner_rows_total']}")
    print(f"Identity substitutions    : {summary['winner_rows_identity_substituted']:,}")
    print(f"Unknown numeric rows      : {summary['numeric_winner_rows_unknown_numeric']:,}")
    print(f"Final panel               : {OUT_ACTION_UNKNOWN}")
    print(f"Summary                   : {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
