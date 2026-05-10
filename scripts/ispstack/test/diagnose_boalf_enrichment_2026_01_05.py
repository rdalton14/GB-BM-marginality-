from __future__ import annotations

import json
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())
OUT_DIR = PROJECT_ROOT / "data" / "raw_test" / "ispstack" / "boalf_enrichment_2026_01_05"
RAW_ACCEPTANCES_DIR = OUT_DIR / "raw_acceptances_all"
RAW_BOALF_DIR = OUT_DIR / "raw_boalf"

DATE_STR = "2026-01-05"
SETTLEMENT_PERIODS = list(range(1, 49))
SLEEP_SECONDS = 0.10
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

ACCEPTANCES_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/acceptances/all"
BOALF_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/acceptances/all"

ACCEPTANCES_CSV = OUT_DIR / "acceptances_all_2026_01_05.csv"
BOALF_CSV = OUT_DIR / "boalf_2026_01_05.csv"
ENRICHED_CSV = OUT_DIR / "acceptances_all_with_boalf_flags_2026_01_05.csv"
MATCH_DIAGNOSTICS_CSV = OUT_DIR / "boalf_match_diagnostics_2026_01_05.csv"
MARGINAL_COMPARISON_CSV = OUT_DIR / "marginal_so_filter_comparison_2026_01_05.csv"
SUMMARY_JSON = OUT_DIR / "boalf_enrichment_summary_2026_01_05.json"

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def fetch_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=TIMEOUT_SECONDS)
            if response.status_code == 429:
                time.sleep(attempt)
                last_error = f"429 rate limited on attempt {attempt}"
                continue
            response.raise_for_status()
            payload = response.json()
            return {
                "status": "ok",
                "status_code": response.status_code,
                "url": response.url,
                "payload": payload,
                "rows": extract_rows(payload),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(attempt)
    return {
        "status": "error",
        "status_code": None,
        "url": url,
        "payload": None,
        "rows": [],
        "error": last_error,
    }


def save_raw(result: dict[str, Any], out_dir: Path, settlement_period: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sp_{settlement_period:02d}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "status": result["status"],
                "status_code": result["status_code"],
                "url": result["url"],
                "error": result["error"],
                "payload": result["payload"],
            },
            f,
            indent=2,
        )


def standardize_acceptances(period_rows: list[dict[str, Any]], settlement_period: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in period_rows:
        rec = dict(row)
        rec["settlementDate"] = rec.get("settlementDate", DATE_STR)
        rec["settlementPeriod"] = int(rec.get("settlementPeriod", settlement_period))
        rec["acceptanceNumber"] = pd.to_numeric(rec.get("acceptanceNumber"), errors="coerce")
        rec["acceptanceTime"] = rec.get("acceptanceTime")
        rec["bmUnit"] = rec.get("bmUnit")
        rec["nationalGridBmUnit"] = rec.get("nationalGridBmUnit")
        rec["source_endpoint"] = "acceptances_all"
        out.append(rec)
    return out


def standardize_boalf(period_rows: list[dict[str, Any]], settlement_period: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in period_rows:
        rec = dict(row)
        rec["requestSettlementPeriod"] = settlement_period
        rec["settlementDate"] = rec.get("settlementDate", DATE_STR)
        rec["settlementPeriodFrom"] = pd.to_numeric(rec.get("settlementPeriodFrom"), errors="coerce")
        rec["settlementPeriodTo"] = pd.to_numeric(rec.get("settlementPeriodTo"), errors="coerce")
        rec["acceptanceNumber"] = pd.to_numeric(rec.get("acceptanceNumber"), errors="coerce")
        rec["acceptanceTime"] = rec.get("acceptanceTime")
        rec["bmUnit"] = rec.get("bmUnit")
        rec["nationalGridBmUnit"] = rec.get("nationalGridBmUnit")
        rec["soFlag"] = rec.get("soFlag")
        rec["deemedBoFlag"] = rec.get("deemedBoFlag")
        rec["storFlag"] = rec.get("storFlag")
        rec["rrFlag"] = rec.get("rrFlag")
        rec["source_endpoint"] = "boalf_acceptances_all"
        out.append(rec)
    return out


def pull_acceptances_all() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sp in SETTLEMENT_PERIODS:
        print(f"Fetching acceptances/all {DATE_STR} SP {sp:02d}")
        result = fetch_json(f"{ACCEPTANCES_URL}/{DATE_STR}/{sp}", {"format": "json"})
        save_raw(result, RAW_ACCEPTANCES_DIR, sp)
        rows.extend(standardize_acceptances(result["rows"], sp))
        time.sleep(SLEEP_SECONDS)
    return pd.DataFrame(rows)


def pull_boalf() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sp in SETTLEMENT_PERIODS:
        print(f"Fetching BOALF-style acceptances {DATE_STR} SP {sp:02d}")
        result = fetch_json(
            BOALF_URL,
            {"settlementDate": DATE_STR, "settlementPeriod": sp, "format": "json"},
        )
        save_raw(result, RAW_BOALF_DIR, sp)
        rows.extend(standardize_boalf(result["rows"], sp))
        time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    subset = [
        "settlementDate",
        "settlementPeriodFrom",
        "settlementPeriodTo",
        "acceptanceNumber",
        "acceptanceTime",
        "bmUnit",
        "nationalGridBmUnit",
        "timeFrom",
        "timeTo",
        "levelFrom",
        "levelTo",
        "soFlag",
        "deemedBoFlag",
        "storFlag",
        "rrFlag",
    ]
    subset = [c for c in subset if c in df.columns]
    return df.drop_duplicates(subset=subset).copy()


def score_candidates(acc_row: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.assign(match_confidence=pd.Series(dtype="object"), sp_compatible=pd.Series(dtype="bool"))

    out = candidates.copy()
    sp = int(acc_row["settlementPeriod"])

    out["sp_compatible"] = (
        pd.to_numeric(out["settlementPeriodFrom"], errors="coerce").le(sp)
        & pd.to_numeric(out["settlementPeriodTo"], errors="coerce").ge(sp)
    )
    out["exact_key"] = (
        out["acceptanceNumber"].eq(acc_row["acceptanceNumber"])
        & out["bmUnit"].astype(str).eq(str(acc_row.get("bmUnit")))
    )
    out["strong_key"] = (
        out["acceptanceNumber"].eq(acc_row["acceptanceNumber"])
        & out["nationalGridBmUnit"].astype(str).eq(str(acc_row.get("nationalGridBmUnit")))
    )
    out["time_key"] = (
        out["acceptanceNumber"].eq(acc_row["acceptanceNumber"])
        & out["acceptanceTime"].astype(str).eq(str(acc_row.get("acceptanceTime")))
    )

    acc_number_count = int(out["acceptanceNumber"].eq(acc_row["acceptanceNumber"]).sum())
    out["fallback_key"] = out["acceptanceNumber"].eq(acc_row["acceptanceNumber"]) & (acc_number_count == 1)

    conditions = [
        out["exact_key"] & out["sp_compatible"],
        out["strong_key"] & out["sp_compatible"],
        out["time_key"] & out["sp_compatible"],
        out["fallback_key"] & out["sp_compatible"],
        (out["exact_key"] | out["strong_key"] | out["time_key"] | out["fallback_key"]) & ~out["sp_compatible"],
    ]
    labels = [
        "exact_valid_sp",
        "strong_valid_sp",
        "time_valid_sp",
        "fallback_valid_sp",
        "matched_but_sp_outside_range",
    ]
    out["match_confidence"] = np.select(conditions, labels, default="candidate_unscored")

    priority = {
        "exact_valid_sp": 1,
        "strong_valid_sp": 2,
        "time_valid_sp": 3,
        "fallback_valid_sp": 4,
        "matched_but_sp_outside_range": 5,
        "candidate_unscored": 99,
    }
    out["priority"] = out["match_confidence"].map(priority).fillna(999)
    return out.sort_values(
        ["priority", "settlementPeriodFrom", "settlementPeriodTo", "acceptanceTime", "bmUnit"],
        kind="stable",
    )


def enrich_with_boalf_flags(acc: pd.DataFrame, boalf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []

    if acc.empty:
        return acc.copy(), pd.DataFrame()

    for _, row in acc.iterrows():
        same_acceptance = boalf[boalf["acceptanceNumber"].eq(row["acceptanceNumber"])].copy()
        scored = score_candidates(row, same_acceptance)

        valid_labels = ["exact_valid_sp", "strong_valid_sp", "time_valid_sp", "fallback_valid_sp"]
        valid = scored[scored["match_confidence"].isin(valid_labels)].copy()
        best_label = valid["match_confidence"].iloc[0] if not valid.empty else None
        top_valid = valid[valid["match_confidence"].eq(best_label)].copy() if best_label else valid

        if not valid.empty and len(top_valid) == 1:
            chosen = top_valid.iloc[0]
            match_confidence = str(chosen["match_confidence"])
        elif not valid.empty and len(top_valid) > 1:
            chosen = None
            match_confidence = "duplicate_match"
        else:
            outside = scored[scored["match_confidence"].eq("matched_but_sp_outside_range")]
            if not outside.empty:
                chosen = None
                match_confidence = "matched_but_sp_outside_range"
            else:
                chosen = None
                match_confidence = "unmatched"

        base = row.to_dict()
        base["match_confidence"] = match_confidence
        base["boalf_match_count"] = int(len(valid))
        base["boalf_candidate_count"] = int(len(scored))

        if chosen is not None:
            for col in ["soFlag", "deemedBoFlag", "storFlag", "rrFlag", "settlementPeriodFrom", "settlementPeriodTo", "timeFrom", "timeTo", "levelFrom", "levelTo"]:
                if col in chosen.index:
                    base[f"boalf_{col}"] = chosen[col]
            base["boalf_bmUnit"] = chosen.get("bmUnit")
            base["boalf_nationalGridBmUnit"] = chosen.get("nationalGridBmUnit")
            base["boalf_acceptanceTime"] = chosen.get("acceptanceTime")
        else:
            for col in [
                "boalf_soFlag",
                "boalf_deemedBoFlag",
                "boalf_storFlag",
                "boalf_rrFlag",
                "boalf_settlementPeriodFrom",
                "boalf_settlementPeriodTo",
                "boalf_timeFrom",
                "boalf_timeTo",
                "boalf_levelFrom",
                "boalf_levelTo",
                "boalf_bmUnit",
                "boalf_nationalGridBmUnit",
                "boalf_acceptanceTime",
            ]:
                base[col] = pd.NA

        enriched_rows.append(base)

        diag = {
            "settlementDate": row["settlementDate"],
            "settlementPeriod": row["settlementPeriod"],
            "acceptanceNumber": row["acceptanceNumber"],
            "bmUnit": row.get("bmUnit"),
            "nationalGridBmUnit": row.get("nationalGridBmUnit"),
            "acceptanceTime": row.get("acceptanceTime"),
            "match_confidence": match_confidence,
            "n_same_acceptance_number_candidates": int(len(same_acceptance)),
            "n_valid_sp_matches": int(len(valid)),
        }
        if not scored.empty:
            diag["candidate_bmUnits"] = "; ".join(pd.Series(scored["bmUnit"].dropna().astype(str).unique()).tolist()[:10])
            diag["candidate_confidences"] = "; ".join(scored["match_confidence"].astype(str).unique().tolist()[:10])
        if len(top_valid) > 1:
            diag["duplicate_bmUnits"] = "; ".join(pd.Series(top_valid["bmUnit"].dropna().astype(str).unique()).tolist()[:10])
        diagnostics.append(diag)

    return pd.DataFrame(enriched_rows), pd.DataFrame(diagnostics)


def marginal_offer_before_after(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame()

    df = enriched.copy()
    df["offerPrice_num"] = pd.to_numeric(df["offerPrice"], errors="coerce")
    df["boalf_soFlag_bool"] = df["boalf_soFlag"].astype("boolean")

    records: list[dict[str, Any]] = []
    for (settlement_date, settlement_period), grp in df.groupby(["settlementDate", "settlementPeriod"], sort=True):
        grp = grp[grp["offerPrice_num"].notna()].copy()
        if grp.empty:
            continue

        before = grp.sort_values(["offerPrice_num", "acceptanceNumber"], ascending=[False, True], kind="stable").iloc[0]
        after_pool = grp[grp["boalf_soFlag_bool"].eq(False)].copy()
        after = None if after_pool.empty else after_pool.sort_values(
            ["offerPrice_num", "acceptanceNumber"], ascending=[False, True], kind="stable"
        ).iloc[0]

        records.append(
            {
                "settlementDate": settlement_date,
                "settlementPeriod": int(settlement_period),
                "highest_offer_identifier_before": before.get("bmUnit"),
                "highest_offer_price_before": float(before["offerPrice_num"]),
                "highest_offer_acceptanceNumber_before": int(before["acceptanceNumber"]) if pd.notna(before["acceptanceNumber"]) else None,
                "highest_offer_match_confidence_before": before.get("match_confidence"),
                "highest_offer_identifier_after_so_filter": None if after is None else after.get("bmUnit"),
                "highest_offer_price_after_so_filter": None if after is None else float(after["offerPrice_num"]),
                "highest_offer_acceptanceNumber_after_so_filter": None if after is None or pd.isna(after["acceptanceNumber"]) else int(after["acceptanceNumber"]),
                "n_acceptances_rows": int(len(grp)),
                "n_rows_with_valid_soflag_false": int(len(after_pool)),
                "so_filter_changes_marginal_plant": bool(
                    after is not None and str(before.get("bmUnit")) != str(after.get("bmUnit"))
                ),
                "so_filter_drops_all_candidates": bool(after is None),
            }
        )

    return pd.DataFrame(records)


def build_summary(acc: pd.DataFrame, boalf: pd.DataFrame, enriched: pd.DataFrame, diagnostics: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    matched_valid = enriched["match_confidence"].isin(
        ["exact_valid_sp", "strong_valid_sp", "time_valid_sp", "fallback_valid_sp"]
    )
    marginal_candidates = enriched.groupby(["settlementDate", "settlementPeriod"])["offerPrice"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").eq(pd.to_numeric(s, errors="coerce").max()).fillna(False)
    )
    matched_rows = enriched[matched_valid].copy()

    summary = {
        "date": DATE_STR,
        "acceptances_all_rows": int(len(acc)),
        "boalf_rows": int(len(boalf)),
        "valid_sp_match_share": float(matched_valid.mean()),
        "unmatched_share": float(enriched["match_confidence"].eq("unmatched").mean()),
        "duplicate_match_share": float(enriched["match_confidence"].eq("duplicate_match").mean()),
        "sp_outside_range_share": float(enriched["match_confidence"].eq("matched_but_sp_outside_range").mean()),
        "marginal_candidate_rows_with_valid_soflag_share": float(
            (marginal_candidates & matched_valid).sum() / max(int(marginal_candidates.sum()), 1)
        ),
        "matched_soflag_true_share": float(matched_rows["boalf_soFlag"].astype("boolean").eq(True).mean()) if not matched_rows.empty else None,
        "matched_soflag_false_share": float(matched_rows["boalf_soFlag"].astype("boolean").eq(False).mean()) if not matched_rows.empty else None,
        "top_unmatched_bmUnits": (
            enriched.loc[enriched["match_confidence"].eq("unmatched"), "bmUnit"]
            .fillna("MISSING_BMU")
            .astype(str)
            .value_counts()
            .head(20)
            .to_dict()
        ),
        "duplicate_match_examples": diagnostics.loc[
            diagnostics["match_confidence"].eq("duplicate_match"),
            ["settlementDate", "settlementPeriod", "acceptanceNumber", "bmUnit", "duplicate_bmUnits"],
        ].head(20).to_dict(orient="records"),
        "so_filter_changes_marginal_plant_count": int(comparison["so_filter_changes_marginal_plant"].sum()) if not comparison.empty else 0,
        "so_filter_changes_marginal_plant_share": float(comparison["so_filter_changes_marginal_plant"].mean()) if not comparison.empty else 0.0,
        "so_filter_drops_all_candidates_count": int(comparison["so_filter_drops_all_candidates"].sum()) if not comparison.empty else 0,
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("BOALF ENRICHMENT DIAGNOSTIC")
    print("=" * 72)
    print(f"acceptances/all rows                      : {summary['acceptances_all_rows']:,}")
    print(f"BOALF rows                                : {summary['boalf_rows']:,}")
    print(f"Matched with valid SP range               : {summary['valid_sp_match_share']:.2%}")
    print(f"Unmatched                                 : {summary['unmatched_share']:.2%}")
    print(f"Duplicate / ambiguous                     : {summary['duplicate_match_share']:.2%}")
    print(f"Matched but SP outside BOALF range        : {summary['sp_outside_range_share']:.2%}")
    print(f"Marginal candidate rows with valid soFlag : {summary['marginal_candidate_rows_with_valid_soflag_share']:.2%}")
    print(f"Matched rows soFlag == true               : {summary['matched_soflag_true_share']:.2%}" if summary["matched_soflag_true_share"] is not None else "Matched rows soFlag == true               : n/a")
    print(f"Matched rows soFlag == false              : {summary['matched_soflag_false_share']:.2%}" if summary["matched_soflag_false_share"] is not None else "Matched rows soFlag == false              : n/a")
    print(f"SPs where SO filtering changes marginal   : {summary['so_filter_changes_marginal_plant_count']:,} ({summary['so_filter_changes_marginal_plant_share']:.2%})")
    print(f"SPs where SO filtering drops all offers   : {summary['so_filter_drops_all_candidates_count']:,}")
    print("\nRecommendation:")
    reliable = summary["valid_sp_match_share"] >= 0.95 and summary["duplicate_match_share"] <= 0.01 and summary["sp_outside_range_share"] <= 0.01
    material = summary["so_filter_changes_marginal_plant_share"] >= 0.05
    print(f"- Is BOALF matching reliable enough to attach soFlag? {'Yes' if reliable else 'Not yet'}")
    print(f"- Does SO filtering materially change the marginal action? {'Yes' if material else 'Probably not materially on this one-day test'}")
    if reliable:
        print(f"- Should this be scaled? {'Scale to one week next' if not material else 'Scale to one week, then Q1 if stable'}")
    else:
        print("- Should this be scaled? No, improve the matching logic first")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    acc = pull_acceptances_all()
    boalf = pull_boalf()

    acc.to_csv(ACCEPTANCES_CSV, index=False)
    boalf.to_csv(BOALF_CSV, index=False)

    enriched, diagnostics = enrich_with_boalf_flags(acc, boalf)
    comparison = marginal_offer_before_after(enriched)

    enriched.to_csv(ENRICHED_CSV, index=False)
    diagnostics.to_csv(MATCH_DIAGNOSTICS_CSV, index=False)
    comparison.to_csv(MARGINAL_COMPARISON_CSV, index=False)

    summary = build_summary(acc, boalf, enriched, diagnostics, comparison)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print_summary(summary)


if __name__ == "__main__":
    main()
