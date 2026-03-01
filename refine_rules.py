# src/refine_rules.py
# -*- coding: utf-8 -*-
"""
Rule-based refinement combining:
- mean probability p (slope-unit)
- dominant LISA type (HH/HL/LH/LL/N)
- CSI (0-1)
- InSAR trend class (accelerating/seasonal/stable/mixed)

This script is designed to be fully reproducible WITHOUT any geometry.

Inputs (CSV, de-identified):
1) slope-unit base table (from your 3.3 aggregation), containing at least:
   - unit_id
   - p_mean
   - dom_lisa  (HH/HL/LH/LL/N)
   - CSI

2) InSAR trend classification table:
   - unit_id
   - trend_class  (accelerating/seasonal/stable/mixed)

Outputs:
- refined_susceptibility.csv including:
   - preliminary class (Table 3)
   - refined class (after rules)
   - a "rule_trace" field indicating what rule changed the class (reviewer-friendly)

Notes:
- Your Methods mention that LISA alone does not quantify stability/reliability.
  CSI is used to enhance robustness.
- Since your paper’s exact refinement rules may be specific, this file provides a
  transparent and configurable rule set. You should keep the rules consistent with
  your manuscript text.

Default refinement logic (reasonable + easy to justify):
A) Start from preliminary class = Table 3 (probability level x dom_lisa).
B) CSI-based reliability adjustment:
   - If CSI is very low (<= csi_low), downgrade one level unless already Low risk.
   - If CSI is high (>= csi_high) and class is Uncertain, upgrade to Moderate risk.
C) InSAR-based dynamic adjustment:
   - If trend is accelerating, upgrade one level (up to Very high risk).
   - If trend is stable and CSI low, downgrade one level (boundary stabilization).
   - Seasonal does not automatically upgrade; it flags cyclic activity but not monotonic acceleration.

You can modify thresholds and upgrade/downgrade behavior in CFG.
"""

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


CFG = {
    "base_csv": r"F:\your_project\outputs_spatial_csi\slopeunit_csi.csv",  # contains p_mean, dom_lisa, CSI
    "base_unit_id": "unit_id",
    "base_p": "p_mean",
    "base_dom": "dom_lisa",
    "base_csi": "CSI",

    "insar_csv": r"F:\your_project\outputs_insar_trend\insar_trend_classification.csv",
    "insar_unit_id": "unit_id",
    "insar_class": "trend_class",

    # Table 3 probability thresholds
    "p_low": 0.3,
    "p_high": 0.7,

    # CSI thresholds
    "csi_low": 0.10,     # very weak: likely transitional/boundary
    "csi_high": 0.35,    # strong stable cluster

    # Output
    "out_dir": r"F:\your_project\outputs_refined_rules",
}


# Table 3 mapping (Probability level x Dominant LISA) -> preliminary class
TABLE3 = {
    ("H", "HH"): "Very high risk",
    ("H", "HL"): "Very high risk",
    ("H", "LH"): "High risk",
    ("H", "N"):  "High risk",
    ("H", "LL"): "Moderate risk",

    ("M", "HH"): "High risk",
    ("M", "HL"): "High risk",
    ("M", "LH"): "Uncertain",
    ("M", "N"):  "Uncertain",
    ("M", "LL"): "Moderate risk",

    ("L", "HH"): "High risk",
    ("L", "HL"): "Uncertain",
    ("L", "LH"): "Low risk",
    ("L", "N"):  "Low risk",
    ("L", "LL"): "Low risk",
}

# Ordered risk levels for upgrade/downgrade operations
RISK_ORDER = ["Low risk", "Moderate risk", "High risk", "Very high risk"]
# "Uncertain" is treated as a special state (can move to Moderate or remain)


def prob_level(p, p_low, p_high):
    if p < p_low:
        return "L"
    if p < p_high:
        return "M"
    return "H"


def preliminary_class(p_mean, dom_lisa, p_low, p_high):
    pl = prob_level(p_mean, p_low, p_high)
    return TABLE3.get((pl, dom_lisa), "Uncertain")


def upgrade(cls):
    """Upgrade one level within RISK_ORDER; Uncertain -> Moderate risk by default."""
    if cls == "Uncertain":
        return "Moderate risk"
    if cls not in RISK_ORDER:
        return cls
    i = RISK_ORDER.index(cls)
    return RISK_ORDER[min(i + 1, len(RISK_ORDER) - 1)]


def downgrade(cls):
    """Downgrade one level within RISK_ORDER; Uncertain -> Low risk by default."""
    if cls == "Uncertain":
        return "Low risk"
    if cls not in RISK_ORDER:
        return cls
    i = RISK_ORDER.index(cls)
    return RISK_ORDER[max(i - 1, 0)]


def apply_rules(row, cfg):
    """
    Apply refinement rules and return (refined_class, trace).
    """
    base_cls = row["prelim_class"]
    csi = row.get("CSI", np.nan)
    trend = row.get("trend_class", "mixed")

    refined = base_cls
    trace = []

    # --- Rule B: CSI reliability adjustment ---
    # B1) very low CSI -> downgrade (unless already Low risk)
    if np.isfinite(csi) and (csi <= cfg["csi_low"]):
        new_cls = downgrade(refined)
        if new_cls != refined:
            trace.append(f"CSI_low({csi:.3f}): {refined} -> {new_cls}")
            refined = new_cls

    # B2) high CSI and Uncertain -> promote to Moderate risk
    if np.isfinite(csi) and (csi >= cfg["csi_high"]) and (refined == "Uncertain"):
        new_cls = "Moderate risk"
        trace.append(f"CSI_high({csi:.3f}) resolves Uncertain -> Moderate risk")
        refined = new_cls

    # --- Rule C: InSAR dynamic adjustment ---
    # C1) accelerating -> upgrade one level (cap at Very high risk)
    if str(trend).lower() == "accelerating":
        new_cls = upgrade(refined)
        if new_cls != refined:
            trace.append(f"InSAR_accelerating: {refined} -> {new_cls}")
            refined = new_cls

    # C2) stable + low CSI -> (optional) downgrade one more level
    if str(trend).lower() == "stable" and np.isfinite(csi) and (csi <= cfg["csi_low"]):
        new_cls = downgrade(refined)
        if new_cls != refined:
            trace.append(f"InSAR_stable & CSI_low: {refined} -> {new_cls}")
            refined = new_cls

    # Seasonal/mixed: no forced change by default
    if not trace:
        trace = ["no_change"]

    return refined, " | ".join(trace)


def main():
    os.makedirs(CFG["out_dir"], exist_ok=True)

    # 1) Load base table
    base = pd.read_csv(CFG["base_csv"])
    need = [CFG["base_unit_id"], CFG["base_p"], CFG["base_dom"], CFG["base_csi"]]
    for c in need:
        if c not in base.columns:
            raise ValueError(f"Missing '{c}' in base_csv.")

    base = base.rename(columns={
        CFG["base_unit_id"]: "unit_id",
        CFG["base_p"]: "p_mean",
        CFG["base_dom"]: "dom_lisa",
        CFG["base_csi"]: "CSI",
    }).copy()

    base["unit_id"] = base["unit_id"].astype(int)
    base["p_mean"] = base["p_mean"].astype(float)
    base["dom_lisa"] = base["dom_lisa"].astype(str)
    base["CSI"] = pd.to_numeric(base["CSI"], errors="coerce")

    # 2) Load InSAR trend table (optional but recommended)
    insar = pd.read_csv(CFG["insar_csv"])
    if CFG["insar_unit_id"] not in insar.columns or CFG["insar_class"] not in insar.columns:
        raise ValueError("InSAR table must contain unit_id and trend_class columns.")
    insar = insar.rename(columns={
        CFG["insar_unit_id"]: "unit_id",
        CFG["insar_class"]: "trend_class",
    }).copy()
    insar["unit_id"] = insar["unit_id"].astype(int)
    insar["trend_class"] = insar["trend_class"].astype(str)

    # 3) Merge (left join: keep all units)
    df = base.merge(insar, on="unit_id", how="left")
    df["trend_class"] = df["trend_class"].fillna("mixed")

    # 4) Preliminary class (Table 3)
    df["prob_level"] = df["p_mean"].apply(lambda x: prob_level(x, CFG["p_low"], CFG["p_high"]))
    df["prelim_class"] = [
        preliminary_class(p, d, CFG["p_low"], CFG["p_high"])
        for p, d in zip(df["p_mean"].values, df["dom_lisa"].values)
    ]

    # 5) Apply refinement rules
    refined_list = []
    trace_list = []
    for _, row in df.iterrows():
        refined, trace = apply_rules(row, CFG)
        refined_list.append(refined)
        trace_list.append(trace)

    df["refined_class"] = refined_list
    df["rule_trace"] = trace_list

    # 6) Save
    out_path = os.path.join(CFG["out_dir"], "refined_susceptibility.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Quick summary for manuscript/reviewer
    summary = df["refined_class"].value_counts(dropna=False).to_dict()
    with open(os.path.join(CFG["out_dir"], "refined_class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Done ===")
    print("Saved:", out_path)
    print("Class counts:", summary)


if __name__ == "__main__":
    main()