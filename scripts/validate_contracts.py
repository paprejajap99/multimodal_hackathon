#!/usr/bin/env python3
"""Validate Phase 0 contracts. Usage: python scripts/validate_contracts.py  (needs pyyaml, jsonschema)."""
import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


tax = yaml.safe_load((ROOT / "config/flaw_types.yaml").read_text(encoding="utf-8"))
schema = json.loads((ROOT / "docs/contracts/analysis.schema.json").read_text(encoding="utf-8"))
fx = json.loads((ROOT / "fixtures/analysis_example.json").read_text(encoding="utf-8"))

Draft202012Validator.check_schema(schema)
for e in sorted(Draft202012Validator(schema).iter_errors(fx), key=lambda e: list(e.path)):
    errors.append(f"fixture schema error at {list(e.path)}: {e.message}")

# --- taxonomy internal consistency
flaws, feats = tax["flaws"], tax["features"]
comps = {c["id"] for c in tax["scoring"]["components"]}
check(abs(sum(c["weight"] for c in tax["scoring"]["components"]) - 1.0) < 1e-9, "rubric weights must sum to 1.0")
check(len(tax["levels"]) == 5 and len(tax["scoring"]["default_level_penalty"]) == 5, "need 5 levels / 5 penalties")
for fid, f in flaws.items():
    check(f["id"] == fid, f"{fid}: id mismatch")
    check(len(f["injection"]["values"]) == 5, f"{fid}: injection.values needs 5 entries")
    check(len(f["detection"]["severity_bounds"]) == 5, f"{fid}: severity_bounds needs 5 entries")
    check(f["detection"]["severity_bounds"] == sorted(f["detection"]["severity_bounds"]), f"{fid}: bounds must ascend")
    check(f["primary_feature"] in feats and all(x in feats for x in f["features"]), f"{fid}: unknown feature")
    check(f["scoring"]["component"] in comps, f"{fid}: unknown rubric component")
    check(f["region_type"] in ("words", "gap"), f"{fid}: bad region_type")

# --- taxonomy <-> schema enums
defs = schema["$defs"]
check(set(defs["flawType"]["enum"]) == set(flaws), "schema flawType enum != taxonomy flaw IDs")
check(set(defs["featureId"]["enum"]) == set(feats), "schema featureId enum != taxonomy feature IDs")
check(set(defs["componentId"]["enum"]) == comps, "schema componentId enum != taxonomy components")
check(set(defs["severityLabel"]["enum"]) == {l["id"] for l in tax["levels"]}, "schema severityLabel enum != taxonomy levels")

# --- fixture semantic checks
ref_w, par_w = fx["words"]["reference"], fx["words"]["participant"]
check(len(ref_w) == len(par_w), "reference/participant word counts differ")
ids = [r["id"] for r in fx["flaw_regions"]]
check(len(set(ids)) == len(ids), "duplicate region ids")
check([r["start_s"] for r in fx["flaw_regions"]] == sorted(r["start_s"] for r in fx["flaw_regions"]), "regions not sorted")
last_end = -1.0
for r in fx["flaw_regions"]:
    f = flaws[r["flaw_type"]]
    a, b = r["transcript"]["word_start_index"], r["transcript"]["word_end_index"]
    check(0 <= a <= b < len(par_w), f"{r['id']}: bad word range")
    check(r["end_s"] >= r["start_s"] and abs(r["duration_s"] - (r["end_s"] - r["start_s"])) < 1e-2, f"{r['id']}: time mismatch")
    check(r["start_s"] >= last_end - 1e-9, f"{r['id']}: overlaps previous region")
    last_end = r["end_s"]
    if f["region_type"] == "gap":
        check(abs(r["start_s"] - par_w[a]["end_s"]) < 1e-2 and abs(r["end_s"] - par_w[b]["start_s"]) < 1e-2, f"{r['id']}: gap bounds != word boundaries")
    else:
        check(abs(r["start_s"] - par_w[a]["start_s"]) < 1e-2 and abs(r["end_s"] - par_w[b]["end_s"]) < 1e-2, f"{r['id']}: bounds != word bounds")
    bounds = f["detection"]["severity_bounds"]
    exp = max([i for i, x in enumerate(bounds, 1) if r["detection"]["metric_value"] >= x] or [0])
    check(exp == r["severity_level"], f"{r['id']}: severity {r['severity_level']} != taxonomy-derived {exp}")
    check(r["detection"]["metric"] == f["detection"]["metric"], f"{r['id']}: metric mismatch")
    check(r["evidence"][0]["feature"] == f["primary_feature"], f"{r['id']}: first evidence must be primary feature")
    check(r["severity_label"] == tax["levels"][r["severity_level"] - 1]["id"], f"{r['id']}: severity_label mismatch")
rub_ids = {rid for c in fx["rubric_scores"] for rid in c["flaw_region_ids"]}
check(rub_ids == set(ids), "rubric flaw_region_ids must cover all regions exactly")
check(abs(sum(c["weight"] for c in fx["rubric_scores"]) - 1.0) < 1e-9, "fixture rubric weights != 1")
check({s["feature_id"] for s in fx["series"]} <= set(feats), "unknown series feature")

# --- docs mention every flaw id (catches renamed/missing IDs)
for doc in ("docs/SPEC.md", "docs/ARCHITECTURE.md"):
    p = ROOT / doc
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        for fid in flaws:
            check(fid in txt, f"{doc} does not mention flaw id {fid}")

if errors:
    print("CONTRACT VALIDATION FAILED:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print(f"OK: taxonomy ({len(flaws)} flaws, {len(feats)} features), schema, fixture ({len(fx['flaw_regions'])} regions) are consistent.")
