#!/usr/bin/env python3
"""
Demo batch: annotate 5 experimental structures + 2 AlphaFold models.
====================================================================

Reproducible evidence that `analyze` runs on both experimental crystals and
predicted (AlphaFold) models, tags provenance, handles the full 85-residue
pocket, and produces schema-valid output.

  * Experimental  : 5 KLIFS PDB codes across 5 kinases (looked up in KLIFS).
  * Predicted     : 2 AlphaFold DB models (ABL1 P00519, EGFR P00533); the
                    85-residue pocket is extracted by aligning to a cached KLIFS
                    reference for that kinase, and provenance is tagged
                    ``predicted``.

Every result is validated against hyaline/kinase/analysis_schema.json.

Outputs (research/kinase/demo/):
  * experimental/<pdb>.json          -- one file per experimental structure
  * predicted/<kinase>_alphafold.json-- one file per AlphaFold model
  * demo_summary.json / demo_summary.md -- combined table

Usage:
    python scripts/demo_analyze_batch.py

Network: KLIFS (experimental, cached after first run) and the AlphaFold DB
(predicted models, cached under klifs_cache/alphafold/).
"""
import importlib.util
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "kinase" / "demo"
AF_CACHE = ROOT / "klifs_cache" / "alphafold"

# 5 experimental structures across 5 distinct kinases.
EXPERIMENTAL = [
    ("2hyy", "ABL1"),   # imatinib-bound ABL1  -> DFG-out / Type II
    ("1m17", "EGFR"),   # erlotinib-bound EGFR -> DFG-in  / Type I
    ("1uwh", "BRAF"),   # sorafenib-bound BRAF -> DFG-out / Type II
    ("1t46", "KIT"),    # imatinib-bound KIT   -> DFG-out / Type II
    ("2src", "SRC"),    # SRC
]

# 2 predicted (AlphaFold) models; both kinases have a cached KLIFS reference.
PREDICTED = [
    ("ABL1", "P00519"),
    ("EGFR", "P00533"),
]


def _load_analyze():
    p = ROOT / "hyaline" / "kinase" / "analyze.py"
    spec = importlib.util.spec_from_file_location("hyaline_kinase_analyze", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def fetch_alphafold(uniprot: str) -> Path:
    """Download the current AlphaFold model PDB for a UniProt id (cached)."""
    AF_CACHE.mkdir(parents=True, exist_ok=True)
    dest = AF_CACHE / f"AF-{uniprot}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    meta = requests.get(
        f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}", timeout=30).json()
    url = meta[0]["pdbUrl"]
    pdb = requests.get(url, timeout=60).text
    dest.write_text(pdb, encoding="utf-8")
    return dest


def main():
    az = _load_analyze()
    (OUT / "experimental").mkdir(parents=True, exist_ok=True)
    (OUT / "predicted").mkdir(parents=True, exist_ok=True)

    summary = []
    n_bad = 0

    print("Experimental (KLIFS):")
    for pdb, kinase in EXPERIMENTAL:
        res = az.analyze(pdb, provenance="experimental").to_dict()
        errs = az.validate_result(res)
        (OUT / "experimental" / f"{pdb}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        status = "OK" if not errs else f"SCHEMA-INVALID ({len(errs)})"
        n_bad += bool(errs)
        summary.append({"input": pdb, "kinase": kinase, **res, "schema_valid": not errs})
        print(f"  {pdb} ({kinase}): {res['dfg_call']} / {res['inhibitor_class']}  [{status}]")

    print("Predicted (AlphaFold):")
    for kinase, uniprot in PREDICTED:
        model = fetch_alphafold(uniprot)
        res = az.analyze(kinase, provenance="predicted",
                         local_pdb=str(model), kinase=kinase).to_dict()
        errs = az.validate_result(res)
        (OUT / "predicted" / f"{kinase}_alphafold.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        status = "OK" if not errs else f"SCHEMA-INVALID ({len(errs)})"
        n_bad += bool(errs)
        summary.append({"input": f"AF-{uniprot}", "kinase": kinase, **res,
                        "schema_valid": not errs})
        print(f"  AF-{uniprot} ({kinase}): {res['dfg_call']} / {res['inhibitor_class']}  "
              f"[{res['n_pocket_residues_resolved']}/85 pocket, {status}]")

    (OUT / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary)

    print(f"\nWrote {OUT.relative_to(ROOT)}/ "
          f"({len(EXPERIMENTAL)} experimental + {len(PREDICTED)} predicted)")
    if n_bad:
        print(f"FAILED: {n_bad} result(s) did not validate against the schema")
        sys.exit(1)
    print("All results validate against analysis_schema.json.")


def _write_markdown(summary):
    lines = [
        "# Analyze demo - 5 experimental + 2 AlphaFold",
        "",
        "Reproduce: `python scripts/demo_analyze_batch.py`. Every row is validated",
        "against `hyaline/kinase/analysis_schema.json`.",
        "",
        "| Input | Kinase | Provenance | DFG | Conf. | Inhibitor class | Pocket | Schema |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['input']} | {r['kinase']} | {r['provenance']} | {r['dfg_call']} | "
            f"{r['dfg_confidence']} | {r['inhibitor_class']} | "
            f"{r['n_pocket_residues_resolved']}/85 | {'valid' if r['schema_valid'] else 'INVALID'} |")
    (OUT / "demo_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
