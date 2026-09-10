#!/usr/bin/env python3
"""
Annotate a kinase structure with its conformational state + inhibitor class.

Examples
--------
    # by PDB code (experimental, looked up in KLIFS)
    python scripts/analyze_kinase.py 2hyy

    # by KLIFS structure_ID
    python scripts/analyze_kinase.py --klifs-id 1081

    # a predicted (AlphaFold) model whose 85-residue pocket has been extracted
    python scripts/analyze_kinase.py --local pocket.mol2 --provenance predicted
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Load the analyze module directly by path so we don't trigger the heavy
# hyaline package __init__ (GPCR deps like h5py). Kinase stays self-contained.
_mod_path = Path(__file__).resolve().parent.parent / "hyaline" / "kinase" / "analyze.py"
_spec = importlib.util.spec_from_file_location("hyaline_kinase_analyze", _mod_path)
_m = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _m  # required for dataclass introspection
_spec.loader.exec_module(_m)
analyze = _m.analyze
validate_result = _m.validate_result


def main():
    ap = argparse.ArgumentParser(description="Annotate a kinase structure (DFG/inhibitor class)")
    ap.add_argument("identifier", nargs="?", help="PDB code present in KLIFS")
    ap.add_argument("--klifs-id", type=int, help="KLIFS structure_ID")
    ap.add_argument("--local", help="Path to a KLIFS-format 85-residue pocket .mol2")
    ap.add_argument("--pdb-file", help="Path to an arbitrary PDB (crystal or AlphaFold); needs --kinase")
    ap.add_argument("--kinase", help="Kinase name (required with --pdb-file)")
    ap.add_argument("--chain", help="Chain to use in --pdb-file (default: longest)")
    ap.add_argument("--provenance", default="experimental",
                    choices=["experimental", "predicted", "unknown"],
                    help="Tag input as experimental or predicted (AlphaFold)")
    ap.add_argument("--pymol", metavar="OUT.pml",
                    help="Also write an annotated PyMOL session script")
    args = ap.parse_args()

    if args.pdb_file:
        if not args.kinase:
            ap.error("--pdb-file requires --kinase")
        res = analyze(args.identifier, provenance=args.provenance,
                      local_pdb=args.pdb_file, kinase=args.kinase, chain=args.chain,
                      pymol_out=args.pymol)
    elif args.local:
        ident = args.identifier or Path(args.local).stem
        res = analyze(ident, provenance=args.provenance, local_mol2=args.local,
                      pymol_out=args.pymol)
    elif args.klifs_id is not None:
        res = analyze(args.klifs_id, provenance=args.provenance, pymol_out=args.pymol)
    elif args.identifier:
        res = analyze(args.identifier, provenance=args.provenance, pymol_out=args.pymol)
    else:
        ap.error("provide a PDB code, --klifs-id, --local <mol2>, or --pdb-file <pdb> --kinase <name>")

    out = res.to_dict()
    print(json.dumps(out, indent=2))

    errors = validate_result(out)
    if errors:
        print("Error: output failed schema validation:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(3)

    if args.pymol:
        print(f"\nPyMOL session written to: {args.pymol}\n  open with: pymol {args.pymol}")


if __name__ == "__main__":
    main()
