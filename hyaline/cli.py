#!/usr/bin/env python3
"""
Hyaline CLI
===========

Command-line interface for GPCR activation prediction.

Usage:
    hyaline predict structure.pdb        # GPCR activation state
    hyaline analyze 2hyy                  # kinase DFG / inhibitor-class call
    hyaline predict --help
"""
import argparse
import json
import sys
from pathlib import Path


def analyze_command(args):
    """Annotate a kinase structure with its DFG state + inhibitor class.

    Dependency-light path (numpy + requests): imported lazily so this command
    works in a kinase-only install without torch / torch_geometric.
    """
    from hyaline.kinase.analyze import analyze, validate_result

    if args.pdb_file:
        if not args.kinase:
            print("Error: --pdb-file requires --kinase", file=sys.stderr)
            sys.exit(2)
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
        print("Error: provide a PDB code, --klifs-id, --local <mol2>, "
              "or --pdb-file <pdb> --kinase <name>", file=sys.stderr)
        sys.exit(2)

    out = res.to_dict()
    print(json.dumps(out, indent=2))

    errors = validate_result(out)
    if errors:
        print("Error: output failed schema validation:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(3)

    if args.pymol:
        print(f"\nPyMOL session written to: {args.pymol}"
              f"\n  open with: pymol {args.pymol}", file=sys.stderr)
    return res


def predict_command(args):
    """Run prediction on a PDB file."""
    from hyaline.predict import predict
    
    pdb_path = args.input
    checkpoint = args.checkpoint
    device = args.device
    allow_random = args.allow_random
    
    if not Path(pdb_path).exists():
        print(f"Error: File not found: {pdb_path}")
        sys.exit(1)
    
    score, prediction = predict(pdb_path, checkpoint, device, allow_random)
    
    if score is None:
        sys.exit(1)
    
    return score, prediction


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='hyaline',
        description='Geometric Deep Learning for GPCR and Kinase Conformational State Prediction'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command (kinase DFG / inhibitor-class annotation)
    analyze_parser = subparsers.add_parser(
        'analyze', help='Annotate a kinase structure (DFG state + inhibitor class)')
    analyze_parser.add_argument('identifier', nargs='?', help='PDB code present in KLIFS')
    analyze_parser.add_argument('--klifs-id', type=int, help='KLIFS structure_ID')
    analyze_parser.add_argument('--local', help='Path to a KLIFS-format 85-residue pocket .mol2')
    analyze_parser.add_argument('--pdb-file',
                                help='Path to an arbitrary PDB (crystal or AlphaFold); needs --kinase')
    analyze_parser.add_argument('--kinase', help='Kinase name (required with --pdb-file)')
    analyze_parser.add_argument('--chain', help='Chain to use in --pdb-file (default: longest)')
    analyze_parser.add_argument('--provenance', default='experimental',
                                choices=['experimental', 'predicted', 'unknown'],
                                help='Tag input as experimental or predicted (AlphaFold)')
    analyze_parser.add_argument('--pymol', metavar='OUT.pml',
                                help='Also write an annotated PyMOL session script')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict GPCR activation state')
    predict_parser.add_argument('input', type=str, help='Path to PDB file')
    predict_parser.add_argument(
        '--checkpoint', '-c', 
        type=str, 
        default=None,
        help='Path to model checkpoint (default: bundled model)'
    )
    predict_parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: cuda)'
    )
    predict_parser.add_argument(
        '--allow-random',
        action='store_true',
        default=False,
        help='Allow random embeddings and untrained model (testing only)'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'predict':
        predict_command(args)
    elif args.command == 'analyze':
        analyze_command(args)


if __name__ == '__main__':
    main()
