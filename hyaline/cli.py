#!/usr/bin/env python3
"""
Hyaline CLI
===========

Command-line interface for GPCR activation prediction.

Usage:
    hyaline predict structure.pdb
    hyaline predict /path/to/pdb_directory/
    hyaline predict --help
"""
import argparse
import sys
from pathlib import Path


def predict_command(args):
    """Run prediction on a PDB file or all PDB files in a directory."""
    from hyaline.predict import predict, predict_batch
    
    input_path = Path(args.input)
    checkpoint = args.checkpoint
    device = args.device
    allow_random = args.allow_random
    
    if not input_path.exists():
        print(f"Error: Path not found: {args.input}")
        sys.exit(1)
    
    if input_path.is_dir():
        return predict_batch(
            str(input_path), checkpoint, device, allow_random,
            output_csv=args.output
        )
    
    score, prediction = predict(str(input_path), checkpoint, device, allow_random)
    
    if score is None:
        sys.exit(1)
    
    return score, prediction


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='hyaline',
        description='Geometric Deep Learning for GPCR Activation State Prediction'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict GPCR activation state')
    predict_parser.add_argument('input', type=str, help='Path to PDB file or directory of PDB files')
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
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV path for batch results (default: <input_dir>/hyaline_results.csv)'
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


if __name__ == '__main__':
    main()
