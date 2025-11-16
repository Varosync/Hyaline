#!/usr/bin/env python3
"""
Hyaline CLI
===========

Command-line interface for GPCR activation prediction.

Usage:
    hyaline predict structure.pdb
    hyaline predict --help
"""
import argparse
import sys
from pathlib import Path


def predict_command(args):
    """Run prediction on a PDB file."""
    from hyaline.predict import predict
    
    pdb_path = args.input
    checkpoint = args.checkpoint
    device = args.device
    
    if not Path(pdb_path).exists():
        print(f"Error: File not found: {pdb_path}")
        sys.exit(1)
    
    score, prediction = predict(pdb_path, checkpoint, device)
    
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
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'predict':
        predict_command(args)


if __name__ == '__main__':
    main()
