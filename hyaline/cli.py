#!/usr/bin/env python3
"""
Hyaline CLI
===========

Command-line interface for GPCR activation prediction and transcription
factor function modeling.

Usage:
    hyaline predict structure.pdb
    hyaline tf-predict sequence.fasta
    hyaline --help
"""
import argparse
import sys
from pathlib import Path


def predict_command(args):
    """Run GPCR activation prediction on a PDB file."""
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


def tf_predict_command(args):
    """Run TF function prediction on a FASTA file."""
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from hyaline import HyalineTF, TF_FUNCTION_CLASSES
    from hyaline.tf_data import load_tf_sequences, get_esm_embeddings, sequence_to_data

    fasta_path = args.input
    checkpoint = args.checkpoint
    device = args.device

    if not Path(fasta_path).exists():
        print(f"Error: File not found: {fasta_path}")
        sys.exit(1)

    print("=" * 60)
    print("HYALINE TF PREDICTION")
    print("=" * 60)

    sequences = load_tf_sequences(fasta_path)
    if not sequences:
        print("Error: No sequences found in FASTA file.")
        sys.exit(1)

    names = [n for n, _ in sequences]
    seqs = [s for _, s in sequences]

    print(f"\nInput: {fasta_path} ({len(seqs)} sequence(s))")
    print("Computing ESM embeddings...")
    embeddings = get_esm_embeddings(seqs, device=device)

    data_list = [
        sequence_to_data(seq, emb)
        for seq, emb in zip(seqs, embeddings)
    ]

    model = HyalineTF(
        node_input_dim=embeddings[0].shape[1],
        hidden_dim=256,
        num_layers=4,
        use_domain_bias=True,
    ).to(device)

    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
            print("Warning: Loaded with strict=False due to architecture mismatch")
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        if checkpoint:
            print(f"Warning: Checkpoint not found: {checkpoint}")
        print("Using untrained model (results will be random).")

    model.eval()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, data in zip(names, data_list):
        # Add batch dimension
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        data = data.to(device)

        with torch.no_grad():
            out = model(data)

        func_probs = torch.softmax(out['function_logits'], dim=-1).squeeze(0)
        func_idx = func_probs.argmax().item()
        binding = out['binding'].item()
        regulatory = out['regulatory'].item()

        print(f"\n  {name}")
        print(f"    TF function:    {TF_FUNCTION_CLASSES[func_idx]}"
              f"  (conf {func_probs[func_idx]:.2f})")
        print(f"    DNA binding:    {binding:.4f}")
        print(f"    Regulatory:     {regulatory:.4f}")

    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='hyaline',
        description=(
            'Hyaline: Geometric Deep Learning for Protein Function Modeling\n'
            '  • GPCR activation state prediction (structure-based)\n'
            '  • Transcription factor function modeling (sequence or structure)'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ── predict (GPCR) ────────────────────────────────────────────────────────
    predict_parser = subparsers.add_parser(
        'predict', help='Predict GPCR activation state from a PDB file'
    )
    predict_parser.add_argument('input', type=str, help='Path to PDB file')
    predict_parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to model checkpoint (default: bundled model)',
    )
    predict_parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device (default: cuda)',
    )

    # ── tf-predict (TF) ───────────────────────────────────────────────────────
    tf_parser = subparsers.add_parser(
        'tf-predict',
        help='Predict TF function, DNA-binding, and regulatory impact from FASTA',
    )
    tf_parser.add_argument('input', type=str, help='Path to FASTA file')
    tf_parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to HyalineTF checkpoint (optional)',
    )
    tf_parser.add_argument(
        '--device', '-d',
        type=str,
        default='cpu',
        choices=['cuda', 'cpu'],
        help='Device (default: cpu)',
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == 'predict':
        predict_command(args)
    elif args.command == 'tf-predict':
        tf_predict_command(args)


if __name__ == '__main__':
    main()
