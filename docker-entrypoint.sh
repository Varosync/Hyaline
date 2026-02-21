#!/bin/bash
set -e

# Hyaline Docker Entrypoint
# Supports: train, predict, evaluate, shell

case "$1" in
    train)
        shift
        echo "Starting TF Activation training..."
        python /app/scripts/train_robust.py "$@"
        ;;
    
    train-real)
        shift
        echo "Training with real PDB structures..."
        python /app/scripts/train_tf_real.py "$@"
        ;;
    
    predict)
        shift
        echo "Running prediction..."
        python -c "
import sys
sys.path.insert(0, '/app')
import torch
from pathlib import Path
from hyaline.models.tf_activation_model import TFActivationModel, TFActivationConfig
from hyaline.loaders.pdb_loader import load_pdb_structure
from hyaline.loaders.tf_activation_data import SCENICContext

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('/app/checkpoints/tf_robust/best.pt', map_location=device)
config = TFActivationConfig(**ckpt['config'])
model = TFActivationModel(config).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

# Load structure
pdb_path = Path('$1')
if not pdb_path.exists():
    print(f'Error: PDB file not found: {pdb_path}')
    sys.exit(1)

struct = load_pdb_structure(pdb_path, tf_name='${2:-Unknown}')
cell_type = '${3:-melanocyte}'
tf_name = '${2:-Unknown}'

ctx = SCENICContext.from_cell_type(cell_type, tf_name)

with torch.no_grad():
    out = model(
        torch.from_numpy(struct.node_features).float().to(device),
        torch.from_numpy(struct.pos).float().to(device),
        torch.from_numpy(struct.edge_index).long().to(device),
        torch.from_numpy(struct.edge_attr).float().to(device),
        torch.tensor([ctx.cell_type_idx]).long().to(device),
        torch.from_numpy(ctx.tf_activity).float().unsqueeze(0).to(device),
        torch.from_numpy(ctx.chromatin_topics).float().unsqueeze(0).to(device),
        torch.from_numpy(ctx.coactivator_expr).float().unsqueeze(0).to(device),
    )

print(f'TF: {tf_name}')
print(f'Cell Type: {cell_type}')
print(f'Activation Probability: {out.activation_prob.item():.4f}')
print(f'Confidence: {out.confidence.item():.4f}')
print(f'Sync Score: {out.sync_score.item():.4f}')
"
        ;;
    
    evaluate)
        shift
        echo "Evaluating on validation cases..."
        python -c "
import sys
sys.path.insert(0, '/app')
import torch
from pathlib import Path
from hyaline.models.tf_activation_model import TFActivationModel, TFActivationConfig
from hyaline.loaders.pdb_loader import load_tf_structures
from hyaline.loaders.tf_activation_data import SCENICContext, VALIDATION_CASES, EXPECTED_LEVELS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('/app/checkpoints/tf_robust/best.pt', map_location=device)
config = TFActivationConfig(**ckpt['config'])
model = TFActivationModel(config).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

structures = load_tf_structures(Path('/app/data/tf_dna_structures'))

correct = 0
for tf_name, cell_type, expected in VALIDATION_CASES:
    if tf_name not in structures:
        continue
    struct = structures[tf_name][0]
    ctx = SCENICContext.from_cell_type(cell_type, tf_name, noise_scale=0.0)
    
    with torch.no_grad():
        out = model(
            torch.from_numpy(struct.node_features).float().to(device),
            torch.from_numpy(struct.pos).float().to(device),
            torch.from_numpy(struct.edge_index).long().to(device),
            torch.from_numpy(struct.edge_attr).float().to(device),
            torch.tensor([ctx.cell_type_idx]).long().to(device),
            torch.from_numpy(ctx.tf_activity).float().unsqueeze(0).to(device),
            torch.from_numpy(ctx.chromatin_topics).float().unsqueeze(0).to(device),
            torch.from_numpy(ctx.coactivator_expr).float().unsqueeze(0).to(device),
        )
    
    pred = out.activation_prob.item()
    expected_prob = EXPECTED_LEVELS.get(expected, 0.5)
    is_correct = (pred > 0.5) == (expected_prob > 0.5)
    correct += int(is_correct)
    symbol = 'Y' if is_correct else 'N'
    print(f'{tf_name:10} {cell_type:15} {expected:12} {pred:.4f} {symbol}')

print(f'Accuracy: {correct}/9')
"
        ;;
    
    shell)
        exec /bin/bash
        ;;
    
    --help|help)
        echo "Hyaline TF Activation Model"
        echo ""
        echo "Commands:"
        echo "  train        Train the model with SCENIC+ context"
        echo "  train-real   Train with real PDB structures"  
        echo "  predict      Predict TF activation for a structure"
        echo "  evaluate     Evaluate on biological validation cases"
        echo "  shell        Start interactive shell"
        echo ""
        echo "Examples:"
        echo "  docker run --gpus all hyaline:latest train --epochs 50"
        echo "  docker run --gpus all -v \$(pwd)/data:/app/data hyaline:latest evaluate"
        ;;
    
    *)
        exec "$@"
        ;;
esac
