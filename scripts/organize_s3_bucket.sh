#!/bin/bash
# Organize S3 Bucket for Hyaline Project
# Bucket: arn:aws:s3:::amzn-s3-proteinbucket

set -e

BUCKET="s3://amzn-s3-proteinbucket"

echo "=========================================="
echo "S3 Bucket Organization for Hyaline"
echo "=========================================="
echo ""
echo "Bucket: $BUCKET"
echo ""

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials not configured."
    exit 1
fi

echo "✓ AWS credentials valid"
echo ""

# Current structure:
# - Hyaline/ (older, 15.3 GB, 1648 objects) - has scripts + data
# - hyaline/ (newer, 15.3 GB, 1627 objects) - has figures in results/
# - MLCrunchData/ (318 MB, 10 objects) - keep this

echo "Current bucket structure:"
echo "  Hyaline/  - 15.3 GB (older, has scripts)"
echo "  hyaline/  - 15.3 GB (newer, has figures)"
echo "  MLCrunchData/ - 318 MB (keep)"
echo ""

# New structure we want:
# hyaline/
# ├── gpcr/              # Original GPCR work
# │   ├── data/          # GPCR PDB files (1596 files, ~15 GB)
# │   └── checkpoints/   # GPCR model checkpoint
# ├── kinase/            # New kinase work
# │   ├── klifs_cache/   # KLIFS structures (12 MB)
# │   ├── data/          # Additional data (12 MB)
# │   └── checkpoints/   # Kinase checkpoints (11 MB)
# └── MLCrunchData/      # Separate project (keep as-is)

echo "Target structure:"
echo "  hyaline/"
echo "    ├── gpcr/data/          # GPCR PDB files"
echo "    ├── gpcr/checkpoints/   # GPCR models"
echo "    ├── kinase/klifs_cache/ # KLIFS data"
echo "    ├── kinase/data/        # Additional kinase data"
echo "    └── kinase/checkpoints/ # Kinase models"
echo "  MLCrunchData/             # Keep as-is"
echo ""

read -p "Proceed with reorganization? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1: Create new structure..."
echo ""

# Move GPCR data to organized location
echo "[1/5] Moving GPCR data..."
aws s3 sync $BUCKET/Hyaline/data/gpcrdb_all/ $BUCKET/hyaline/gpcr/data/gpcrdb_all/ \
    --exclude "*.gitkeep" \
    --quiet

echo "✓ GPCR data moved"

# Move GPCR checkpoint
echo "[2/5] Moving GPCR checkpoint..."
aws s3 cp $BUCKET/Hyaline/checkpoints/hyaline.pt $BUCKET/hyaline/gpcr/checkpoints/hyaline_gpcr.pt \
    --quiet

echo "✓ GPCR checkpoint moved"

# Upload kinase data
echo "[3/5] Uploading kinase KLIFS cache..."
aws s3 sync ./klifs_cache/ $BUCKET/hyaline/kinase/klifs_cache/ \
    --exclude "*.pyc" \
    --exclude "__pycache__/*" \
    --quiet

echo "✓ Kinase KLIFS cache uploaded"

echo "[4/5] Uploading kinase data cache..."
aws s3 sync ./data/klifs_cache/ $BUCKET/hyaline/kinase/data/klifs_cache/ \
    --exclude "*.pyc" \
    --quiet

echo "✓ Kinase data cache uploaded"

echo "[5/5] Uploading kinase checkpoints..."
aws s3 sync ./checkpoints/ $BUCKET/hyaline/kinase/checkpoints/ \
    --exclude "*.pyc" \
    --exclude "*.pt" \
    --include "*.json" \
    --quiet

echo "✓ Kinase checkpoints uploaded"

echo ""
echo "Step 2: Remove old/redundant data..."
echo ""

# Remove figures (not needed in S3)
echo "Removing figures from results/..."
aws s3 rm $BUCKET/hyaline/results/figures/ --recursive --quiet || true
aws s3 rm $BUCKET/Hyaline/results/ --recursive --quiet || true

echo "✓ Figures removed"

# Remove scripts (should be in git, not S3)
echo "Removing scripts (should be in git)..."
aws s3 rm $BUCKET/Hyaline/scripts/ --recursive --quiet || true

echo "✓ Scripts removed"

# Remove README (should be in git)
aws s3 rm $BUCKET/Hyaline/README.md --quiet || true

echo "✓ README removed"

echo ""
echo "Step 3: Clean up old folders..."
echo ""

read -p "Delete old 'Hyaline/' folder? (yes/no): " delete_old
if [ "$delete_old" == "yes" ]; then
    echo "Deleting Hyaline/ folder..."
    aws s3 rm $BUCKET/Hyaline/ --recursive --quiet
    echo "✓ Old Hyaline/ folder deleted"
fi

echo ""
echo "=========================================="
echo "✓ S3 Bucket Organized!"
echo "=========================================="
echo ""

# Show final structure
echo "Final structure:"
aws s3 ls $BUCKET/ --recursive --human-readable --summarize | tail -10

echo ""
echo "Access instructions for students:"
echo ""
echo "# Download GPCR data (original Hyaline)"
echo "aws s3 sync $BUCKET/hyaline/gpcr/ ./gpcr/ --region us-east-1"
echo ""
echo "# Download kinase data (new work)"
echo "aws s3 sync $BUCKET/hyaline/kinase/ ./kinase/ --region us-east-1"
echo ""
