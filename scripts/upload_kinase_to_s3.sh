#!/bin/bash
# Upload Kinase Data to S3 (Safe - doesn't touch old data)
# Bucket: arn:aws:s3:::amzn-s3-proteinbucket

set -e

BUCKET="s3://amzn-s3-proteinbucket"
PREFIX="hyaline/kinase"

echo "=========================================="
echo "Upload Kinase Data to S3"
echo "=========================================="
echo ""
echo "Bucket: $BUCKET"
echo "Prefix: $PREFIX"
echo ""

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials not configured."
    exit 1
fi

echo "✓ AWS credentials valid"
echo ""

# Upload kinase data
echo "[1/3] Uploading KLIFS cache (12 MB)..."
aws s3 sync ./klifs_cache/ $BUCKET/$PREFIX/klifs_cache/ \
    --exclude "*.pyc" \
    --exclude "__pycache__/*"

echo "✓ KLIFS cache uploaded"
echo ""

echo "[2/3] Uploading data/klifs_cache (12 MB)..."
aws s3 sync ./data/klifs_cache/ $BUCKET/$PREFIX/data/klifs_cache/ \
    --exclude "*.pyc"

echo "✓ Data cache uploaded"
echo ""

echo "[3/3] Uploading checkpoints (JSON only, 11 MB)..."
aws s3 sync ./checkpoints/ $BUCKET/$PREFIX/checkpoints/ \
    --exclude "*.pyc" \
    --exclude "*.pt" \
    --include "*.json"

echo "✓ Checkpoints uploaded"
echo ""

# Verify upload
echo "Verifying upload..."
aws s3 ls $BUCKET/$PREFIX/ --recursive --human-readable --summarize | tail -2

echo ""
echo "=========================================="
echo "✓ Upload Complete!"
echo "=========================================="
echo ""
echo "Students can download with:"
echo "  aws s3 sync $BUCKET/$PREFIX/ ./ --region us-east-1"
echo ""
echo "Or specific folders:"
echo "  aws s3 sync $BUCKET/$PREFIX/klifs_cache/ ./klifs_cache/"
echo "  aws s3 sync $BUCKET/$PREFIX/data/klifs_cache/ ./data/klifs_cache/"
echo "  aws s3 sync $BUCKET/$PREFIX/checkpoints/ ./checkpoints/"
echo ""
