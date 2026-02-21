#!/bin/bash
# Upload Hyaline Kinase Data to S3
# Usage: ./scripts/upload_to_s3.sh

set -e

BUCKET="s3://hyaline-kinase-data"
REGION="us-east-1"

echo "=========================================="
echo "Hyaline Kinase Data Upload to S3"
echo "=========================================="
echo ""
echo "Bucket: $BUCKET"
echo "Region: $REGION"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found. Please install it first."
    echo "  pip install awscli"
    exit 1
fi

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials not configured."
    echo "  Run: aws configure"
    exit 1
fi

echo "✓ AWS credentials valid"
echo ""

# Upload KLIFS cache
echo "[1/3] Uploading KLIFS cache (12 MB)..."
aws s3 sync ./klifs_cache/ $BUCKET/klifs_cache/ \
    --region $REGION \
    --exclude "*.pyc" \
    --exclude "__pycache__/*"

echo "✓ KLIFS cache uploaded"
echo ""

# Upload checkpoints
echo "[2/3] Uploading checkpoints (11 MB)..."
aws s3 sync ./checkpoints/ $BUCKET/checkpoints/ \
    --region $REGION \
    --exclude "*.pyc"

echo "✓ Checkpoints uploaded"
echo ""

# Upload data/klifs_cache
echo "[3/3] Uploading data/klifs_cache (12 MB)..."
aws s3 sync ./data/klifs_cache/ $BUCKET/data/klifs_cache/ \
    --region $REGION \
    --exclude "*.pyc"

echo "✓ Data cache uploaded"
echo ""

# Verify upload
echo "Verifying upload..."
aws s3 ls $BUCKET/ --recursive --human-readable --summarize | tail -2

echo ""
echo "=========================================="
echo "✓ Upload complete!"
echo "=========================================="
echo ""
echo "Students can download with:"
echo "  aws s3 sync $BUCKET/ ./data/ --region $REGION"
echo ""
