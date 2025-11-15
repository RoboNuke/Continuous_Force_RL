#!/bin/bash

# WandB Eval Launch Script for HPC
# Usage: ./launcher/sbatch_wandb_eval.bash --tags "tag1 tag2 tag3" [--video/--no-video]

# Default values
TAGS=""
VIDEO_ENABLED="false"
EVAL_MODE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --video)
            VIDEO_ENABLED="true"
            shift 1
            ;;
        --no-video)
            VIDEO_ENABLED="false"
            shift 1
            ;;
        --eval_mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --tags \"tag1 tag2 tag3\" --eval_mode \"performance|noise|rotation\" [--video/--no-video]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TAGS" ]]; then
    echo "Error: --tags is required"
    echo "Usage: $0 --tags \"tag1 tag2 tag3\" --eval_mode \"performance|noise|rotation\" [--video/--no-video]"
    exit 1
fi

if [[ -z "$EVAL_MODE" ]]; then
    echo "Error: --eval_mode is required"
    echo "Usage: $0 --tags \"tag1 tag2 tag3\" --eval_mode \"performance|noise|rotation\" [--video/--no-video]"
    exit 1
fi

# Validate eval_mode
if [[ "$EVAL_MODE" != "performance" && "$EVAL_MODE" != "noise" && "$EVAL_MODE" != "rotation" ]]; then
    echo "Error: --eval_mode must be either 'performance', 'noise', or 'rotation'"
    exit 1
fi

echo "=== WandB Eval Launch Script ==="
echo "Tags: $TAGS"
echo "Eval Mode: $EVAL_MODE"
echo "Video Enabled: $VIDEO_ENABLED"
echo ""

# Convert tags string to array
read -ra TAG_ARRAY <<< "$TAGS"

# Process each tag
for tag in "${TAG_ARRAY[@]}"; do
    # Create job name (replace colons and slashes with underscores for valid job names)
    job_name="WANDB_EVAL_$(echo "${tag}" | tr ':/' '__')"

    echo "Submitting WandB eval job: $job_name"
    echo "  Tag: $tag"
    echo "  Eval Mode: $EVAL_MODE"
    echo "  Video: $VIDEO_ENABLED"

    # Submit SLURM job with dynamic output file paths
    output_name="wandb_eval_$(echo "${tag}" | tr ':/' '__')"
    sbatch -J "$job_name" \
           -o "exp_logs/wandb_eval/${output_name}_%j.out" \
           -e "exp_logs/wandb_eval/${output_name}_%j.err" \
           launcher/hpc_wandb_eval_batch.bash "$tag" "$VIDEO_ENABLED" "$EVAL_MODE"

    echo "  Job submitted successfully"
    echo ""
done

echo "All WandB eval jobs submitted!"
