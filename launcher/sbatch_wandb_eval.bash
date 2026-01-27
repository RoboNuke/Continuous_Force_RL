#!/bin/bash

# WandB Eval Launch Script for HPC
# Usage: ./launcher/sbatch_wandb_eval.bash --tags "tag1 tag2 tag3" --eval_modes "performance noise rotation" [--video/--no-video] [--checkpoint_range start:end:step]

# Default values
TAGS=""
VIDEO_ENABLED="false"
EVAL_MODES=""
PROJECT=""
CHECKPOINT_RANGE=""

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
        --eval_modes)
            EVAL_MODES="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --checkpoint_range)
            CHECKPOINT_RANGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --tags \"tag1 tag2 tag3\" --eval_modes \"performance noise rotation\" [--video/--no-video] [--checkpoint_range start:end:step]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TAGS" ]]; then
    echo "Error: --tags is required"
    echo "Usage: $0 --tags \"tag1 tag2 tag3\" --eval_modes \"performance noise rotation\" [--video/--no-video] [--checkpoint_range start:end:step]"
    exit 1
fi

if [[ -z "$EVAL_MODES" ]]; then
    echo "Error: --eval_modes is required"
    echo "Usage: $0 --tags \"tag1 tag2 tag3\" --eval_modes \"performance noise rotation\" [--video/--no-video] [--checkpoint_range start:end:step]"
    exit 1
fi

# Convert eval_modes string to array
read -ra EVAL_MODE_ARRAY <<< "$EVAL_MODES"

# Validate each eval_mode
for eval_mode in "${EVAL_MODE_ARRAY[@]}"; do
    if [[ "$eval_mode" != "performance" && "$eval_mode" != "noise" && "$eval_mode" != "rotation" && "$eval_mode" != "dynamics" && "$eval_mode" != "gain" && "$eval_mode" != "yaw" ]]; then
        echo "Error: Invalid eval_mode '$eval_mode'. Must be 'performance', 'noise', 'rotation', 'dynamics', 'gain', or 'yaw'"
        exit 1
    fi
done

echo "=== WandB Eval Launch Script ==="
echo "Tags: $TAGS"
echo "Eval Modes: $EVAL_MODES"
echo "Video Enabled: $VIDEO_ENABLED"
echo "Project: ${PROJECT:-<default>}"
echo "Checkpoint Range: ${CHECKPOINT_RANGE:-<all>}"
echo ""

# Convert tags string to array
read -ra TAG_ARRAY <<< "$TAGS"

# Process each tag and eval_mode combination
for tag in "${TAG_ARRAY[@]}"; do
    for eval_mode in "${EVAL_MODE_ARRAY[@]}"; do
        # Create job name (replace colons and slashes with underscores for valid job names)
        job_name="EVAL_$(echo "${tag}" | tr ':/' '__')_${eval_mode}"

        echo "Submitting WandB eval job: $job_name"
        echo "  Tag: $tag"
        echo "  Eval Mode: $eval_mode"
        echo "  Video: $VIDEO_ENABLED"

        # Submit SLURM job with dynamic output file paths
        output_name="eval_$(echo "${tag}" | tr ':/' '__')_${eval_mode}"
        sbatch -J "$job_name" \
               -o "exp_logs/wandb_eval/${output_name}_%j.out" \
               -e "exp_logs/wandb_eval/${output_name}_%j.err" \
               launcher/hpc_wandb_eval_batch.bash "$tag" "$VIDEO_ENABLED" "$eval_mode" "$PROJECT" "$CHECKPOINT_RANGE"

        echo "  Job submitted successfully"
        echo ""
    done
done

echo "All WandB eval jobs submitted!"
