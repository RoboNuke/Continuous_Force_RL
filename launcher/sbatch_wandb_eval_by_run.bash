#!/bin/bash

# WandB Eval by Run ID Launch Script for HPC
# Usage: ./launcher/sbatch_wandb_eval_by_run.bash --run_ids "run_id1 run_id2 run_id3" --tag "tag_name" [--video/--no-video]

# Default values
RUN_IDS=""
TAG=""
VIDEO_ENABLED="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_ids)
            RUN_IDS="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --run_ids \"run_id1 run_id2 run_id3\" --tag \"tag_name\" [--video/--no-video]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RUN_IDS" ]]; then
    echo "Error: --run_ids is required"
    echo "Usage: $0 --run_ids \"run_id1 run_id2 run_id3\" --tag \"tag_name\" [--video/--no-video]"
    exit 1
fi

if [[ -z "$TAG" ]]; then
    echo "Error: --tag is required"
    echo "Usage: $0 --run_ids \"run_id1 run_id2 run_id3\" --tag \"tag_name\" [--video/--no-video]"
    exit 1
fi

echo "=== WandB Eval by Run ID Launch Script ==="
echo "Run IDs: $RUN_IDS"
echo "Tag: $TAG"
echo "Video Enabled: $VIDEO_ENABLED"
echo ""

# Convert run_ids string to array
read -ra RUN_ID_ARRAY <<< "$RUN_IDS"

# Process each run ID
for run_id in "${RUN_ID_ARRAY[@]}"; do
    # Create job name (use run_id as part of name)
    job_name="WANDB_EVAL_${run_id}"

    echo "Submitting WandB eval job: $job_name"
    echo "  Run ID: $run_id"
    echo "  Tag: $TAG"
    echo "  Video: $VIDEO_ENABLED"

    # Submit SLURM job with dynamic output file paths
    output_name="wandb_eval_${run_id}"
    sbatch -J "$job_name" \
           -o "exp_logs/wandb_eval/${output_name}_%j.out" \
           -e "exp_logs/wandb_eval/${output_name}_%j.err" \
           launcher/hpc_wandb_eval_by_run_batch.bash "$run_id" "$TAG" "$VIDEO_ENABLED"

    echo "  Job submitted successfully"
    echo ""
done

echo "All WandB eval jobs submitted!"
