#!/bin/bash

# Resume Training Launch Script for HPC
# Usage: ./launcher/sbatch_resume.bash --tags "tag1 tag2 tag3" [--checkpoint_step STEP] [--overrides "key=value key2=value2"]

# Default values
TAGS=""
CHECKPOINT_STEP=""
OVERRIDES=""
NEW_PROJECT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --checkpoint_step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --overrides)
            OVERRIDES="$2"
            shift 2
            ;;
        --new_project)
            NEW_PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --tags \"tag1 tag2 tag3\" [--checkpoint_step STEP] [--overrides \"key=value key2=value2\"] [--new_project PROJECT]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TAGS" ]]; then
    echo "Error: --tags is required"
    echo "Usage: $0 --tags \"tag1 tag2 tag3\" [--checkpoint_step STEP] [--overrides \"key=value key2=value2\"] [--new_project PROJECT]"
    exit 1
fi

echo "=== Resume Training Launch Script ==="
echo "Tags: $TAGS"
if [[ -n "$CHECKPOINT_STEP" ]]; then
    echo "Checkpoint Step: $CHECKPOINT_STEP"
else
    echo "Checkpoint Step: (auto-discover latest)"
fi
if [[ -n "$NEW_PROJECT" ]]; then
    echo "New Project: $NEW_PROJECT"
fi
echo "Overrides: $OVERRIDES"
echo ""

# Convert tags string to array
read -ra TAG_ARRAY <<< "$TAGS"

# Create log directory
mkdir -p "exp_logs/resume"

# Process each tag
for tag_name in "${TAG_ARRAY[@]}"; do
    # Create job name (use tag directly)
    job_name="resume_${tag_name}"

    echo "Submitting job: $job_name"
    echo "  Checkpoint Tag: $tag_name"
    if [[ -n "$CHECKPOINT_STEP" ]]; then
        echo "  Checkpoint Step: $CHECKPOINT_STEP"
    fi
    if [[ -n "$OVERRIDES" ]]; then
        echo "  Overrides: $OVERRIDES"
    fi

    # Build args for hpc script - only pass non-empty args
    HPC_ARGS="--checkpoint_tag $tag_name"
    if [[ -n "$CHECKPOINT_STEP" ]]; then
        HPC_ARGS="$HPC_ARGS --checkpoint_step $CHECKPOINT_STEP"
    fi
    if [[ -n "$OVERRIDES" ]]; then
        HPC_ARGS="$HPC_ARGS --overrides $OVERRIDES"
    fi
    if [[ -n "$NEW_PROJECT" ]]; then
        HPC_ARGS="$HPC_ARGS --new_project $NEW_PROJECT"
    fi

    # Submit SLURM job with dynamic output file paths
    sbatch -J "$job_name" \
           -o "exp_logs/resume/${tag_name}_%j.out" \
           -e "exp_logs/resume/${tag_name}_%j.err" \
           launcher/hpc_batch_resume.bash $HPC_ARGS

    echo "  Job submitted successfully"
    echo ""
done

echo "All jobs submitted!"
