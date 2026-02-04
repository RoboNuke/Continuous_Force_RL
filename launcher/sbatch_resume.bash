#!/bin/bash

# Resume Training Launch Script for HPC
# Usage: ./launcher/sbatch_resume.bash --tags "tag1 tag2 tag3" [--overrides "key=value key2=value2"]

# Default values
TAGS=""
OVERRIDES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --overrides)
            OVERRIDES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --tags \"tag1 tag2 tag3\" [--overrides \"key=value key2=value2\"]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TAGS" ]]; then
    echo "Error: --tags is required"
    echo "Usage: $0 --tags \"tag1 tag2 tag3\" [--overrides \"key=value key2=value2\"]"
    exit 1
fi

echo "=== Resume Training Launch Script ==="
echo "Tags: $TAGS"
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
    if [[ -n "$OVERRIDES" ]]; then
        echo "  Overrides: $OVERRIDES"
    fi

    # Submit SLURM job with dynamic output file paths
    sbatch -J "$job_name" \
           -o "exp_logs/resume/${tag_name}_%j.out" \
           -e "exp_logs/resume/${tag_name}_%j.err" \
           launcher/hpc_batch_resume.bash "$tag_name" "$OVERRIDES"

    echo "  Job submitted successfully"
    echo ""
done

echo "All jobs submitted!"
