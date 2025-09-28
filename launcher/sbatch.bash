#!/bin/bash

# New Test Launch Script for HPC
# Usage: ./launcher/sbatch.bash --configs "config1 config2" --experiment_tag "EXP_TAG" --overrides "key=value key2=value2"

# Default values
CONFIGS=""
EXPERIMENT_TAG=""
OVERRIDES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --configs)
            CONFIGS="$2"
            shift 2
            ;;
        --experiment_tag)
            EXPERIMENT_TAG="$2"
            shift 2
            ;;
        --overrides)
            OVERRIDES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIGS" ]]; then
    echo "Error: --configs is required"
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
    exit 1
fi

if [[ -z "$EXPERIMENT_TAG" ]]; then
    echo "Error: --experiment_tag is required"
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
    exit 1
fi

echo "=== Test Launch Script ==="
echo "Configs: $CONFIGS"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "Overrides: $OVERRIDES"
echo ""

# Convert configs string to array
read -ra CONFIG_ARRAY <<< "$CONFIGS"

# Process each config
for config_name in "${CONFIG_ARRAY[@]}"; do
    # Construct full config path
    config_path="configs/experiments/${config_name}.yaml"

    # Check if config file exists
    if [[ ! -f "$config_path" ]]; then
        echo "Error: Config file not found: $config_path"
        continue
    fi

    # Create job name
    job_name="${EXPERIMENT_TAG}_${config_name}"

    echo "Submitting job: $job_name"
    echo "  Config: $config_path"
    echo "  Experiment Tag: $EXPERIMENT_TAG"
    if [[ -n "$OVERRIDES" ]]; then
        echo "  Overrides: $OVERRIDES"
    fi

    # Submit SLURM job with dynamic output file paths
    sbatch -J "$job_name" \
           -o "../exp_logs/${EXPERIMENT_TAG}/${config_name}_%j.out" \
           -e "../exp_logs/${EXPERIMENT_TAG}/${config_name}_%j.err" \
           launcher/hpc_batch.bash "$config_path" "$EXPERIMENT_TAG" "$OVERRIDES"

    echo "  Job submitted successfully"
    echo ""
done

echo "All jobs submitted!"