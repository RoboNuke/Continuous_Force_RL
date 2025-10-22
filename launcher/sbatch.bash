#!/bin/bash

# New Test Launch Script for HPC
# Usage: ./launcher/sbatch.bash --configs "config1 config2" --experiment_tag "EXP_TAG" --overrides "key=value key2=value2"
# Or:    ./launcher/sbatch.bash --folder "folder_name" --experiment_tag "EXP_TAG" --overrides "key=value key2=value2"

# Default values
CONFIGS=""
FOLDER=""
EXPERIMENT_TAG=""
OVERRIDES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --configs)
            CONFIGS="$2"
            shift 2
            ;;
        --folder)
            FOLDER="$2"
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
            echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -n "$CONFIGS" && -n "$FOLDER" ]]; then
    echo "Error: --configs and --folder are mutually exclusive. Use one or the other."
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
    exit 1
fi

if [[ -z "$CONFIGS" && -z "$FOLDER" ]]; then
    echo "Error: Either --configs or --folder is required"
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
    exit 1
fi

if [[ -z "$EXPERIMENT_TAG" ]]; then
    echo "Error: --experiment_tag is required"
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--overrides \"key=value key2=value2\"]"
    exit 1
fi

echo "=== Test Launch Script ==="
echo "Configs: $CONFIGS"
echo "Folder: $FOLDER"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "Overrides: $OVERRIDES"
echo ""

# Handle folder mode or individual config mode
if [[ -n "$FOLDER" ]]; then
    # Folder mode: discover all .yaml files in the specified folder
    folder_path="configs/experiments/${FOLDER}"

    # Check if folder exists
    if [[ ! -d "$folder_path" ]]; then
        echo "Error: Folder not found: $folder_path"
        exit 1
    fi

    # Find all .yaml files in the folder
    CONFIG_ARRAY=()
    while IFS= read -r -d '' file; do
        # Extract just the filename without extension
        config_name=$(basename "$file" .yaml)
        CONFIG_ARRAY+=("${FOLDER}/${config_name}")
    done < <(find "$folder_path" -name "*.yaml" -type f -print0)

    # Check if any configs were found
    if [[ ${#CONFIG_ARRAY[@]} -eq 0 ]]; then
        echo "Error: No .yaml files found in folder: $folder_path"
        exit 1
    fi

    echo "Found ${#CONFIG_ARRAY[@]} config(s) in folder '$FOLDER': ${CONFIG_ARRAY[*]}"
    echo ""
else
    # Individual config mode: convert configs string to array
    read -ra CONFIG_ARRAY <<< "$CONFIGS"
fi

# Process each config
for config_name in "${CONFIG_ARRAY[@]}"; do
    # Construct full config path
    config_path="configs/experiments/${config_name}.yaml"

    # Check if config file exists
    if [[ ! -f "$config_path" ]]; then
        echo "Error: Config file not found: $config_path"
        continue
    fi

    # Create job name (replace slashes with underscores for valid job names)
    job_name="${EXPERIMENT_TAG}_$(echo "${config_name}" | tr '/' '_')"

    echo "Submitting job: $job_name"
    echo "  Config: $config_path"
    echo "  Experiment Tag: $EXPERIMENT_TAG"
    if [[ -n "$OVERRIDES" ]]; then
        echo "  Overrides: $OVERRIDES"
    fi

    # Submit SLURM job with dynamic output file paths (replace slashes with underscores for file names)
    output_name="$(echo "${config_name}" | tr '/' '_')"
    sbatch -J "$job_name" \
           -o "exp_logs/${EXPERIMENT_TAG}/${output_name}_%j.out" \
           -e "exp_logs/${EXPERIMENT_TAG}/${output_name}_%j.err" \
           launcher/hpc_batch.bash "$config_path" "$EXPERIMENT_TAG" "$OVERRIDES"

    echo "  Job submitted successfully"
    echo ""
done

echo "All jobs submitted!"