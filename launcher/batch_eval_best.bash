#!/bin/bash
set -e  # Stop immediately on any error

# =============================================================================
# Batch Evaluation Script for Best Policies
# Runs wandb_eval.py with --eval_best_policies for multiple tags and eval modes
# =============================================================================

usage() {
    echo "Usage: $0 --tags <tag1,tag2,...> --modes <mode1,mode2,...> [--project <project_name>]"
    echo ""
    echo "Arguments:"
    echo "  --tags     Comma-separated list of tags to evaluate"
    echo "  --modes    Comma-separated list of eval modes (noise, rotation, gain, dynamics, yaw, trajectory)"
    echo "  --project  (Optional) WandB project name (default: uses wandb_eval.py default)"
    echo ""
    echo "Example:"
    echo "  $0 --tags \"exp_v1,exp_v2\" --modes \"noise,rotation\" --project \"Peg_in_Hole\""
    exit 1
}

# Parse command line arguments
TAGS=""
MODES=""
PROJECT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TAGS" ]]; then
    echo "ERROR: --tags is required"
    usage
fi

if [[ -z "$MODES" ]]; then
    echo "ERROR: --modes is required"
    usage
fi

# Convert comma-separated strings to arrays
IFS=',' read -ra TAG_ARRAY <<< "$TAGS"
IFS=',' read -ra MODE_ARRAY <<< "$MODES"

# Create single temp log file for all python output
LOG_FILE=$(mktemp /tmp/batch_eval_best_XXXXXX.log)

# Print header
echo "============================================================================="
echo "Batch Evaluation - Best Policies"
echo "============================================================================="
echo "Tags: ${TAG_ARRAY[*]}"
echo "Modes: ${MODE_ARRAY[*]}"
echo "Project: ${PROJECT:-<default>}"
echo "Log file: $LOG_FILE"
echo "============================================================================="
echo ""

# Count total evaluations
TOTAL_EVALS=$((${#TAG_ARRAY[@]} * ${#MODE_ARRAY[@]}))
CURRENT_EVAL=0

# Nested loops: outer=modes, inner=tags
for mode in "${MODE_ARRAY[@]}"; do
    for tag in "${TAG_ARRAY[@]}"; do
        CURRENT_EVAL=$((CURRENT_EVAL + 1))
        echo "[$CURRENT_EVAL/$TOTAL_EVALS] Evaluating tag=$tag, mode=$mode ..."

        # Build python command
        python_cmd="python -m eval.wandb_eval"
        python_cmd="$python_cmd --tag $tag"
        python_cmd="$python_cmd --eval_mode $mode"
        python_cmd="$python_cmd --eval_best_policies"
        python_cmd="$python_cmd --headless"

        # Add project if specified
        if [[ -n "$PROJECT" ]]; then
            python_cmd="$python_cmd --project $PROJECT"
        fi

        # Log the command being run
        echo "" >> "$LOG_FILE"
        echo "=============================================================================" >> "$LOG_FILE"
        echo "[$CURRENT_EVAL/$TOTAL_EVALS] tag=$tag, mode=$mode" >> "$LOG_FILE"
        echo "Command: $python_cmd" >> "$LOG_FILE"
        echo "=============================================================================" >> "$LOG_FILE"

        # Run python command, append output to log file
        if ! eval $python_cmd >> "$LOG_FILE" 2>&1; then
            echo ""
            echo "ERROR: Evaluation failed for tag=$tag, mode=$mode"
            echo "Review log file: $LOG_FILE"
            echo ""
            echo "Last 50 lines of output:"
            echo "-----------------------------------------------------------------------------"
            tail -50 "$LOG_FILE"
            exit 1
        fi

        echo "[$CURRENT_EVAL/$TOTAL_EVALS] Completed tag=$tag, mode=$mode"
        echo ""
    done
done

echo "============================================================================="
echo "All evaluations complete! ($TOTAL_EVALS/$TOTAL_EVALS)"
echo "Full log available at: $LOG_FILE"
echo "============================================================================="
