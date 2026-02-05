#!/bin/bash
#SBATCH -J WANDB_EVAL_JOB		    # job name (will be overridden by sbatch -J)
#SBATCH -A virl-grp	                # sponsored account name
#SBATCH -p tiamat,gpu,dgx2,dgxh,eecs2         # partition names
#SBATCH --time=0-02:59:59           # time limit: 4 hours
#SBATCH --gres=gpu:1                # number of GPUs to request
#SBATCH --mem=32G                   # request 32 gigabytes memory
#SBATCH -c 12                       # number of cores/threads per task
# Note: Output files will be set dynamically based on tag parameter

# Script arguments
TAG=$1
VIDEO_FLAG=$2
EVAL_MODE=$3
PROJECT=$4
CHECKPOINT_RANGE=$5
RESUME=$6

echo "=== HPC WandB Eval Batch Script Started ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Tag: $TAG"
echo "Video Enabled: $VIDEO_FLAG"
echo "Eval Mode: $EVAL_MODE"
echo "Project: $(if [[ "$PROJECT" == "NONE" ]]; then echo "<default>"; else echo "$PROJECT"; fi)"
echo "Checkpoint Range: $(if [[ "$CHECKPOINT_RANGE" == "NONE" ]]; then echo "<all>"; else echo "$CHECKPOINT_RANGE"; fi)"
echo "Resume: ${RESUME:-false}"
echo ""

# Create log directory if it doesn't exist
mkdir -p exp_logs/wandb_eval

# Load environment
echo "Loading conda environment..."
source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate isaaclab_drail

echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Set Vulkan ICD for headless rendering
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
echo "Set VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
echo ""

# Build the python command
python_cmd="python -m eval.wandb_eval"
python_cmd="$python_cmd --tag $TAG"
python_cmd="$python_cmd --eval_mode $EVAL_MODE"

# Add video flag if enabled
if [[ "$VIDEO_FLAG" == "true" ]]; then
    python_cmd="$python_cmd --enable_video"
fi

# Add project if specified (check for "NONE" placeholder)
if [[ -n "$PROJECT" && "$PROJECT" != "NONE" ]]; then
    python_cmd="$python_cmd --project $PROJECT"
fi

# Add checkpoint_range if specified (check for "NONE" placeholder)
if [[ -n "$CHECKPOINT_RANGE" && "$CHECKPOINT_RANGE" != "NONE" ]]; then
    python_cmd="$python_cmd --checkpoint_range $CHECKPOINT_RANGE"
fi

# Add resume flag if enabled
if [[ "$RESUME" == "true" ]]; then
    python_cmd="$python_cmd --resume"
fi

# Add headless flag
python_cmd="$python_cmd --headless"

echo "Executing command:"
echo "$python_cmd"
echo ""

# Execute the evaluation
eval $python_cmd

echo ""
echo "=== HPC WandB Eval Batch Script Completed ==="
