#!/bin/bash
#SBATCH -J WANDB_EVAL_RUN_JOB		# job name (will be overridden by sbatch -J)
#SBATCH -A virl-grp	                # sponsored account name
#SBATCH -p tiamat,gpu,eecs2         # partition names
#SBATCH --time=0-04:00:00           # time limit: 4 hours
#SBATCH --gres=gpu:1                # number of GPUs to request
#SBATCH --mem=32G                   # request 32 gigabytes memory
#SBATCH -c 12                       # number of cores/threads per task
# Note: Output files will be set dynamically based on run_id parameter

# Script arguments
RUN_ID=$1
TAG=$2
VIDEO_FLAG=$3
EVAL_MODE=$4

echo "=== HPC WandB Eval by Run ID Batch Script Started ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "Tag: $TAG"
echo "Video Enabled: $VIDEO_FLAG"
echo "Eval Mode: $EVAL_MODE"
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
python_cmd="$python_cmd --run_id $RUN_ID"
python_cmd="$python_cmd --eval_mode $EVAL_MODE"

# Add video flag if enabled
if [[ "$VIDEO_FLAG" == "true" ]]; then
    python_cmd="$python_cmd --enable_video"
fi

# Add headless flag
python_cmd="$python_cmd --headless"

echo "Executing command:"
echo "$python_cmd"
echo ""

# Execute the evaluation
eval $python_cmd

echo ""
echo "=== HPC WandB Eval by Run ID Batch Script Completed ==="
