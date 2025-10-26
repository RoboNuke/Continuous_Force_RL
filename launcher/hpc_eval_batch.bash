#!/bin/bash
#SBATCH -J EVAL_JOB		            # job name (will be overridden by sbatch -J)
#SBATCH -A virl-grp	                # sponsored account name
#SBATCH -p tiamat,gpu,eecs2,dgx2,dgxh               # partition names
#SBATCH --time=0-02:00:00           # time limit: 2 hours (evals are shorter than training)
#SBATCH --gres=gpu:1                # number of GPUs to request
#SBATCH --mem=32G                   # request 32 gigabytes memory
#SBATCH -c 12                       # number of cores/threads per task
# Note: Output files will be set dynamically based on EXPERIMENT_TAG parameter

# Script arguments
CONFIG_PATH=$1
EXPERIMENT_TAG=$2
TRACKER_PATH=$3
VIDEO_FLAG=$4

echo "=== HPC Eval Batch Script Started ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Config Path: $CONFIG_PATH"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "Tracker Path: $TRACKER_PATH"
echo "Video Enabled: $VIDEO_FLAG"
echo ""

# Create log directory if it doesn't exist
mkdir -p "../exp_logs/$EXPERIMENT_TAG"

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
python_cmd="python eval/block_eval_script.py"
python_cmd="$python_cmd --config $CONFIG_PATH"
python_cmd="$python_cmd --ckpt_tracker_path $TRACKER_PATH"
python_cmd="$python_cmd --num_envs_per_agent 100"
python_cmd="$python_cmd --override primary.total_agents=2"
python_cmd="$python_cmd --upload_ckpt"

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
echo "=== HPC Eval Batch Script Completed ==="
