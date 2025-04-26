#!/bin/bash
##SBATCH --array=1-100             # set up the array
#SBATCH -J RLUPDATE			    # name of job
#SBATCH -A virl-grp	            # name of my sponsored account, e.g. class or research group, NOT ONID!
##SBATCH -p gpu,eecs2,tiamat,dgxh,dgx2,ampere		# name of partition or queue
#SBATCH -p tiamat,eecs2,gpu,dgx2,dgxh,preempt
#SBATCH --time=0-02:00:00        # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
##SBATCH -N 1                   # number of nodes (default 1)
#SBATCH --gres=gpu:1            # number of GPUs to request (default 0)
#SBATCH --mem=32G               # request 10 gigabytes memory (per node, default depends on node)
#SBATCH -c 2                   # number of cores/threads per task (default 1)
#SBATCH -o ../outs/ckpt_out/Ckpt_%A_%a.out		# name of output file for this submission script
#SBATCH -e ../outs/ckpt_out/Ckpt_%A_%a.err		# name of error file for this submission script
# load any software environment module required for app (e.g. matlab, gcc, cuda)

#echo "Job Name:" $SLURM_JOB_NAME
#echo "Array:" $SLURM_ARRAY_TASK_COUNT
##module load cuda/10.1
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env_isaaclab


python -m exp_control.record_ckpts.single_ckpt_video \
    --headless \
    --ckpt_filepath=/nfs/hpc/share/brownhun/Continuous_Force_RL/logs/Factory_Baseline_1/vid_tests_2025-04-25_17-45-31/checkpoints/agent_0.pt \
    --ckpt_step=0 \
    --task=Isaac-Factory-PegInsert-Local-v0 \
    --output_path=/nfs/hpc/share/brownhun/Continuous_Force_RL/logs/Factory_Baseline_1/vid_tests_2025-04-25_17-45-31/eval_videos/agent_0.gif