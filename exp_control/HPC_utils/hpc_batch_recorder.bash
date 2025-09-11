#!/bin/bash
##SBATCH --array=1-100             # set up the array
#SBATCH -J RLUPDATE			    # name of job
#SBATCH -A virl-grp	            # name of my sponsored account, e.g. class or research group, NOT ONID!
##SBATCH -p gpu,eecs2,tiamat,dgxh,dgx2,ampere		# name of partition or queue
##SBATCH -p tiamat,eecs2,gpu,dgx2,dgxh
#SBATCH -p eecs2
#SBATCH --time=1-23:59:59        # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
##SBATCH -N 1                   # number of nodes (default 1)
#SBATCH --gres=gpu:1            # number of GPUs to request (default 0)
#SBATCH --mem=32G               # request 10 gigabytes memory (per node, default depends on node)
#SBATCH -c 8                   # number of cores/threads per task (default 1)
#SBATCH -o ../recorder_logs/%A_%a.out		# name of output file for this submission script
#SBATCH -e ../recorder_logs/%A_%a.err		# name of error file for this submission script
# load any software environment module required for app (e.g. matlab, gcc, cuda)

#echo "Job Name:" $SLURM_JOB_NAME
#echo "Array:" $SLURM_ARRAY_TASK_COUNT
##module load cuda/10.1
#source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate isaaclab_drail

echo "Conda env: $CONDA_DEFAULT_ENV"


free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
numeric_string="${free_memory//[^0-9]/}"
int_free=$((numeric_string))
echo "Free memory:$int_free"
#if (( int_free > 8000 )); then
#    echo "8 parallel it is"
#    #bash "exp_control/run_exp.bash" $1 8 $2
#else
#    echo "two 4 parallel"
#    #bash "exp_control/run_exp.bash" $1 4 $2
#    #   bash "exp_control/run_exp.bash" $1 4 $2
#fi

#echo "call run_exp.bash"
echo $6

#task_idx
#obs_idx
#name
#exp_tag
#break_force
# hybrid_agent
#ctrl_torque
#rew_type



tasks=( "PegInsert" "GearMesh")
obs_modes=( "Local" "HistoryObs" "DMPObs")

task_idx=$1
obs_idx=$2
break_force=$3
hybrid_agent=$4
sel_adjs=$5
ckpt_path=$6
use_ft_sensor=$7
hybrid_control=$8

task_name="Isaac-Factory-TaskType-ObsType-v0"
echo "$task_name"
echo "Task Type: ${tasks[$task_idx]}"
echo "Obs Mode: ${obs_modes[$obs_idx]}"


task_name="${task_name/ObsType/${obs_modes[obs_idx]}}" 
task_name="${task_name/TaskType/${tasks[task_idx]}}"
echo "Env name: $task_name"

#use_ft_sensor=1
parallel_control=0
#hybrid_control=1

echo "Break Force: $break_force"
echo "Hybrid Agent: $hybrid_agent"
echo "Selection Adjustments: $sel_adjs"
echo "Ckpt Path: $ckpt_path"

#python -m exp_control.record_ckpts.single_task_ckpt_recorder \
python -m exp_control.record_ckpts.block_simba_ckpt_recorder \   
       --task=$task_name \
       --use_ft_sensor=$use_ft_sensor \
       --decimation=8 \
       --break_force=$break_force \
       --headless \
       --hybrid_control=$hybrid_control \
       --hybrid_agent=$hybrid_agent \
       --control_torque=0 \
       --hybrid_selection_reward="base" \
       --ckpt_record_path=$ckpt_path \
       --eval_seed=42 \
       --sel_adjs=$sel_adjs 

       #--log_smoothness_metrics=0

#bash "exp_control/run_hybrid_exp.bash" $1 $2 "$3_$SLURM_ARRAY_TASK_ID" $4 $5 $6 $7 $8
# bash exp_control/HPC_utils/hpc_batch_recorder.bash 0 0 25 0 "init_bias" ~/ckpt_trackers/pose_ctrl_FT\(OFF\)_ckpt_tracker.txt 0 0
