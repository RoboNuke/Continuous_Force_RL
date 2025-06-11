#!/bin/bash
tasks=( "PegInsert" "GearMesh")
obs_modes=( "Local" "HistoryObs" "DMPObs")

task_idx=$1
obs_idx=$2
#if [ "$exp_idx" -gt 4 ]; then
#  exp_idx=0
#fi
task_name="Isaac-Factory-TaskType-ObsType-v0"
echo "$task_name"
echo "Task Type: ${tasks[$task_idx]}"
echo "Obs Mode: ${obs_modes[$obs_idx]}"


task_name="${task_name/ObsType/${obs_modes[obs_idx]}}" 
task_name="${task_name/TaskType/${tasks[task_idx]}}"
echo "Env name: $task_name"

use_ft_sensor=0
exp_tag="debug"
prefix="debug_rates"
num_agents=1
break_force=-1
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
num_history_samples=16


#CUDA_LAUNCH_BLOCKING=1 
HYDRA_FULL_ERROR=1 python -m learning.ppo_factory_trainer \
    --task=$task_name \
    --wandb_project="Continuous_Force_RL"\
    --use_ft_sensor=$use_ft_sensor \
    --exp_tag=$exp_tag \
    --wandb_group_prefix=$prefix \
    --max_steps=5000 \
    --num_envs=$((128 * $num_agents)) \
    --num_agents=$num_agents \
    --exp_name="$4_${obs_modes[$obs_idx]}_$break_force_$current_datetime" \
    --exp_dir="$4_$current_datetime" \
    --no_vids \
    --decimation=16 \
    --history_sample_size=$num_history_samples \
    --break_force=$break_force \
    --no_log_wandb \
    --headless \
    --ckpt_path="/home/hunter/good_hist_agent.pt" 
    #--hybrid_control 


