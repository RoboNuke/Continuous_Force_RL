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

use_ft_sensor=1
exp_tag="debug"
#prefix="stab_rew_bias_fixes"
num_agents=1
break_force=-1
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
num_history_samples=8
num_envs_per_agent=16
parallel_control=0
parallel_agent=0
hybrid_control=1
hybrid_agent=1
control_torques=0
hybrid_selection_reward='base'  #'simp' # 'dirs' 'delta'
force_bias_sel=1
#CUDA_LAUNCH_BLOCKING=1 
#HYDRA_FULL_ERROR=1
python -m learning.ppo_factory_trainer \
    --task=$task_name \
    --wandb_project="Continuous_Force_RL"\
    --use_ft_sensor=$use_ft_sensor \
    --exp_tag=$exp_tag \
    --wandb_group_prefix="$3_${obs_modes[$obs_idx]}_$break_force_$current_datetime" \
    --max_steps=500000 \
    --num_envs=$(($num_envs_per_agent * $num_agents)) \
    --num_agents=$num_agents \
    --exp_name="$3_${obs_modes[$obs_idx]}_$break_force_$current_datetime" \
    --exp_dir="$3_$current_datetime" \
    --no_vids \
    --decimation=8 \
    --history_sample_size=$num_history_samples \
    --break_force=$break_force \
    --headless \
    --parallel_control=$parallel_control \
    --parallel_agent=$parallel_agent \
    --hybrid_control=$hybrid_control \
    --hybrid_agent=$hybrid_agent \
    --log_ckpt_data=1 \
    --control_torques=$control_torques \
    --hybrid_selection_reward=$hybrid_selection_reward \
    --force_bias_sel=$force_bias_sel \
    --init_eval \
    --ckpt_tracker_path="/nfs/stak/users/brownhun/ckpt_trackers/$3_$current_datetime_ckpt_tracker.txt"
    #--ckpt_path="/home/hunter/good_hist_agent.pt" 
    #--hybrid_control 
    #--init_eval \
    #--no_log_wandb \


