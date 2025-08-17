#!/bin/bash
tasks=( "PegInsert" "GearMesh")
obs_modes=( "Local" "HistoryObs" "DMPObs")

task_idx=$1
obs_idx=$2
# exp name
exp_tag=$4
break_force=$5
hybrid_agent=$6
control_torques=$7
hybrid_selection_reward=$8 #'delta'  #'simp' # 'dirs' 'delta'
sel_adjs=$9
use_ft_sensor=${10}
hybrid_control=${11}

echo "Hybrid Control:${12}    $hybrid_control"
task_name="Isaac-Factory-TaskType-ObsType-v0"
echo "$task_name"
echo "Task Type: ${tasks[$task_idx]}"
echo "Obs Mode: ${obs_modes[$obs_idx]}"


task_name="${task_name/ObsType/${obs_modes[obs_idx]}}" 
task_name="${task_name/TaskType/${tasks[task_idx]}}"
echo "Env name: $task_name"

#use_ft_sensor=1
num_agents=5
current_datetime=$(date +"%Y-%m-%d_%H:%M:%S")
num_history_samples=8
num_envs_per_agent=256
#hybrid_control=1


echo "Hybrid Agent:$hybrid_agent"
echo "Reward: $hybrid_selection_reward"
echo "Ctrl Torques: $control_torques"
#CUDA_LAUNCH_BLOCKING=1 
#HYDRA_FULL_ERROR=1

ckpt_path="/nfs/stak/users/brownhun/ckpt_trackers/$3_ckpt_tracker.txt"


sbatch -J "$3_recorder" exp_control/HPC_utils/hpc_batch_recorder.bash \
       $task_idx \
       $obs_idx \
       $break_force \
       $hybrid_agent \
       $sel_adjs \
       $ckpt_path \
       $use_ft_sensor \
       $hybrid_control \
       $hybrid_agent \
       

#sbatch -J "$exp_tag_recorder_$current_datetime" exp_control/HPC_utils/hpc_batch_recorder.bash \
#       $task_idx \
#       $obs_idx \
#       $break_force \
#       $hybrid_agent \
#       $ctrl_torque \
#       $ckpt_path \
#       $rew_type


python -m learning.ppo_factory_trainer \
    --task=$task_name \
    --wandb_project="debug" \
    --use_ft_sensor=$use_ft_sensor \
    --exp_tag=$exp_tag \
    --wandb_group_prefix="$3_$4_${11}_${12}_$5" \
    --max_steps=100000000 \
    --num_envs=$(($num_envs_per_agent * $num_agents)) \
    --num_agents=$num_agents \
    --exp_name="$3_${obs_modes[$obs_idx]}_$8_$5_$current_datetime" \
    --exp_dir="$3_$current_datetime_$8" \
    --no_vids \
    --decimation=8 \
    --history_sample_size=$num_history_samples \
    --break_force=$break_force \
    --hybrid_control=$hybrid_control \
    --hybrid_agent=$hybrid_agent \
    --log_ckpt_data=1 \
    --control_torques=$control_torques \
    --hybrid_selection_reward=$hybrid_selection_reward \
    --ckpt_tracker_path=$ckpt_path \
    --init_eval \
    --sel_adjs=$sel_adjs \
    --headless \
    --easy_mode 




###"Continuous_Force_RL" \
