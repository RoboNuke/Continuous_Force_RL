#!/bin/bash
tasks=( "PegInsert" "GearMesh")
obs_modes=( "Local" "HistoryObs" "DMPObs")

task_idx=$1
obs_idx=$2
num_agents=$3
# 4 is exp name
num_history_samples=$5
break_force=$6
use_ft_sensor=$7
exp_tag=$8
prefix=$9
ckpt_filepath=${10}

#if [ "$exp_idx" -gt 4 ]; then
#  exp_idx=0
#fi
task_name="Isaac-Factory-TaskType-ObsType-v0"
echo "Task Type: ${tasks[$task_idx]}"
echo "Obs Mode: ${obs_modes[$obs_idx]}"


task_name="${task_name/ObsType/${obs_modes[obs_idx]}}" 
task_name="${task_name/TaskType/${tasks[task_idx]}}"
echo "Env name: $task_name"

echo "Num Agents: $num_agents"
echo "Break force: $break_force"

exp_name="$4_${tasks[task_idx]}_${obs_modes[obs_idx]}"
echo $task_name
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Ckpt Filepath: $ckpt_filepath"
ckpt_record_path="$ckpt_filepath/$exp_name-$current_datetime.txt"
echo $ckpt_record_path
#touch $ckpt_record_path
#touch "$ckpt_filepath/$exp_name-$current_datetime-recorder_output.log"
python -m exp_control.record_ckpts.single_task_ckpt_recorder \
    --headless \
    --task=$task_name \
    --decimation=16 \
    --ckpt_record_path=$ckpt_record_path \
    --use_ft_sensor=$use_ft_sensor \
    > "$ckpt_filepath/$exp_name-$current_datetime-recorder_output.log" 2>&1 &

python -m learning.ppo_factory_trainer \
    --headless \
    --task=$task_name \
    --wandb_project="Continuous_Force_RL" \
    --use_ft_sensor=$use_ft_sensor \
    --exp_tag=$exp_tag \
    --wandb_group_prefix=$prefix \
    --max_steps=50000000 \
    --num_envs=$((4 * $num_agents)) \
    --num_agents=$num_agents \
    --exp_name="$4_${obs_modes[$obs_idx]}_$break_force_$current_datetime" \
    --exp_dir="$4_$current_datetime" \
    --no_vids \
    --decimation=16 \
    --history_sample_size=$num_history_samples \
    --break_force=$break_force \
    --log_ckpt_data=1 \
    --ckpt_tracker_path=$ckpt_record_path



# "DMP_Observation_Testing" \
#python -m learning.single_agent_train --task TB2-Factor-PiH-v0 --exp_name basic_PiH_baseline --headless --max_steps 50000000 --no_vids --num_agents 5 --num_envs 1280 --wandb_tags multi_agent_tests basic_obs
## "${names[$exp_idx]}"  \
#    --wandb_project="DMP_Observation_Testing" \
#    --wandb_tags="rl_update","no_curriculum","pih_apr11" \
    #--seed=1 \
    #--log_smoothness_metrics \