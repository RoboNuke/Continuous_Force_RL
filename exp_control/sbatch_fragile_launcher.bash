#!/bin/bash

num_agents=4
num_exp_per=2

nick_names=("PiH" "Gear" "Nut")

task_idx=0

num_forces=5
forces=(1 5 10 25 50 -1)
use_ft_sensor=0
exp_tag="jun8_fragile_exps"
wandb_group_prefix="No-Force-Fragile"

for force_idx in $(seq 0 $((num_forces - 1)))
do
    for obs_idx in $(seq 0 1) # only doing local and history
    do 
        sbatch -J "${nick_names[$task_idx]}_$1_${samples[$sample_idx]}" -a 1-$num_exp_per exp_control/hpc_batch.bash \
                $task_idx \
                $obs_idx \
                $num_agents \
                $1 \
                16 \
                ${forces[$force_idx]} \
                $use_ft_sensor \
                $exp_tag \
                $wandb_group_prefix
    done
done