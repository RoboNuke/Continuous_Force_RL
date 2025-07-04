#!/bin/bash

num_agents=4
num_exp_per=2

nick_names=("PiH" "Gear" "Nut")

task_idx=0
obs_idx=1

break_force="-1.0"
samples=(1 4 8 12 16)
use_ft_sensor=0
exp_tag="jun8_noforce_tween"
wandb_group_prefix="No-Force_Tween"

for sample_idx in $(seq 0 $((5 - 1)))
do
    sbatch -J "${nick_names[$task_idx]}_$1_${samples[$sample_idx]}" -a 1-$num_exp_per exp_control/hpc_batch.bash \
            $task_idx \
            $obs_idx \
            $num_agents \
            $1 \
            "${samples[$sample_idx]}" \
            $break_force \
            $use_ft_sensor \
            $exp_tag \
            $wandb_group_prefix
done