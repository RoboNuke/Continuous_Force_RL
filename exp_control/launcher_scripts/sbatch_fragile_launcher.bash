#!/bin/bash

num_agents=10
num_exp_per=1

nick_names=("PiH" "Gear" "Nut")

task_idx=0

num_forces=6
forces=(1 5 10 25 50 -1)
use_ft_sensor=1
exp_tag="jun23_force_fragile_exps"
wandb_group_prefix="Force-SuccCont"

for force_idx in $(seq 0 $((num_forces - 1)))
do
    for obs_idx in $(seq 0 1) # only doing local and history
    do 
        sbatch -J "${nick_names[$task_idx]}_$1_${samples[$sample_idx]}" -a 1-$num_exp_per exp_control/HPC_utils/hpc_batch.bash \
                $task_idx \
                $obs_idx \
                $num_agents \
                $1 \
                16 \
                "${forces[$force_idx]}" \
                $use_ft_sensor \
                $exp_tag \
                $wandb_group_prefix \
                "/nfs/stak/users/brownhun/ckpt_trackers"
    done
done