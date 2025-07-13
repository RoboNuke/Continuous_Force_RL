#!/bin/bash

num_agents=10
num_exp_per=1

nick_names=("PiH" "Gear" "Nut")

task_idx=0
obs_idx=0

num_forces=2
rew_types=("simp" "dirs" "delta")
exp_tag="baseline_hybrid"

for rew_type in "${rew_types[@]}";
do
    sbatch -J "${nick_names[$task_idx]}_$1_${samples[$sample_idx]}" -a 1-$num_exp_per exp_control/HPC_utils/hpc_batch_hybrid.bash \
           $task_idx \
           $obs_idx \
	   $1 \
	   $exp_tag \
           -1 \
           $hybrid_agent \
	   $ctrl_torque \
	   $hybrid_reward_type 
done
