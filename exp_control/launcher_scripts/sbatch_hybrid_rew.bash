#!/bin/bash

nick_names=("PiH" "Gear" "Nut")

task_idx=0
obs_idx=0

rew_types=( "base" ) #"simp" "delta" "dirs" "base" )
exp_tag=$1 #"hybrid_baseline_llr"
hybrid_agent=1
ctrl_torque=0

for rew_type in "${rew_types[@]}";
do
    sbatch -J "${nick_names[$task_idx]}_$1_${rew_type}" exp_control/HPC_utils/hpc_batch_hybrid.bash \
           $task_idx \
           $obs_idx \
	   $1 \
	   $exp_tag \
           -1 \
           $hybrid_agent \
	   $ctrl_torque \
	   $rew_type
done
