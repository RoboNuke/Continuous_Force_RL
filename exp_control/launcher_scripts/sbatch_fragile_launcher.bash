#!/bin/bash

num_agents=5
num_exp_per=1

nick_names=("PiH" "Gear" "Nut")

task_idx=0

num_forces=5
forces=(5 10 25 50 -1)



#use_ft_sensor=1
exp_tag=$1 #"jun23_force_fragile_exps"
#wandb_group_prefix="Force-SuccCont"

rew_types=( "base" ) # "pos_simp" "delta" "dirs" ) #"base" )
rew_type="base"
exp_tag=$1 #"hybrid_baseline_llr"
hybrid_agent=0
ctrl_torque=0
#hybrid_control=0
bias_sel=1
force_bias_sel=0

ft_opts=(1 0)
for ft_idx in $(seq 0 1) # only doing local and history
do
    use_ft_sensor="${ft_opts[$ft_idx]}"
    for force_idx in $(seq 0 $((num_forces - 1)))
    do
	force_thresh="${forces[$force_idx]}"
	obs_idx=1
	for hybrid_control in $(seq 0 1) # only doing local ##and history
	do
            sbatch -J "${nick_names[$task_idx]}H8_$1_${use_ft_sensor}_${force_thresh}_${hybrid_control}" exp_control/HPC_utils/hpc_batch_hybrid.bash \
		   $task_idx \
		   $obs_idx \
		   "$1" \
		   $exp_tag \
		   $force_thresh \
		   $hybrid_agent \
		   $ctrl_torque \
		   $rew_type \
		   $bias_sel \
		   $force_bias_sel \
		   $use_ft_sensor \
		   $hybrid_control
	done
    done
done
