#!/bin/bash

nick_names=("PiH" "Gear" "Nut")

task_idx=0
obs_idx=0

rew_types=( "base" ) # "pos_simp" "delta" "dirs" ) #"base" ) "wrench_norm"
exp_tag="hybrid_agent_debug" #"long_baseline" #"exps_AUG15" #"hybrid_baseline_llr"
project="debug" #"AUG31_Exps"
hybrid_agent=1
hybrid_control=1
ctrl_torque=0
use_ft_sensor=0
break_force="25"

sel_adjs="init_bias" #,force_add_zout,init_scale_weights,scale_zin" # "init_bias", "force_add_zout", "init_scale_weights", "scale_zin"


bash exp_control/run_hybrid_exp.bash \
    $task_idx \
    $obs_idx \
    $1 \
    $exp_tag \
    $break_force \
    $hybrid_agent \
    $ctrl_torque \
    "base" \
    $sel_adjs \
    $use_ft_sensor \
    $hybrid_control \
    $project
