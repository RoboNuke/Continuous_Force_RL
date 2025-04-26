#!/bin/bash

xterm -hold -e "$(conda shell.bash hook) && conda activate isaaclab && python -m exp_control.record_ckpts.single_ckpt_video \
            --headless \
            --ckpt_filepath=$1 \
            --ckpt_step=$2 \
            --task=$3 \
            --output_path=$4; \
        bash" &

#xterm -e "echo $1 echo $2 echo $3 echo $4; bash"