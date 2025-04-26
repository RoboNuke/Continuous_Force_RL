xterm -e "conda run -n isaaclab \
        python -m exp_control.record_ckpts.single_ckpt_video \
            --headless \
            --ckpt_filepath=$1 \
            --ckpt_step=$2 \
            --task=$3 \
            --output_path=$4; \
        bash" & 