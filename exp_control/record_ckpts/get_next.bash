# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/nfs/hpc/share/brownhun/miniforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nfs/hpc/share/brownhun/miniforge/etc/profile.d/conda.sh" ]; then
        . "/nfs/hpc/share/brownhun/miniforge/etc/profile.d/conda.sh"
    else
        export PATH="/nfs/hpc/share/brownhun/miniforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/nfs/hpc/share/brownhun/miniforge/etc/profile.d/mamba.sh" ]; then
    . "/nfs/hpc/share/brownhun/miniforge/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<



python /nfs/stak/users/brownhun/hpc-share/Continuous_Force_RL/exp_control/record_ckpts/get_next_ckpt.py $1