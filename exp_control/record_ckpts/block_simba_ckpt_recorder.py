import argparse
import sys
try:
    from isaaclab.app import AppLauncher
except:
    from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record checkpoints with Block SimBa agent.")
# exp
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Local-v0", help="Name of the task.")
parser.add_argument("--ckpt_record_path", type=str, default="/nfs/stak/users/brownhun/ckpt_tracker.txt", help="Path the ckpt recording data")
parser.add_argument("--easy_mode", action="store_true", default=False, help="Limits the intialization to simplify problem")
parser.add_argument("--use_obs_noise", action="store_true", default=False, help="Adds Gaussian noise specificed by env cfg to observations")
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment")
parser.add_argument("--decimation", type=int, default=8, help="How many simulation steps between policy observations")
parser.add_argument("--policy_hz", type=int, default=15, help="Rate in hz that the policy should get new observations")
parser.add_argument("--break_force", type=str, default="-1.0", help="Force at which the held object breaks (peg, gear or nut)")
parser.add_argument("--use_ft_sensor", type=int, default=0, help="Adds force sensor data to the observation space")
parser.add_argument("--num_agents", type=int, default=1, help="Number of agents in Block SimBa setup")
parser.add_argument("--history_sample_size", type=int, default=8, help="How many samples to keep from sim steps")
parser.add_argument("--exp_tag", type=str, default="eval", help="Tag to apply to exp in wandb")
parser.add_argument("--wandb_group_prefix", type=str, default="", help="Prefix of wandb group to add this to")

# controller/agent params
parser.add_argument("--parallel_control", type=int, default=0, help="Switches to parallel force position control as the action space")
parser.add_argument("--parallel_agent", type=int, default=0, help="Switches to parallel force position agent using calculated log probs based on controller")
parser.add_argument("--hybrid_control", type=int, default=0, help="Switches to hybrid force/position control as the action space")
parser.add_argument("--hybrid_agent", type=int, default=0, help="Switches to hybrid force/position agent using calculated log probs based on controller")
parser.add_argument("--hybrid_selection_reward", type=str, default="simp", help="Allows different rewards on the force/position selection: options are [simp, dirs, delta]")
parser.add_argument("--control_torques", type=int, default=0, help="Allows hybrid control to effect torques not just forces")
parser.add_argument("--impedance_control", type=int, default=0, help="Switches to impedance control as the action space")
parser.add_argument("--impedance_agent", type=int, default=0, help="Switches to impedance agent using calculated log probs based on controller")
parser.add_argument("--control_damping", type=int, default=0, help="Allows Impedance Controller Policy to predict the damping constants")
parser.add_argument("--log_smoothness_metrics", action="store_true", default=False, help="Log the sum squared velocity, jerk and force metrics")
parser.add_argument("--sel_adjs", type=str, default="none", help="Adds different selection biases")

# learning params
parser.add_argument("--lr_scheduler_type", type=str, default="cfg", help="Sets the learning rate scheduler type")

# wandb
parser.add_argument("--wandb_entity", type=str, default="hur", help="Name of wandb entity")
parser.add_argument("--wandb_project", type=str, default="Continuous_Force_RL", help="Wandb project to save logging to")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

args_cli.video = True
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import torchvision.transforms.functional as F
from PIL import Image
import gymnasium as gym
import random
from datetime import datetime
import numpy as np
from collections import defaultdict

import skrl
from packaging import version

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from memories.multi_random import MultiRandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
import learning.launch_utils as lUtils

try:
    import isaaclab.sim as sim_utils
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab.sensors import TiledCameraCfg, ImuCfg
except:
    from omni.isaac.lab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.sensors import TiledCameraCfg, ImuCfg
import time
import envs.factory

import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from filelock import FileLock
import wandb

wandb.login(
    key='a593b534585893ad93cf62243228a866c9053247',
    force=True
)

def getNextCkpt(
        task, 
        ckpt_tracker_path="/nfs/stak/users/brownhun/ckpt_tracker.txt"
    ):
    lock = FileLock(ckpt_tracker_path + ".lock")
    with lock:
        try:
            with open(ckpt_tracker_path, 'r') as file:
                lines = file.readlines()

            next_line = None
            with open(ckpt_tracker_path, 'w+') as file:
                for line in reversed(lines): # get last example
                    if task in line:
                        next_line = line
                        break
                for line in lines: # rewrite file
                    if line == next_line:
                        continue
                    else:
                        file.write(line)
            if len(next_line.split()) == 5:
                ckpt_path, task, gif_path, wandb_project, run_id = next_line.split()
            else: # len(next_line.split()) == 7:
                a, b, e, task, c, d,f, wandb_project, run_id = next_line.split()
                ckpt_path = a + " " + b + " " + e
                gif_path = c + " " + d + " " + f
                
            return ckpt_path, gif_path, wandb_project, run_id
        except Exception as e:
            print("Exception:", e)
            return False, False, False, False

def save_tensor_as_gif(tensor_list, filename, tot_rew, vals, succ_step, engaged_mask, force_control, duration=100, loop=0):
    """
    Saves a list of PyTorch tensors as a GIF image with proper visualization for Block SimBa.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to be saved as frames.
        filename (str): Output filename for the GIF.
        tot_rew (tensor): Return for each environment
        vals (any): List to write onto corner of images
        succ_step (tensor): Step at which each environment succeeded
        engaged_mask (tensor): Mask indicating engagement for each environment at each step
        force_control (tensor): Force control information for X,Y,Z dimensions
        duration (int, optional): Duration of each frame in milliseconds. Defaults to 100.
        loop (int, optional): Number of times the GIF should loop. 0 means infinite loop. Defaults to 0.
    """
    images = []
    ordered_rew_idxs = torch.squeeze(torch.argsort(tot_rew.T))
    n = ordered_rew_idxs.size()[0]
    
    # Select 16 environments: 4 worst, 4 middle, 4 best (repeated twice for 16 total)
    worst_4 = ordered_rew_idxs[:4]
    mid_4 = ordered_rew_idxs[n//2-2:n//2+2]
    best_4 = ordered_rew_idxs[-4:]
    
    img_idxs_display = torch.cat((worst_4, mid_4, best_4, best_4), dim=-1)  # 16 total
    
    vals = vals[img_idxs_display, :]
    succ_step = succ_step[img_idxs_display]
    engaged_mask = engaged_mask[img_idxs_display,:]
    force_control = force_control[img_idxs_display,:,:]

    # Create tensor for 16 images in 4x4 grid
    new_tensor_list = torch.zeros((engaged_mask.size()[1], 180 * 4, 240 * 4, 3))
    for i, idx in enumerate(img_idxs_display):
        y = i // 4
        x = i % 4
        new_tensor_list[:, y*180:(y+1)*180, x*240:(x+1)*240, :] = tensor_list[:, idx*180:(idx+1)*180, 0:240, :]

    tensor_list = new_tensor_list
    tensor_list = tensor_list.permute(0, 3, 1, 2)
    tensor_list = 1.0 - tensor_list
    
    for i in range(engaged_mask.size()[1]):
        # Ensure the tensor is in CPU and convert it to a PIL Image
        tensor = tensor_list[i,:,:,:] 
        img = F.to_pil_image(tensor.to("cpu"))
        
        for img_idx, val in enumerate(vals[:,i]):
            x = img_idx % 4
            y = img_idx // 4
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("DejaVuSans.ttf",20)
            
            # Gray out terminated environments
            if img_idx < len(succ_step) and i >= succ_step[img_idx] and succ_step[img_idx] < engaged_mask.size()[1]:
                # Add gray overlay for failed environments
                overlay = Image.new('RGBA', (240, 180), (128, 128, 128, 128))
                img.paste(overlay, (x*240, y*180), overlay)
            
            # Add borders based on success and engagement
            border_color = None
            if i >= succ_step[img_idx]:  # Success
                border_color = "green"
            elif engaged_mask[img_idx, i]:  # Engaged
                border_color = "orange"
            
            if border_color:
                draw.rectangle( 
                    ((x*240, y*180), ((x+1)*240, (y+1)*180)), 
                    outline=border_color, 
                    width=3
                )

            # Add text information
            draw.text((x * 240, y*180+160), f"Value Est={round(val.item(),2)}", (0,255,0), font=font)
            
            # Force control indicators (X,Y,Z)
            ctrl_x = force_control[img_idx, i, 0] > 0.5
            ctrl_y = force_control[img_idx, i, 1] > 0.5
            ctrl_z = force_control[img_idx, i, 2] > 0.5
            draw.text((x * 240+20, y*180+135), f"X", (0 if ctrl_x else 255, 255 if ctrl_x else 0, 0), font=font)
            draw.text((x * 240+50, y*180+135), f"Y", (0 if ctrl_y else 255, 255 if ctrl_y else 0, 0), font=font)
            draw.text((x * 240+80, y*180+135), f"Z", (0 if ctrl_z else 255, 255 if ctrl_z else 0, 0), font=font)

        images.append(img)

    # Save the list of PIL Images as a GIF
    folder_path = os.path.dirname(filename)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=loop)

agent_cfg_entry_point = f"SimBaNet_ppo_cfg_entry_point"

class Img2InfoWrapperclass(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        infos['img'] = self.unwrapped.scene['tiled_camera'].data.output['rgb']
        return observations, rewards, terminateds, truncateds, infos 
    
    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations, info = super().reset(**kwargs)
        info['img'] = self.unwrapped.scene['tiled_camera'].data.output['rgb']
        return observations, info

def calculate_evaluation_metrics(infos_history, rewards_history, success_history, engagement_history):
    """
    Calculate evaluation statistics similar to MultiWandbLoggerPPO.
    
    Args:
        infos_history: List of info dictionaries from each step
        rewards_history: List of reward tensors from each step  
        success_history: List of success masks from each step
        engagement_history: List of engagement masks from each step
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    num_envs = len(rewards_history[0]) if rewards_history else 0
    
    if num_envs == 0:
        return metrics
    
    # Calculate total returns
    total_returns = torch.zeros(num_envs, device=rewards_history[0].device)
    for rewards in rewards_history:
        total_returns += rewards.squeeze()
    
    # Success and engagement rates
    final_success = success_history[-1] if success_history else torch.zeros(num_envs, dtype=torch.bool)
    any_engagement = torch.zeros(num_envs, dtype=torch.bool, device=final_success.device)
    for engagement in engagement_history:
        any_engagement |= engagement.squeeze()
    
    # Failure rate is 1 - success rate
    success_rate = final_success.float().mean().item()
    failure_rate = 1.0 - success_rate
    engagement_rate = any_engagement.float().mean().item()
    
    metrics["Eval / Success Rate"] = success_rate
    metrics["Eval / Failure Rate"] = failure_rate 
    metrics["Eval / Engagement Rate"] = engagement_rate
    metrics["Eval / Total Returns (Avg)"] = total_returns.mean().item()
    metrics["Eval / Total Returns (Std)"] = total_returns.std().item()
    
    # Component rewards
    if infos_history:
        # Aggregate component rewards over all steps
        component_sums = defaultdict(lambda: torch.zeros(num_envs, device=total_returns.device))
        step_count = len(infos_history)
        
        for step_info in infos_history:
            for key, value in step_info.items():
                if 'Reward /' in key:
                    if hasattr(value, 'squeeze'):
                        component_sums[key] += value.squeeze()
                    else:
                        component_sums[key] += torch.tensor(value, device=total_returns.device)
        
        # Average component rewards per step
        for key, total_reward in component_sums.items():
            avg_reward = total_reward / step_count
            clean_key = key.replace('Reward /', '').strip()
            metrics[f"Eval / {clean_key} (Avg)"] = avg_reward.mean().item()
    
    # Smoothness metrics
    if infos_history and 'smoothness' in infos_history[-1]:
        smoothness_info = infos_history[-1]['smoothness']
        for key, value in smoothness_info.items():
            if hasattr(value, 'squeeze'):
                smoothness_val = value.squeeze()
            else:
                smoothness_val = torch.tensor(value, device=total_returns.device)
            
            # Remove any prefixes and add Eval prefix
            clean_key = key.replace('Smoothness /', '').strip()
            metrics[f"Eval / {clean_key}"] = smoothness_val.mean().item()
    
    return metrics

if args_cli.seed == -1:
    args_cli.seed = random.randint(0, 10000)
seed = args_cli.seed
print("Seed:", seed)
set_seed(seed)
print("Args:", args_cli)

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: dict
):    
    
    env_cfg.filter_collisions = False
    task = args_cli.task
    print("task:", task)
    print("initialize Block SimBa ckpt video recorder")
    
    # Use launch_utils to configure environment and agent settings
    num_envs = 100
    args_cli.num_envs = num_envs  # Set for consistency with launch_utils
    
    # Set break force using launch_utils pattern
    if len(args_cli.break_force.split(",")) > 1:
        forces = [float(val) for val in args_cli.break_force.split(",")]
        env_cfg.break_force = forces
    else:
        env_cfg.break_force = float(args_cli.break_force)
    agent_cfg['agent']['break_force'] = env_cfg.break_force
    
    # Use launch_utils configuration functions
    lUtils.set_use_force(
        env_cfg, agent_cfg, args_cli,
        args_cli.parallel_control==1 or args_cli.hybrid_control==1 or args_cli.use_ft_sensor==1
    )
    
    lUtils.set_reward_shaping(env_cfg, agent_cfg)
    lUtils.set_easy_mode(env_cfg, agent_cfg, args_cli.easy_mode)
    lUtils.set_time_params(env_cfg, args_cli)
    lUtils.set_use_obs_noise(env_cfg, args_cli.use_obs_noise)
    
    max_rollout_steps = int((1/env_cfg.sim.dt) / env_cfg.decimation * env_cfg.episode_length_s)
    print("[INFO]: Maximum Steps ", max_rollout_steps)
    
    lUtils.set_controller_tagging(env_cfg, agent_cfg, args_cli, max_rollout_steps)
    lUtils.set_selection_adjustments(env_cfg, agent_cfg, args_cli)
    lUtils.set_wandb_env_data(env_cfg, agent_cfg, args_cli)
    lUtils.set_learn_rate_scheduler(env_cfg, agent_cfg, args_cli)
    lUtils.set_individual_agent_log_paths(env_cfg, agent_cfg, args_cli)

    # Setup camera configuration for recording
    env_cfg.scene.tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.35), 
            rot=(0.6123724, 0.3535534, 0.3535534, 0.6123724,), 
            convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=0.05, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=240,
        height=180,
        debug_vis=False,
    )

    print("[INFO]: Creating Env...")
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    print("[INFO]: Env Built!")
    
    # Add image capture wrapper
    env = Img2InfoWrapperclass(env)
    
    # Use launch_utils for controller wrapper
    lUtils.set_controller_wrapper(env_cfg, agent_cfg, args_cli, env)
    
    if args_cli.log_smoothness_metrics:
        print("\n\n[INFO] Recording Smoothness Metrics in info.\n\n")
        env = SmoothnessObservationWrapper(env)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    
    # Use launch_utils for preprocessors
    lUtils.set_preprocessors(
        env_cfg, agent_cfg, env, 
        state=agent_cfg['agent']['state_preprocessor'], 
        value=agent_cfg['agent']['value_preprocessor']
    )
    
    print("[INFO]: Observation Space Size:", env.cfg.observation_space)
    print("[INFO]: State Space Size:", env.cfg.state_space)
    print("[INFO]: Action Space Size:", env.action_space)
    
    device = env.device
    images = torch.zeros((max_rollout_steps, num_envs*180, 240, 3), device=env.device)

    # Use launch_utils for model creation
    memory = MultiRandomMemory(
        memory_size=agent_cfg['agent']["rollouts"], 
        num_envs=env.num_envs, 
        device=device,
        replacement=True
    )
    
    models = lUtils.set_block_models(env_cfg, agent_cfg, args_cli, env)
    agent = lUtils.set_block_agent(env_cfg, agent_cfg, args_cli, models, memory, env)

    print("configured...")
    while True:
        print("Ckpt Record Path:", args_cli.ckpt_record_path)
        ckpt_path, gif_path, wandb_project, run_id = getNextCkpt(args_cli.task, ckpt_tracker_path=args_cli.ckpt_record_path)
        
        if ckpt_path == False:
            print("Waiting 2 minutes")
            for i in tqdm.tqdm(range(120), file=sys.stdout):
                time.sleep(1) 
            continue
        
        print(f"Filming {ckpt_path}")
        
        with torch.no_grad():
            # Load agent
            try:
                agent.load(ckpt_path)
            except Exception as e:
                print(f"Initial load failed: {e}")
                print("Attempting to switch FT sensor state and reload...")
                
                # Toggle force sensor setting
                original_use_ft = args_cli.use_ft_sensor
                args_cli.use_ft_sensor = 1 - args_cli.use_ft_sensor
                
                # Reconfigure environment and agent with new force sensor setting
                lUtils.set_use_force(
                    env_cfg, agent_cfg, args_cli,
                    args_cli.parallel_control==1 or args_cli.hybrid_control==1 or args_cli.use_ft_sensor==1
                )
                
                # Recreate environment with new configuration
                env.unwrapped._configure_gym_env_spaces()
                
                # Recreate agent with new model architecture
                models = lUtils.set_block_models(env_cfg, agent_cfg, args_cli, env)
                agent = lUtils.set_block_agent(env_cfg, agent_cfg, args_cli, models, memory, env)
                
                try:
                    agent.load(ckpt_path)
                    print(f"Successfully loaded checkpoint with use_ft_sensor={args_cli.use_ft_sensor}")
                except Exception as e2:
                    print(f"Failed to load checkpoint after toggling FT sensor: {e2}")
                    args_cli.use_ft_sensor = original_use_ft  # Restore original setting
                    continue
            
            print("agent loaded")
            # Reset env
            agent.set_running_mode("eval")
            states, infos = env.unwrapped.reset()
            states, infos = env.reset()
            states = env.unwrapped._get_observations()
            print("\tPolicy Size:", states['policy'].size())
            print("\tCritic Size:", states['critic'].size())
            states = torch.cat((states['policy'], states['critic']), dim=1)
            
            env.unwrapped.evaluating = True
            alive_mask = torch.ones(size=(states.shape[0], 1), device=states.device, dtype=bool)
            vals = []
            success_mask = torch.zeros(size=(states.shape[0], 1), device=states.device, dtype=bool)
            succ_step = torch.ones(size=(states.shape[0],), device=states.device, dtype=torch.int32) * max_rollout_steps
            tot_rew = torch.zeros(size=(states.shape[0], 1), device=states.device, dtype=torch.float32)
            engaged_mask = torch.zeros(size=(states.shape[0], max_rollout_steps), dtype=bool, device=device)
            force_control = torch.zeros(size=(states.shape[0], max_rollout_steps, 3), dtype=bool, device=device)
            
            # Track evaluation metrics
            infos_history = []
            rewards_history = []
            success_history = []
            engagement_history = []
            
            for i in tqdm.tqdm(range(max_rollout_steps), file=sys.stdout):
                # Get action
                actions = agent.act(
                    states, 
                    timestep=i, 
                    timesteps=max_rollout_steps
                )[-1]['mean_actions']

                vals.append(agent.value.act({"states": states}, role="value")[0])
                
                # Zero out actions for failed or successful environments
                actions[~alive_mask[:,0],:] *= 0.0
                actions[success_mask[:,0],:] *= 0.0
                
                # Take action
                next_states, step_rew, terminated, truncated, infos = env.step(actions)
                next_states = torch.cat((env.unwrapped.obs_buf['policy'], env.unwrapped.obs_buf['critic']), dim=1)
                env.cfg.recording = True
                
                # Collect force control information
                force_control[:,i,:] = env.unwrapped.actions[:,:3] > 0.5
                
                # Collect images
                for k in range(env.num_envs):
                    images[i, k*180:(k+1)*180, 0:240, :] = infos['img'][k,:,:,:]

                # Only track statistics for environments that haven't terminated
                active_envs = alive_mask[:,0]
                tot_rew[active_envs] += step_rew[active_envs]

                # Update alive mask
                mask_update = ~torch.logical_or(terminated, truncated)
                alive_mask *= mask_update
                
                # Get current successes and engagements
                curr_successes = env.unwrapped._get_curr_successes(
                    success_threshold=env.cfg_task.success_threshold, 
                    check_rot=env.cfg_task.name == "nut_thread"
                )
                engaged_mask[:,i] = env.unwrapped._get_curr_successes(
                    success_threshold=env.cfg_task.engage_threshold,
                    check_rot=env.cfg_task.name == "nut_thread"
                )

                if (curr_successes).any():
                    succ_step[
                        torch.logical_and(
                            succ_step == max_rollout_steps,
                            curr_successes
                        )
                    ] = i
                    success_mask[curr_successes,0] = True

                # Store evaluation data (only for non-terminated environments)
                eval_infos = {}
                for key, value in infos.items():
                    if 'Reward /' in key:
                        eval_infos[key] = value[active_envs] if active_envs.any() else value * 0
                    elif key == 'smoothness':
                        eval_infos[key] = {}
                        for smooth_key, smooth_val in value.items():
                            eval_infos[key][smooth_key] = smooth_val[active_envs] if active_envs.any() else smooth_val * 0
                
                infos_history.append(eval_infos)
                rewards_history.append(step_rew[active_envs] if active_envs.any() else torch.zeros_like(step_rew))
                success_history.append(curr_successes[active_envs] if active_envs.any() else torch.zeros_like(curr_successes))
                engagement_history.append(engaged_mask[active_envs,i] if active_envs.any() else torch.zeros_like(engaged_mask[:,i]))

                # Reset environments
                if env.num_envs > 1:
                    states = next_states
                else:
                    if terminated.any() or truncated.any():
                        with torch.no_grad():
                            states, infos = env.reset()
                    else:
                        states = next_states
            
            # Create GIF
            save_tensor_as_gif(images, gif_path, tot_rew, torch.cat(vals, dim=1), succ_step, engaged_mask, force_control)
            step_num = int(ckpt_path.split("/")[-1][6:-3])
            
            # Calculate evaluation metrics
            eval_metrics = calculate_evaluation_metrics(infos_history, rewards_history, success_history, engagement_history)
            
            # Log to WandB
            wandb.init(
                project=wandb_project,
                id=run_id, 
                resume="must"
            )
            
            log_data = {
                'Checkpoint videos': wandb.Video(
                    data_or_path=gif_path,
                    caption=f'Agent at Checkpoint {step_num}',
                    fps=args_cli.policy_hz,
                    format='gif'
                ),
                "env_step": step_num, 
                "exp_step": step_num // 256
            }
            
            # Add evaluation metrics
            log_data.update(eval_metrics)
            
            wandb.log(log_data)
            wandb.finish()

if __name__ == "__main__":
    main()
    simulation_app.close()