import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")

# exp
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Local-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment")
parser.add_argument("--ckpt_filepath", type=str, default=None, help="The full path to the checkpoint file")
parser.add_argument("--output_path", type=str, default=None, help="The full path to the checkpoint file")
parser.add_argument("--ckpt_step", type=int, default=-1, help="The step the checkpoint was taken at")

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
import os
import random
from datetime import datetime

import skrl
from packaging import version

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

#from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.close_gripper_action_wrapper import GripperCloseEnv
from models.default_mixin import Shared
#from models.bro_model import BroAgent
#from wrappers.DMP_observation_wrapper import DMPObservationWrapper

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import isaaclab_tasks  # noqa: F401
#import envs.FPiH.config.franka
import envs.factory

from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from agents.wandb_logger_ppo_agent import WandbLoggerPPO
from models.SimBa import SimBaActor, SimBaCritic
from skrl.resources.schedulers.torch import KLAdaptiveLR
import tqdm

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

def save_tensor_as_gif(tensor_list, filename, vals, succ_step, duration=100, loop=0):
    """
    Saves a list of PyTorch tensors as a GIF image.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to be saved as frames.
        filename (str): Output filename for the GIF.
        vals (any): List to write onto corner of images
        duration (int, optional): Duration of each frame in milliseconds. Defaults to 100.
        loop (int, optional): Number of times the GIF should loop. 0 means infinite loop. Defaults to 0.
    """
    tensor_list = tensor_list.permute(0, 3, 1, 2)
    tensor_list = 1.0 - tensor_list
    images = []
    #print(torch.max(tensor_list), torch.min(tensor_list))
    for i in range(50):#tensor_list:
        # Ensure the tensor is in CPU and convert it to a PIL Image
        tensor = tensor_list[i,:,:,:] 
        img = F.to_pil_image(tensor.to("cpu"))
        for img_idx, val in enumerate(vals[i]):
            x = img_idx % 4
            y = img_idx // 4
            draw = ImageDraw.Draw(img)
            #font = ImageFont.truetype("sans-serif.ttf", 16)
            #font = ImageFont.truetype("UbuntuMono-R.ttf", 20)            
            font = ImageFont.truetype("DejaVuSans.ttf",20)
            if i >= succ_step[img_idx]:
                draw.rectangle( 
                    (
                        (x*240, y*180),
                        ((x+1)*240, (y+1)*180)
                    ), 
                    outline="green", 
                    width=3
                )
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((x * 240, y*180+160),f"Value Est={round(val.item(),2)}",(0,255,0),font=font)

        images.append(img)
        #plt.plot(img)
        #plt.show()

    # Save the list of PIL Images as a GIF
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
        info['img'] = self.unwrapped.scene['tiled_camera'].data.output['rgb'] #observations['info']['img']
        return observations, info


if args_cli.seed == -1:
    args_cli.seed = random.randint(0, 10000)
seed = args_cli.seed
print("Seed:", seed)
set_seed(seed)

from isaaclab.sensors import TiledCameraCfg, ImuCfg
import isaaclab.sim as sim_utils
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: dict
):    
    print("initialize single ckpt video")
    ckpt_fp = args_cli.ckpt_filepath
    ckpt_step = args_cli.ckpt_step
    
    # create env
    num_envs = 8
    """Train with skrl agent."""
    max_rollout_steps = int((1/env_cfg.sim.dt) / env_cfg.decimation * env_cfg.episode_length_s)
    agent_cfg['agent']['rollouts'] = max_rollout_steps

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = num_envs 
    #env_cfg.scene.replicate_physics = True
    env_cfg.sim.device = env_cfg.sim.device

    env_cfg.num_agents = 1
    if 'learning_rate_scheduler' in agent_cfg['agent'].keys():
        # yaml doesn't read it as a class, but as a string idk
        agent_cfg['agent']['learning_rate_scheduler'] = KLAdaptiveLR

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
        debug_vis = False,
    )
    
    print("configured...")
    # create env
    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array"
    )
    
    env = Img2InfoWrapperclass(env)

    env = SkrlVecEnvWrapper(
        env, 
        ml_framework="torch"
    )  # same as: `wrap_env(env, wrapper="auto")    
    #env._reset_once = False
    env = GripperCloseEnv(env)
    
    print("env ready...")
    env.cfg.recording = True
    device = env.device
    
    models = {}

    models['policy'] = SimBaActor( #BroAgent(
        observation_space=env.cfg.observation_space, 
        action_space=env.action_space,
        #action_gain=0.05,
        device=device,
        act_init_std = agent_cfg['models']['act_init_std'],
        actor_n = agent_cfg['models']['actor']['n'],
        actor_latent = agent_cfg['models']['actor']['latent_size']
    ) 

    models["value"] = SimBaCritic( #BroAgent(
        state_space_size=env.cfg.state_space, 
        device=device,
        critic_output_init_mean = agent_cfg['models']['critic_output_init_mean'],
        critic_n = agent_cfg['models']['critic']['n'],
        critic_latent = agent_cfg['models']['critic']['latent_size']
    )

    agent_cfg['agent']['experiment']['wandb'] = False
    agent = WandbLoggerPPO(
        models=models,
        memory=None,
        cfg=agent_cfg['agent'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=env.num_envs,
        device=device
    )
    
    images = torch.zeros((max_rollout_steps, 2*180, 4*240, 3), device = env.device)
    print("starting main loop...")
    print(f"Filming {ckpt_fp}")
    with torch.no_grad():
        #   load agent
        agent.load(ckpt_fp)
        print("agent loaded")
        # reset env
        
        states, infos = env.reset()
        states = env.unwrapped._get_observations()
        states = torch.cat( (states['policy'], states['critic']),dim=1)
        
        alive_mask = torch.ones(size=(states.shape[0], 1), device=states.device, dtype=bool)
        vals = []
        success_mask = torch.zeros(size=(states.shape[0], 1), device=states.device, dtype=bool)
        succ_step = torch.ones(size=(states.shape[0],), device=states.device, dtype=torch.int32) * max_rollout_steps

        for i in tqdm.tqdm(range(max_rollout_steps), file=sys.stdout):
            # get action
            actions = agent.act(
                states, 
                timestep=1000, 
                timesteps=1000
            )[-1]['mean_actions']

            vals.append(agent.value.act({"states": states}, role="value")[0])
            actions[~alive_mask[:,0],:] *= 0.0
            actions[success_mask[:,0],:] *= 0.0
            
            # take action
            next_states, _, terminated, truncated, infos = env.step(actions)
            next_states = torch.cat( (env.unwrapped.obs_buf['policy'], env.unwrapped.obs_buf['critic']),dim=1)
    
            for k in range(env.num_envs):
                y = k // 4
                x = k % 4
                images[i, y*180:(y+1)*180, x*240:(x+1)*240, :] = infos['img'][k,:,:,:]

            mask_update = ~torch.logical_or(terminated, truncated)
            alive_mask *= mask_update
            curr_successes = env.unwrapped._get_curr_successes(
                success_threshold=env.cfg_task.success_threshold, 
                check_rot = env.cfg_task.name == "nut_thread"
            )

            if (curr_successes).any():
                succ_step[
                    torch.logical_and(
                        succ_step == max_rollout_steps,
                        curr_successes
                    )
                ] = i
                success_mask[curr_successes,0] = True

            # reset environments
            if env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = env.reset()
                else:
                    states = next_states
        # draw eval est + actions on image
        # make imgs into gif
        
        #img_path = f'{ckpt_fp[:-3]}.gif'
        save_tensor_as_gif(images, args_cli.output_path, vals, succ_step)
        
        # add gif to wandb 
        #wandb.log({
        #    "eval_video":wandb.Video(
        #        data_or_path=img_path,
        #        caption=f'Checkpoint at step {ckpt_step}',
        #        #fps=10,
        #        format='gif'
        #    ),
        #    "video_step": int(ckpt_step)
        #})

if __name__ == "__main__":
    main()
    simulation_app.close()