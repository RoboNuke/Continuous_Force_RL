from gym import Env, spaces
import gym
from gym.spaces import Box
import torch
import envs.factory.factory_control as fc
try:
    import isaacsim.core.utils.torch as torch_utils
except:
    import omni.isaac.core.utils.torch as torch_utils


class HybridControlActionWrapper(gym.ActionWrapper):
    """
    Use this wrapper changes the task to using a hybrid control env
        in order to do this within the IsaacLab + Factory framework
        we override the _apply_action function which is called by IsaacLab
        env at each physics step and we overwrite (if we want to) generate_ctrl_signals
        which is called in factory env 

        This is simple hybrid control, we take the error in force and multiply it by a gain
        that theoretically converts it to the equivalent of pose error. This is then added 
        to the change in pose and fed to the initial env.  The logical inverse of the selection
        "matrix" (really a vector) is used to select between force control and pose control in 
        a given dimension 
    """

    def __init__(
            self, 
            env,
            history=1):
        super().__init__(env)
        # get these for factory env
        self.old_randomize_initial_state = self.env.unwrapped.randomize_initial_state
        self.env.unwrapped.randomize_initial_state = self.randomize_initial_state

        self.old_pre_physics_step = self.env.unwrapped._pre_physics_step
        self.env.unwrapped._pre_physics_step = self._pre_physics_step
        
        self.act_size = 12 * history

        # previous action is now larger
        #self.unwrapped.cfg.state_space += self.act_size - self.unwrapped.cfg.action_space
        #self.unwrapped.cfg.observation_space += self.act_size - self.unwrapped.cfg.action_space
        self.unwrapped.cfg.action_space = self.act_size

        # reconfigure spaces to match
        self.unwrapped._configure_gym_env_spaces()

        self.new_action = torch.zeros((self.unwrapped.num_envs, 6), device = self.unwrapped.device)
        self.unwrapped.prev_actions = torch.zeros_like(self.new_action)
        self.unwrapped.actions = torch.zeros_like(self.new_action)
        self.sel_matrix = torch.zeros((self.unwrapped.num_envs, 3), dtype=bool, device=self.unwrapped.device)
        self.force_action = torch.zeros((self.unwrapped.num_envs, 3), device=self.unwrapped.device)
        self.pose_action = torch.zeros_like(self.new_action)
        
        self.kp = torch.tensor(self.unwrapped.cfg.ctrl.default_task_force_gains, device=self.unwrapped.device).repeat(
            (self.unwrapped.num_envs, 1)
        )
        self.force_threshold = torch.tensor(self.unwrapped.cfg.ctrl.force_action_threshold, device=self.unwrapped.device).repeat(
            (self.unwrapped.num_envs, 1)
        )

        self.force_action_bounds = torch.tensor(self.unwrapped.cfg.ctrl.force_action_bounds, device=self.unwrapped.device)

    def randomize_initial_state(self, env_ids):
        self.old_randomize_initial_state(env_ids)
        # set a random force gain
        # this is setup so if we want to vary the gain we can add it here

    def _pre_physics_step(self, action):
        self.sel_matrix = torch.where(action[:,:3] > 0, True, False)
        self.pose_action = action[:,3:9]
        self.force_action = action[:,9:] * self.force_threshold
        self.force_action = torch.clip(
            self.force_action, -self.force_action_bounds, self.force_action_bounds
        )

        new_action = self.pose_action 
        # we inturrpret force action as a difference in force from current force reading
        # this means that goal_force = force_action + current_force
        # the error term is error = goal_force - current_force => error = force_action
        new_action[:,0:3] += self.kp * ~self.sel_matrix * self.force_action
        self.unwrapped.prev_action = self.unwrapped.actions
        self.unwrapped.actions = new_action

        self.old_pre_physics_step(new_action)

    def action(self, action):
        return action