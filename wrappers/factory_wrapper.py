
import gymnasium as gym
from gymnasium import Wrapper
from envs.factory.factory_env import FactoryEnv

class FactoryWrapper(Wrapper):
    def __init__(
            self,
            env#: Env[ObsType, ActType]#,
            #render_mode: str | None=None,
            #**kwargs
    ):
        print("wrapper type:", type(env.unwrapped))
        super().__init__(env)
        #FactoryEnv.__init__(self.unwrapped.cfg, render_mode, **kwargs)

    def __getattr__(self, key: str):
        if hasattr(self.unwrapped, key):
            print(key, getattr(self.unwrapped, key))
            return getattr(self.unwrapped, key)
    #    if hasattr(self.unwrapped,key):
    #        return getattr(self.unwrapped, key)
    #    raise AttributeError(
    #        f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
    #    )
            
    def _pre_physics_step(self, action):
        print("factory  wrapped pre-physics")
        return self.unwrapped._pre_physics_step(action)

    def _apply_action(self, action):
        return self.unwrapped._apply_action(action)

    def _init_tensors(self):
        return self.unwrapped._init_tensors()

    def _set_body_inertias(self):
        return self.unwrapped._set_body_inertias()

    def _set_default_dynamics_parameters(self):
        return self.unwrapped._set_default_dynamics_parameters()

    def _init_force_torque_sensor(self):
        return self.unwrapped._init_force_torque_sensor()

    def _set_friction(self, asset, value):
        return self.unwrapped._set_friction(asset, value)

    def _get_keypoint_offsets(self, num_keypoints):
        return self.unwrapped._get_keypoint_offsets(num_keypoints)

    def _setup_scene(self):
        return self.unwrapped._setup_scene()

    def _compute_intermediate_values(self, dt):
        return self.unwrapped._compute_intermediate_values(dt)

    def _get_observations(self):
        return self.unwrapped._get_observations()

    def _reset_buffers(self, env_ids):
        return self.unwrapped._reset_buffers(env_ids)

    def close_gripper_in_place(self):
        return self.unwrapped.close_gripper_in_place()

    def _calc_ctrl_pos(self, min_idx=0, max_idx=3):
        return self.unwrapped._calc_ctrl_pos(min_idx, max_idx)

    def _calc_ctrl_quat(self, min_idx=0, max_idx=3):
        return self.unwrapped._calc_ctrl_quat(min_idx, max_idx)

    def _apply_actions(self):
        return self.unwrapped._apply_actions()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        return self.unwrapped._set_gains(prop_gains, rot_deriv_scale)

    def generate_ctrl_signals(self):
        return self.unwrapped.generate_ctrl_signals()

    def _get_dones(self):
        return self.unwrapped._get_dones()

    def _get_curr_successes(self, success_threshold, check_rot=False):
        return self.unwrapped._get_curr_successes(success_threshold, check_rot)

    def _get_rewards(self):
        return self.unwrapped._get_rewards()

    def _update_rew_buf(self, curr_successes):
        return self.unwrapped._update_rew_buf(curr_successes)

    def _reset_idx(self, env_ids):
        return self.unwrapped._reset_idx(env_ids)

    def _get_target_gear_base_offset(self):
        return self.unwrapped._get_target_gear_base_offset()

    def _set_assets_to_default_pose(self, env_ids):
        return self.unwrapped._set_assets_to_default_pose(env_ids)

    def set_pos_inverse_kinematics(self, env_ids):
        return self.unwrapped.set_pos_inverse_kinematics(env_ids)

    def get_handheld_asset_relative_pose(self):
        return self.unwrapped.get_handheld_asset_relative_pose()

    def _set_franka_to_default_pose(self, joints, env_ids):
        return self.unwrapped._set_franka_to_default_pose(joints, env_ids)

    def step_sim_no_action(self):
        return self.unwrapped.step_sim_no_action()

    def randomize_initial_state(self, env_ids):
        return self.unwrapped.randomize_initial_state(env_ids)

