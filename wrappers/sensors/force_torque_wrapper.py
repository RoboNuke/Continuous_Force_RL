"""
Force-Torque Sensor Wrapper for Factory Environment

This wrapper adds force-torque sensor functionality to any environment,
enabling measurement and observation of contact forces and torques at the robot end-effector.

Extracted from: factory_env.py lines 56, 78-79, 108-116, 199-201, 324-328, 365-371, 388-392, 415-419
"""

import torch
import gymnasium as gym

try:
    from isaacsim.core.api.robots import RobotView
    from isaaclab.sensors import ContactSensor
    from isaaclab.utils.math import quat_rotate_inverse
except ImportError:
    try:
        from omni.isaac.core.articulations import ArticulationView as RobotView
        from omni.issac.lab.sensors import ContactSensor
        from isaaclab.utils.math import quat_rotate_inverse
    except ImportError:
        RobotView = None
        quat_rotate_inverse = None


class ForceTorqueWrapper(gym.Wrapper):
    """
    Wrapper that adds force-torque sensor functionality to factory environments.

    Features:
    - Initializes force-torque sensor on robot end-effector
    - Provides force-torque measurements in observations
    - Tracks force/torque statistics for episode metrics
    - Supports tanh scaling for force readings
    """

    def __init__(self, env, use_tanh_scaling=False, tanh_scale=0.03, add_force_obs=False,
                 add_contact_obs=False, add_contact_state=True,
                 contact_force_threshold=0.1, contact_torque_threshold=0.01, log_contact_state=True,
                 use_contact_sensor=True):
        """
        Initialize the force-torque sensor wrapper.

        Args:
            env: Base environment to wrap
            use_tanh_scaling: Whether to apply tanh scaling to force-torque readings
            tanh_scale: Scale factor for tanh transformation
            add_force_obs: Whether to add force-torque to observations
            contact_force_threshold: Force threshold for contact detection (N)
            contact_torque_threshold: Torque threshold for contact detection (Nm)
            log_contact_state: Whether to log contact state to extras['to_log']
        """
        super().__init__(env)

        # Store configuration
        self.use_tanh_scaling = use_tanh_scaling
        self.tanh_scale = tanh_scale
        self.contact_force_threshold = contact_force_threshold
        self.contact_torque_threshold = contact_torque_threshold
        self.log_contact_state = log_contact_state
        self.use_contact_sensor = use_contact_sensor

        # Update observation and state dimensions using Isaac Lab's native approach
        if hasattr(self.unwrapped, 'cfg'):
            if add_force_obs:
                self.unwrapped.cfg.obs_order.append('force_torque')
                self.unwrapped.cfg.state_order.append('force_torque')
            if add_contact_state:
                self.unwrapped.cfg.state_order.append('in_contact')
            if add_contact_obs:
                self.unwrapped.cfg.obs_order.append('in_contact')
            self._update_observation_config()

        # Flag to track if sensor is initialized
        self._sensor_initialized = False

        # Store original methods
        self._original_init_tensors = None
        self._original_compute_intermediate_values = None
        self._original_reset_buffers = None
        self._original_pre_physics_step = None
        self._original_get_factory_obs_state_dict = None
        self._original_get_observations = None
        self.unwrapped.has_force_torque_sensor = True
        self._original_setup_scene = None 

        # Initialize in_contact tensor immediately for wrapper compatibility
        # num_envs is already set and won't change through wrappers
        num_envs = self.unwrapped.num_envs
        device = self.unwrapped.device
        self.unwrapped.in_contact = torch.zeros(
            (num_envs, 6), dtype=torch.bool, device=device
        )

        # Initialize after the base environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def reset(self, **kwargs):
        """
        Override reset to ensure wrapper is initialized before first use.

        The wrapper may not be fully initialized during __init__ because the
        environment's _robot attribute might not exist yet. This method ensures
        initialization happens on the first reset call when _robot is available.

        Args:
            **kwargs: Arguments passed to the base environment's reset method

        Returns:
            Observation and info from base environment's reset
        """
        print("="*100, "\nCalling Reset")
        if not self._sensor_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return super().reset(**kwargs)

    def _update_observation_config(self):
        """
        Update observation configuration to include force_torque dimensions.

        This method uses Isaac Lab's native configuration approach instead of
        relying on local factory_env_cfg imports.
        """
        env_cfg = self.unwrapped.cfg

        # Method 1: Try to update observation/state order and recalculate dimensions
        if hasattr(env_cfg, 'obs_order') and hasattr(env_cfg, 'state_order'):
            # Don't automatically add force_torque - let the user control where it goes
            # The wrapper will inject it if it's in the orders, skip if not
            # Try to find and update dimension configurations
            self._update_dimension_configs(env_cfg)

        # Method 2: Directly update observation/state space if available
        elif hasattr(env_cfg, 'observation_space') and hasattr(env_cfg, 'state_space'):
            # This method is a fallback that shouldn't be used - force proper obs_order/state_order usage
            raise ValueError("Environment has observation_space/state_space but missing obs_order/state_order. Use proper Isaac Lab configuration with obs_order and state_order.")

        # Method 3: Update gym environment spaces
        self._update_gym_spaces()

    def _update_dimension_configs(self, env_cfg):
        """
        Update dimension configurations for force-torque sensor integration.

        Imports Isaac Lab's dimension configs and adds force_torque dimensions.
        Provides fallbacks for test environments.

        Args:
            env_cfg: Environment configuration object

        Raises:
            ValueError: If obs_order or state_order are missing from config
        """
        # Import Isaac Lab's dimension configurations
        try:
            from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
        except ImportError:
            try:
                from isaaclab_tasks.isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
            except ImportError:
                # Fallback for test environments or when Isaac Lab is not available
                OBS_DIM_CFG = {
                    "fingertip_pos": 3,
                    "fingertip_pos_rel_fixed": 3,
                    "fingertip_quat": 4,
                    "ee_linvel": 3,
                    "ee_angvel": 3,
                    "joint_pos": 7,
                    "force_torque": 6,
                    "in_contact": 3
                }
                STATE_DIM_CFG = {
                    "fingertip_pos": 3,
                    "fingertip_pos_rel_fixed": 3,
                    "fingertip_quat": 4,
                    "ee_linvel": 3,
                    "ee_angvel": 3,
                    "joint_pos": 7,
                    "held_pos": 3,
                    "held_pos_rel_fixed": 3,
                    "held_quat": 4,
                    "fixed_pos": 3,
                    "fixed_quat": 4,
                    "force_torque": 6,
                    "in_contact": 3
                }

        # Add force_torque dimensions to the configs if not already present
        if 'force_torque' not in OBS_DIM_CFG:
            OBS_DIM_CFG['force_torque'] = 6
        if 'force_torque' not in STATE_DIM_CFG:
            STATE_DIM_CFG['force_torque'] = 6

        # Add in_contact dimensions to the configs if not already present
        if 'in_contact' not in OBS_DIM_CFG:
            OBS_DIM_CFG['in_contact'] = 3
        if 'in_contact' not in STATE_DIM_CFG:
            STATE_DIM_CFG['in_contact'] = 3

        # Verify required config attributes exist
        if not hasattr(env_cfg, 'obs_order'):
            raise ValueError("Environment config missing required 'obs_order' attribute")
        if not hasattr(env_cfg, 'state_order'):
            raise ValueError("Environment config missing required 'state_order' attribute")

        # Recalculate observation and state space dimensions using Isaac Lab's method
        try:
            env_cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in env_cfg.obs_order])
            env_cfg.state_space = sum([STATE_DIM_CFG[state] for state in env_cfg.state_order])
        except KeyError as e:
            raise KeyError(f"Unknown observation/state component in obs_order/state_order: {e}")

        # ALWAYS add action space dimensions - this is what the tensor creation does
        if hasattr(env_cfg, 'action_space'):
            env_cfg.observation_space += env_cfg.action_space
            env_cfg.state_space += env_cfg.action_space
        else:
            # Use default action space size if not specified
            env_cfg.observation_space += 12  # Default 12-DOF
            env_cfg.state_space += 12

        # Update component_attr_map if it exists (for attribute mapping)
        if hasattr(env_cfg, 'component_attr_map'):
            if 'force_torque' not in env_cfg.component_attr_map:
                env_cfg.component_attr_map['force_torque'] = 'robot_force_torque'

    def _update_gym_spaces(self):
        """
        Update gymnasium environment spaces if needed.

        Attempts to reconfigure the environment's observation and action spaces
        to account for the additional force-torque sensor dimensions. Silently
        continues if reconfiguration fails to maintain compatibility.
        """
        if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
            try:
                self.unwrapped._configure_gym_env_spaces()
            except Exception:
                # Silently continue if reconfiguration fails
                pass

    def _check_no_history_wrapper(self):
        """
        Check that no history wrapper is already applied in the wrapper chain.

        History wrapper must be the last wrapper applied to ensure proper observation
        space calculation and buffer management. If a history wrapper is detected,
        raise an error with clear instructions.
        """
        current_env = self.env
        while current_env is not None:
            # Check if current wrapper is a history wrapper
            if hasattr(current_env, '__class__') and 'HistoryObservationWrapper' in str(current_env.__class__):
                raise ValueError(
                    "ERROR: History wrapper detected in wrapper chain before ForceTorqueWrapper.\n"
                    "\n"
                    "SOLUTION: History wrapper must be applied LAST in the wrapper chain.\n"
                    "Correct order:\n"
                    "  1. Apply ForceTorqueWrapper first\n"
                    "  2. Apply other observation/sensor wrappers\n"
                    "  3. Apply HistoryObservationWrapper last\n"
                    "\n"
                    "Current wrapper chain violates this requirement.\n"
                    "Please reorder your wrapper application to ensure HistoryObservationWrapper is applied last."
                )

            # Move to next wrapper in chain
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            else:
                break

    def _initialize_wrapper(self):
        """
        Initialize the wrapper after the base environment is set up.

        This method performs lazy initialization by overriding environment methods
        and setting up the force-torque sensor interface. It's called automatically
        when the environment's robot attribute is detected or during first step/reset.

        The initialization includes:
        - Storing and overriding environment methods (_init_tensors, _compute_intermediate_values, etc.)
        - Initializing the Isaac Sim RobotView for force-torque sensor access
        - Setting up sensor data buffers
        """
        if self._sensor_initialized:
            return

        # Check that no history wrapper is already applied - history must be the last wrapper
        self._check_no_history_wrapper()


        # Store original methods
        if hasattr(self.unwrapped, '_init_tensors'):
            self._original_init_tensors = self.unwrapped._init_tensors
            self.unwrapped._init_tensors = self._wrapped_init_tensors

        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self._original_compute_intermediate_values = self.unwrapped._compute_intermediate_values
            self.unwrapped._compute_intermediate_values = self._wrapped_compute_intermediate_values

        if hasattr(self.unwrapped, '_reset_buffers'):
            self._original_reset_buffers = self.unwrapped._reset_buffers
            self.unwrapped._reset_buffers = self._wrapped_reset_buffers

        if hasattr(self.unwrapped, '_pre_physics_step'):
            self._original_pre_physics_step = self.unwrapped._pre_physics_step
            self.unwrapped._pre_physics_step = self._wrapped_pre_physics_step
            
        # Override the observation creation method to inject force-torque data
        if hasattr(self.unwrapped, '_get_factory_obs_state_dict'):
            # Newer Isaac Lab version - use clean dictionary approach
            self._original_get_factory_obs_state_dict = self.unwrapped._get_factory_obs_state_dict
            self.unwrapped._get_factory_obs_state_dict = self._wrapped_get_factory_obs_state_dict
        elif hasattr(self.unwrapped, '_get_observations'):
            # Older Isaac Lab version - use tensor injection approach
            self._original_get_observations = self.unwrapped._get_observations
            self.unwrapped._get_observations = self._wrapped_get_observations
        else:
            raise ValueError("Factory environment missing required observation methods")

        # Initialize force-torque sensor
        self._init_force_torque_sensor()

        # Reference the contact sensor from the scene ONLY if use_contact_sensor is enabled
        if self.use_contact_sensor:
            if hasattr(self.unwrapped.scene, 'sensors') and "held_fixed_contact_sensor" in self.unwrapped.scene.sensors:
                self._held_fixed_contact_sensor = self.unwrapped.scene.sensors["held_fixed_contact_sensor"]

                # Debug: Print comprehensive sensor and filter information
                if hasattr(self._held_fixed_contact_sensor, 'contact_physx_view'):
                    filter_count = self._held_fixed_contact_sensor.contact_physx_view.filter_count
                    print(f"\n{'='*80}")
                    print(f"[CONTACT SENSOR DEBUG]")
                    print(f"  Sensor prim_path: {self._held_fixed_contact_sensor.cfg.prim_path}")
                    print(f"  Filter expressions: {self._held_fixed_contact_sensor.cfg.filter_prim_paths_expr}")
                    print(f"  Filter count: {filter_count}")
                    print(f"  Number of sensor bodies: {self._held_fixed_contact_sensor._num_bodies}")

                    # Print the actual body names that were found
                    if hasattr(self._held_fixed_contact_sensor, '_body_physx_view'):
                        try:
                            # Try different ways to get body names
                            if hasattr(self._held_fixed_contact_sensor._body_physx_view, 'body_names'):
                                body_names = self._held_fixed_contact_sensor._body_physx_view.body_names
                            elif hasattr(self._held_fixed_contact_sensor, '_body_names'):
                                body_names = self._held_fixed_contact_sensor._body_names
                            else:
                                body_names = "Could not retrieve body names"
                            print(f"  Sensor body names: {body_names}")
                        except Exception as e:
                            print(f"  Could not get sensor body names: {e}")

                    # Check if ContactReportAPI is enabled on filtered bodies
                    try:
                        import omni
                        from pxr import PhysxSchema, UsdPhysics
                        stage = omni.usd.get_context().get_stage()
                        filter_expr = self._held_fixed_contact_sensor.cfg.filter_prim_paths_expr[0]

                        # Check specific path in env_0
                        test_path = filter_expr.replace("env_.*", "env_0")
                        test_prim = stage.GetPrimAtPath(test_path)

                        print(f"\n  Checking filtered body ContactReportAPI:")
                        print(f"    Test path: {test_path}")
                        print(f"    Prim exists: {test_prim.IsValid()}")

                        # If test path doesn't exist, enumerate what does exist
                        if not test_prim.IsValid():
                            print(f"\n  Enumerating FixedAsset children in env_0:")
                            fixed_asset_path = "/World/envs/env_0/FixedAsset"
                            fixed_asset_prim = stage.GetPrimAtPath(fixed_asset_path)
                            if fixed_asset_prim.IsValid():
                                for child in fixed_asset_prim.GetChildren():
                                    child_path = str(child.GetPath())
                                    print(f"    - {child_path} (type: {child.GetTypeName()})")
                            else:
                                print(f"    FixedAsset prim doesn't exist at {fixed_asset_path}")

                        if test_prim.IsValid():
                            # Check for ContactReportAPI
                            has_contact_api = test_prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                            print(f"    Has ContactReportAPI: {has_contact_api}")

                            # Check if it's a rigid body
                            has_rigid_body = test_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                            print(f"    Has RigidBodyAPI: {has_rigid_body}")

                            # If has ContactReportAPI, check threshold
                            if has_contact_api:
                                contact_api = PhysxSchema.PhysxContactReportAPI(test_prim)
                                if contact_api.GetThresholdAttr():
                                    threshold = contact_api.GetThresholdAttr().Get()
                                    print(f"    ContactReport threshold: {threshold}")
                                else:
                                    print(f"    ContactReport threshold: Not set")
                            else:
                                print(f"    [WARNING] FixedAsset body has NO ContactReportAPI!")
                                print(f"    This is why force_matrix_w returns zeros.")
                        else:
                            print(f"    [ERROR] Prim does not exist at path!")
                    except Exception as e:
                        print(f"  Error checking ContactReportAPI: {e}")

                    if filter_count == 0:
                        print(f"\n  [WARNING] Filter count is 0!")
                        print(f"  This means filter_prim_paths_expr is not matching any bodies in the scene.")
                    else:
                        print(f"  [OK] Filter is matching {filter_count} body/bodies")
                    print(f"{'='*80}\n")
            else:
                raise ValueError(
                    "use_contact_sensor=True but held_fixed_contact_sensor not found in scene. "
                    "Ensure the environment was created with create_sensor_enabled_factory_env() "
                    "and the sensor config exists in the task config."
                )
        else:
            self._held_fixed_contact_sensor = None

        self._sensor_initialized = True

    def _init_force_torque_sensor(self):
        """
        Initialize the force-torque sensor interface using Isaac Sim's RobotView.

        Creates a RobotView instance to access joint force measurements from the
        Isaac Sim physics simulation. The sensor targets joint 8 (end-effector)
        to provide 6-DOF force-torque measurements.

        Raises:
            ImportError: If RobotView cannot be imported from Isaac Sim

        Note:
            Gracefully handles initialization failures by setting _robot_av to None
            and printing a warning message.
        """
        if RobotView is None:
            raise ImportError("Could not import RobotView. Please ensure Isaac Sim is properly installed.")

        try:
            self._robot_av = RobotView(prim_paths_expr="/World/envs/env_.*/Robot")
            self._robot_av.initialize()
        except Exception as e:
            print(f"Warning: Failed to initialize force-torque sensor: {e}")
            self._robot_av = None

    def _wrapped_init_tensors(self):
        """
        Initialize tensors including force-torque sensor data.

        This method wraps the environment's original _init_tensors method and
        additionally initializes the robot_force_torque tensor for storing
        6-DOF force-torque measurements (3 force + 3 torque components).

        The force-torque tensor is initialized as zeros with shape (num_envs, 6)
        on the same device as the environment.
        """
        # Call original initialization
        if self._original_init_tensors:
            self._original_init_tensors()

        # Initialize force-torque tensors
        num_envs = self.unwrapped.num_envs
        device = self.unwrapped.device

        # Main force-torque sensor data
        self.unwrapped.robot_force_torque = torch.zeros(
            (num_envs, 6), dtype=torch.float32, device=device
        )

        # Also initialize force_torque attribute for factory obs_dict creation
        # This is what the factory environment will access when creating obs_dict
        self.unwrapped.force_torque = self.unwrapped.robot_force_torque

        # Note: in_contact tensor is now initialized in __init__ for wrapper compatibility


    def _wrapped_compute_intermediate_values(self, dt):
        """
        Compute intermediate values including force-torque measurements.

        This method wraps the environment's original _compute_intermediate_values
        method and additionally updates force-torque sensor readings from the
        Isaac Sim physics simulation.

        Args:
            dt (float): Physics simulation timestep

        The method:
        1. Calls the original compute intermediate values method
        2. Retrieves joint forces from the RobotView sensor interface
        3. Extracts force-torque data from joint 8 (end-effector)
        4. Updates the environment's robot_force_torque tensor
        5. Gracefully handles sensor failures by setting data to zeros
        """
        # Call original computation
        if self._original_compute_intermediate_values:
            self._original_compute_intermediate_values(dt)

        # Get force-torque sensor data and make it available for obs_dict creation
        if self._robot_av is not None:
            #try:
            # Force-torque sensor is at joint 8 (end-effector)
            joint_forces = self._robot_av.get_measured_joint_forces()
            if joint_forces.shape[1] > 8:  # Ensure joint 8 exists
                force_torque_data = joint_forces[:, 8, :]
                self.unwrapped.robot_force_torque = force_torque_data
                # Also set as force_torque attribute for factory obs_dict creation
                self.unwrapped.force_torque = force_torque_data
            else:
                # Fallback to zeros if joint 8 doesn't exist
                self.unwrapped.robot_force_torque.fill_(0.0)
                self.unwrapped.force_torque = self.unwrapped.robot_force_torque
            #except Exception as e:
            #    # Fallback to zeros if measurement fails
            #    self.unwrapped.robot_force_torque.fill_(0.0)
            #    self.unwrapped.force_torque = self.unwrapped.robot_force_torque

        # Compute contact state AFTER scene.update() for current-step data
        # Moved from _wrapped_pre_physics_step to read sensor data at correct time
        if self.use_contact_sensor:
            # Use ContactSensor for contact detection
            self._contact_detected_in_range()
            self.unwrapped.in_contact[:, :3] = self.real_contact
            self.unwrapped.in_contact[:, 3:] = False
        else:
            # Use force-torque thresholds for contact detection
            self.unwrapped.in_contact[:, :3] = torch.abs(self.unwrapped.robot_force_torque[:, :3]) > self.contact_force_threshold
            self.unwrapped.in_contact[:, 3:] = torch.abs(self.unwrapped.robot_force_torque[:, 3:]) > self.contact_torque_threshold

        # Log contact state if enabled
        if self.log_contact_state and hasattr(self.unwrapped, 'extras'):
            if 'to_log' not in self.unwrapped.extras:
                self.unwrapped.extras['to_log'] = {}
            self.unwrapped.extras['to_log']['Contact / In-Contact X'] = self.unwrapped.in_contact[:, 0].float()
            self.unwrapped.extras['to_log']['Contact / In-Contact Y'] = self.unwrapped.in_contact[:, 1].float()
            self.unwrapped.extras['to_log']['Contact / In-Contact Z'] = self.unwrapped.in_contact[:, 2].float()

    ###########################################################################
    ### HERE FOR TESTING ONLY TODO REMOVE ME!! ################################
    def _contact_detected_in_range(self):
        # get true state from directly-held contact sensor (not from scene)
        if self._held_fixed_contact_sensor is not None:
            # Get contact forces in WORLD frame
            net_contact_force_world = self._held_fixed_contact_sensor.data.net_forces_w
            fixed_force_w = self._held_fixed_contact_sensor.data.force_matrix_w[:,0,0,:]

            # Debug: Compare filtered vs unfiltered contact forces
            max_net_force = torch.max(torch.abs(net_contact_force_world))
            max_filtered_force = torch.max(torch.abs(fixed_force_w))
            if max_net_force > 0.01 or max_filtered_force > 0.01:
                print(f"[CONTACT DEBUG] Max net force (all contacts): {max_net_force:.4f}, Max filtered force (peg-hole only): {max_filtered_force:.4f}")
            # We use ee quat because we want the forces in the force-torque frame for hybrid control
            # The sensing is placed here because the data is from the end of the last step, allowing
            # the next (current) step to decide what do to based on step's starting state 
            ee_quat = self.unwrapped.fingertip_midpoint_quat  # [num_envs, 4]
            #held_quat = self.unwrapped.held_quat  # [num_envs, 4]
            net_contact_force_ee = quat_rotate_inverse(ee_quat, fixed_force_w)

            # Detect contact using END-EFFECTOR frame forces
            self.real_contact = torch.where(
                torch.isclose(net_contact_force_ee, torch.zeros_like(net_contact_force_ee), atol=1.0e-3, rtol=0.0),
                torch.zeros_like(net_contact_force_ee),
                torch.ones_like(net_contact_force_ee)
            ).bool()

            # Log END-EFFECTOR frame contact (not world frame)
            self.unwrapped.extras['to_log']['Contact / Real Contact X'] = self.real_contact[:,0].float()
            self.unwrapped.extras['to_log']['Contact / Real Contact Y'] = self.real_contact[:,1].float()
            self.unwrapped.extras['to_log']['Contact / Real Contact Z'] = self.real_contact[:,2].float()

        #########################################################################

    def _wrapped_reset_buffers(self, env_ids):
        """
        Reset force-torque buffers for specified environments.

        This method wraps the environment's original _reset_buffers method
        and additionally resets force-torque sensor readings to zero for
        the specified environment indices.

        Args:
            env_ids (torch.Tensor): Indices of environments to reset
        """
        # Call original reset
        if self._original_reset_buffers:
            self._original_reset_buffers(env_ids)

        # Reset force-torque sensor readings
        if hasattr(self.unwrapped, 'robot_force_torque'):
            self.unwrapped.robot_force_torque[env_ids] = 0.0
            # Also reset force_torque attribute for obs_dict creation
            if hasattr(self.unwrapped, 'force_torque'):
                self.unwrapped.force_torque[env_ids] = 0.0

    def _wrapped_pre_physics_step(self, action):
        """Update during physics step."""
        # Call original pre-physics step
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action)

        # Contact detection moved to _wrapped_compute_intermediate_values
        # to ensure we read sensor data AFTER scene.update() for current-step accuracy

    def _wrapped_get_factory_obs_state_dict(self):
        """
        Override factory observation and state dictionary creation to inject force-torque data.

        This method wraps the environment's _get_factory_obs_state_dict and adds
        force_torque to both observation and state dictionaries.

        Returns:
            tuple: (obs_dict, state_dict) with force_torque injected
        """
        # Call original method to get base dictionaries
        obs_dict, state_dict = self._original_get_factory_obs_state_dict()

        # Inject force-torque data if it's in the observation/state order
        env_cfg = self.unwrapped.cfg
        if hasattr(env_cfg, 'obs_order') and 'force_torque' in env_cfg.obs_order:
            obs_dict['force_torque'] = self.get_force_torque_observation()

        if hasattr(env_cfg, 'state_order') and 'force_torque' in env_cfg.state_order:
            state_dict['force_torque'] = self.get_force_torque_observation()

        # Inject in-contact data if it's in the observation/state order
        if hasattr(env_cfg, 'obs_order') and 'in_contact' in env_cfg.obs_order:
            obs_dict['in_contact'] = self.get_in_contact_observation()

        if hasattr(env_cfg, 'state_order') and 'in_contact' in env_cfg.state_order:
            state_dict['in_contact'] = self.get_in_contact_observation()

        return obs_dict, state_dict

    def _wrapped_get_observations(self):
        """Override _get_observations to inject force_torque into policy/critic tensors."""
        # Get original tensors (they're missing force_torque, so 6 elements shorter)
        noisy_fixed_pos = self.unwrapped.fixed_pos_obs_frame + self.unwrapped.init_fixed_pos_obs_noise

        prev_actions = self.unwrapped.actions.clone()

        obs_dict = {
            "fingertip_pos": self.unwrapped.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.unwrapped.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.unwrapped.fingertip_midpoint_quat,
            "ee_linvel": self.unwrapped.ee_linvel_fd,
            "ee_angvel": self.unwrapped.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.unwrapped.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_obs_frame,
            "fingertip_quat": self.unwrapped.fingertip_midpoint_quat,
            "ee_linvel": self.unwrapped.fingertip_midpoint_linvel,
            "ee_angvel": self.unwrapped.fingertip_midpoint_angvel,
            "joint_pos": self.unwrapped.joint_pos[:, 0:7],
            "held_pos": self.unwrapped.held_pos,
            "held_pos_rel_fixed": self.unwrapped.held_pos - self.unwrapped.fixed_pos_obs_frame,
            "held_quat": self.unwrapped.held_quat,
            "fixed_pos": self.unwrapped.fixed_pos,
            "fixed_quat": self.unwrapped.fixed_quat,
            "task_prop_gains": self.unwrapped.cfg.ctrl.default_task_prop_gains,
            "pos_threshold": self.unwrapped.cfg.ctrl.pos_action_threshold,
            "rot_threshold": self.unwrapped.cfg.ctrl.rot_action_threshold,
            "prev_actions": prev_actions,
        }

        # Get force_torque data
        force_torque_data = self.get_force_torque_observation()  # shape: [num_envs, 6]

        # Inject into policy tensor if needed
        if hasattr(self.unwrapped.cfg, 'obs_order') and 'force_torque' in self.unwrapped.cfg.obs_order:
            obs_dict['force_torque'] = force_torque_data

        # Inject into critic tensor if needed
        if hasattr(self.unwrapped.cfg, 'state_order') and 'force_torque' in self.unwrapped.cfg.state_order:
            state_dict['force_torque'] = force_torque_data

        # Get in_contact data
        in_contact_data = self.get_in_contact_observation()  # shape: [num_envs, 3]

        # Inject into policy tensor if needed
        if hasattr(self.unwrapped.cfg, 'obs_order') and 'in_contact' in self.unwrapped.cfg.obs_order:
            obs_dict['in_contact'] = in_contact_data

        # Inject into critic tensor if needed
        if hasattr(self.unwrapped.cfg, 'state_order') and 'in_contact' in self.unwrapped.cfg.state_order:
            state_dict['in_contact'] = in_contact_data

        obs_tensors = [obs_dict[obs_name] for obs_name in self.unwrapped.cfg.obs_order + ["prev_actions"]]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        state_tensors = [state_dict[state_name] for state_name in self.unwrapped.cfg.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)
        return {"policy": obs_tensors, "critic": state_tensors}

    def _insert_into_tensor(self, original_tensor, force_data, order_list, obs_type):
        """Insert force_torque data at correct position by splitting and concatenating."""
        ft_index = order_list.index('force_torque')

        # Calculate insertion position (sum of dimensions before force_torque)
        insertion_pos = sum(self._get_obs_dim(item, obs_type) for item in order_list[:ft_index])

        # Split original tensor at insertion point
        before_part = original_tensor[..., :insertion_pos]
        after_part = original_tensor[..., insertion_pos:]

        # Concatenate: [before, force_torque, after]
        return torch.cat([before_part, force_data, after_part], dim=-1)

    def _get_obs_dim(self, obs_name, obs_type):
        """Get the dimension of an observation component."""
        # Try to get from Isaac Lab's dimension configurations
        try:
            if obs_type == 'obs':
                from isaaclab_tasks.isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG
                return OBS_DIM_CFG.get(obs_name, 6)  # Default to 6 for force_torque
            else:  # state
                from isaaclab_tasks.isaaclab_tasks.direct.factory.factory_env_cfg import STATE_DIM_CFG
                return STATE_DIM_CFG.get(obs_name, 6)  # Default to 6 for force_torque
        except ImportError:
            # Fallback dimensions for common observations
            default_dims = {
                'fingertip_pos': 3,
                'fingertip_pos_rel_fixed': 3,
                'fingertip_quat': 4,
                'ee_linvel': 3,
                'ee_angvel': 3,
                'joint_pos': 7,
                'held_pos': 3,
                'held_pos_rel_fixed': 3,
                'held_quat': 4,
                'fixed_pos': 3,
                'fixed_quat': 4,
                'force_torque': 6,
                'in_contact': 3,
                'prev_actions': 12  # Assuming 12-DOF action space
            }
            return default_dims.get(obs_name, 3)  # Default to 3 if unknown

    def has_force_torque_data(self):
        """
        Check if force-torque data is available in the environment.

        Returns:
            bool: True if robot_force_torque attribute exists, False otherwise
        """
        return hasattr(self.unwrapped, 'robot_force_torque')

    def get_current_force_torque(self):
        """
        Get current force-torque readings split into force and torque components.

        Returns:
            dict: Dictionary with 'current_force' and 'current_torque' keys containing
                  tensors of shape (num_envs, 3) each, or empty dict if no data available
        """
        if self.has_force_torque_data():
            return {
                'current_force': self.unwrapped.robot_force_torque[:, :3],
                'current_torque': self.unwrapped.robot_force_torque[:, 3:],
            }
        return {}

    def get_stats(self):
        """
        Get statistics about the force-torque wrapper state.

        Returns:
            dict: Dictionary containing wrapper statistics and sensor status
        """
        stats = {
            'sensor_initialized': self._sensor_initialized,
            'has_force_torque_data': self.has_force_torque_data(),
            'use_tanh_scaling': self.use_tanh_scaling,
            'tanh_scale': self.tanh_scale
        }

        if self.has_force_torque_data():
            force_torque_data = self.unwrapped.robot_force_torque
            stats.update({
                'force_torque_shape': list(force_torque_data.shape),
                'force_torque_device': str(force_torque_data.device),
                'force_torque_dtype': str(force_torque_data.dtype)
            })

        return stats

    def get_force_torque_observation(self):
        """
        Get force-torque data formatted for observations.

        Retrieves the current force-torque sensor readings and optionally
        applies tanh scaling for bounded output values. This method is
        suitable for including force-torque data in observation vectors.

        Returns:
            torch.Tensor: Force-torque observation data with shape (num_envs, 6)
                         Optionally scaled with tanh if use_tanh_scaling is True.
                         Returns zeros if no sensor data is available.

        Note:
            When tanh scaling is enabled, the output values are bounded to [-1, 1]
            using the formula: tanh(tanh_scale * force_torque_data)
        """
        if self.has_force_torque_data():
            force_torque_obs = self.unwrapped.robot_force_torque.clone()

            if self.use_tanh_scaling:
                force_torque_obs = torch.tanh(self.tanh_scale * force_torque_obs)

            return force_torque_obs
        else:
            # Return zeros if no data available
            return torch.zeros((self.unwrapped.num_envs, 6), device=self.unwrapped.device)

    def get_in_contact_observation(self):
        """
        Get in-contact data formatted for observations.

        Retrieves the current in-contact state (first 3 elements - force contact flags)
        and converts from boolean to float for use in observation vectors.

        Returns:
            torch.Tensor: In-contact observation data with shape (num_envs, 3)
                         Float values (0.0 or 1.0) representing contact state.
                         Returns zeros if no contact data is available.
        """
        if hasattr(self.unwrapped, 'in_contact'):
            # Take only first 3 elements (force contact flags) and convert to float
            in_contact_obs = self.unwrapped.in_contact[:, :3].float()
            return in_contact_obs
        else:
            # Return zeros if no data available
            return torch.zeros((self.unwrapped.num_envs, 3), device=self.unwrapped.device)