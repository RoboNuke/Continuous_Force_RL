# Solution 1: Nested Dataclasses Implementation Plan

## Overview
Transform the flat configuration dataclasses to nested dataclasses that mirror the YAML structure while maintaining the defaults-first approach. This will ensure launch_utils functions receive the exact nested dictionary structure they expect (e.g., `model['actor']['n']`).

## Implementation Steps

### Phase 1: Create Nested Dataclass Structure

#### Step 1.1: Create Actor Configuration Dataclass
**File**: `configs/cfg_exts/actor_cfg.py`
```python
@dataclass
class ActorConfig:
    """Actor network configuration with Isaac Lab defaults."""
    n: int = 1                    # Number of SimBa layers (Isaac Lab default)
    latent_size: int = 256        # Hidden dimension (Isaac Lab default)

    def validate(self) -> None:
        """Validate actor configuration parameters."""
        if self.n <= 0:
            raise ValueError(f"Actor layers (n) must be positive, got: {self.n}")
        if self.latent_size <= 0:
            raise ValueError(f"Actor latent_size must be positive, got: {self.latent_size}")
```

#### Step 1.2: Create Critic Configuration Dataclass
**File**: `configs/cfg_exts/critic_cfg.py`
```python
@dataclass
class CriticConfig:
    """Critic network configuration with Isaac Lab defaults."""
    n: int = 3                    # Number of SimBa layers (Isaac Lab default)
    latent_size: int = 1024       # Hidden dimension (Isaac Lab default)

    def validate(self) -> None:
        """Validate critic configuration parameters."""
        if self.n <= 0:
            raise ValueError(f"Critic layers (n) must be positive, got: {self.n}")
        if self.latent_size <= 0:
            raise ValueError(f"Critic latent_size must be positive, got: {self.latent_size}")
```

#### Step 1.3: Create Wrapper Sub-Configuration Dataclasses
**File**: `configs/cfg_exts/wrapper_sub_configs.py`
```python
@dataclass
class ForceTorqueSensorConfig:
    """Force-torque sensor wrapper configuration."""
    enabled: bool = False
    use_tanh_scaling: bool = False
    tanh_scale: float = 0.03

@dataclass
class HybridControlConfig:
    """Hybrid control wrapper configuration."""
    enabled: bool = False
    reward_type: str = "simp"

@dataclass
class ObservationNoiseConfig:
    """Observation noise wrapper configuration."""
    enabled: bool = False
    global_scale: float = 1.0
    apply_to_critic: bool = True
    seed: Optional[int] = None

@dataclass
class WandbLoggingConfig:
    """Wandb logging wrapper configuration."""
    enabled: bool = True
    wandb_project: str = "Continuous_Force_RL"
    wandb_entity: str = "hur"
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

@dataclass
class ActionLoggingConfig:
    """Action logging wrapper configuration."""
    enabled: bool = False
    track_selection: bool = True
    track_pos: bool = True
    track_rot: bool = True
    track_force: bool = True
    track_torque: bool = True
    force_size: int = 6
    logging_frequency: int = 100
```

### Phase 2: Update Main Configuration Dataclasses

#### Step 2.1: Replace ExtendedModelConfig
**File**: `configs/cfg_exts/extended_model_cfg.py`
```python
from .actor_cfg import ActorConfig
from .critic_cfg import CriticConfig

@dataclass
class ExtendedModelConfig:
    """Model configuration with nested structure mirroring YAML."""

    # Nested configurations (maintain natural YAML structure)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    hybrid_agent: Optional[ExtendedHybridAgentConfig] = None

    # Flat configurations (Isaac Lab defaults)
    force_encoding: Optional[str] = None
    last_layer_scale: float = 1.0
    act_init_std: float = 1.0
    critic_output_init_mean: float = 50.0
    use_hybrid_agent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary structure for launch_utils."""
        result = {
            'actor': {
                'n': self.actor.n,
                'latent_size': self.actor.latent_size
            },
            'critic': {
                'n': self.critic.n,
                'latent_size': self.critic.latent_size
            },
            'force_encoding': self.force_encoding,
            'last_layer_scale': self.last_layer_scale,
            'act_init_std': self.act_init_std,
            'critic_output_init_mean': self.critic_output_init_mean,
            'use_hybrid_agent': self.use_hybrid_agent
        }

        if self.hybrid_agent is not None:
            result['hybrid_agent'] = self.hybrid_agent.to_dict()

        return result

    def validate(self) -> None:
        """Validate model configuration."""
        self.actor.validate()
        self.critic.validate()

        if self.use_hybrid_agent and self.hybrid_agent is None:
            raise ValueError("use_hybrid_agent=True but no hybrid_agent config provided")

        if not self.use_hybrid_agent and self.hybrid_agent is not None:
            print(f"\033[93m[CONFIG]: Warning: hybrid_agent config provided but use_hybrid_agent=False\033[0m")
```

#### Step 2.2: Replace ExtendedWrapperConfig
**File**: `configs/cfg_exts/extended_wrapper_cfg.py`
```python
from .wrapper_sub_configs import (
    ForceTorqueSensorConfig, HybridControlConfig, ObservationNoiseConfig,
    WandbLoggingConfig, ActionLoggingConfig
)

@dataclass
class ExtendedWrapperConfig:
    """Wrapper configuration with nested structure mirroring YAML."""

    # Nested wrapper configurations
    force_torque_sensor: ForceTorqueSensorConfig = field(default_factory=ForceTorqueSensorConfig)
    hybrid_control: HybridControlConfig = field(default_factory=HybridControlConfig)
    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)
    wandb_logging: WandbLoggingConfig = field(default_factory=WandbLoggingConfig)
    action_logging: ActionLoggingConfig = field(default_factory=ActionLoggingConfig)

    # Simple wrapper configurations (flat)
    fragile_objects_enabled: bool = True
    efficient_reset_enabled: bool = True
    observation_manager_enabled: bool = True
    observation_manager_merge_strategy: str = "concatenate"
    factory_metrics_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary structure for launch_utils."""
        return {
            'fragile_objects': {
                'enabled': self.fragile_objects_enabled,
                # num_agents and break_force computed from primary config
            },
            'efficient_reset': {
                'enabled': self.efficient_reset_enabled
            },
            'force_torque_sensor': {
                'enabled': self.force_torque_sensor.enabled,
                'use_tanh_scaling': self.force_torque_sensor.use_tanh_scaling,
                'tanh_scale': self.force_torque_sensor.tanh_scale
            },
            'observation_manager': {
                'enabled': self.observation_manager_enabled,
                'merge_strategy': self.observation_manager_merge_strategy
            },
            'observation_noise': {
                'enabled': self.observation_noise.enabled,
                'global_scale': self.observation_noise.global_scale,
                'apply_to_critic': self.observation_noise.apply_to_critic,
                'seed': self.observation_noise.seed
            },
            'hybrid_control': {
                'enabled': self.hybrid_control.enabled,
                'reward_type': self.hybrid_control.reward_type
            },
            'factory_metrics': {
                'enabled': self.factory_metrics_enabled
            },
            'wandb_logging': {
                'enabled': self.wandb_logging.enabled,
                'wandb_project': self.wandb_logging.wandb_project,
                'wandb_entity': self.wandb_logging.wandb_entity,
                'wandb_name': self.wandb_logging.wandb_name,
                'wandb_group': self.wandb_logging.wandb_group,
                'wandb_tags': self.wandb_logging.wandb_tags
            },
            'action_logging': {
                'enabled': self.action_logging.enabled,
                'track_selection': self.action_logging.track_selection,
                'track_pos': self.action_logging.track_pos,
                'track_rot': self.action_logging.track_rot,
                'track_force': self.action_logging.track_force,
                'track_torque': self.action_logging.track_torque,
                'force_size': self.action_logging.force_size,
                'logging_frequency': self.action_logging.logging_frequency
            }
        }
```

### Phase 3: Update Configuration Loading Logic

#### Step 3.1: Simplify Model Configuration Loading
**File**: `configs/config_manager_v2.py` - Update `_create_model_config` method
```python
@staticmethod
def _create_model_config(yaml_config: Dict[str, Any], primary_cfg: PrimaryConfig) -> ExtendedModelConfig:
    """Create model configuration with nested structure support."""

    # Start with Isaac Lab defaults
    model_cfg = ExtendedModelConfig()

    if 'model' in yaml_config:
        model_data = yaml_config['model']

        # Handle nested actor configuration
        if 'actor' in model_data:
            actor_data = model_data['actor']
            model_cfg.actor = ActorConfig(
                n=actor_data.get('n', model_cfg.actor.n),
                latent_size=actor_data.get('latent_size', model_cfg.actor.latent_size)
            )
            print(f"[CONFIG V2]: Applied actor config: n={model_cfg.actor.n}, latent_size={model_cfg.actor.latent_size}")

        # Handle nested critic configuration
        if 'critic' in model_data:
            critic_data = model_data['critic']
            model_cfg.critic = CriticConfig(
                n=critic_data.get('n', model_cfg.critic.n),
                latent_size=critic_data.get('latent_size', model_cfg.critic.latent_size)
            )
            print(f"[CONFIG V2]: Applied critic config: n={model_cfg.critic.n}, latent_size={model_cfg.critic.latent_size}")

        # Handle nested hybrid_agent configuration
        if 'hybrid_agent' in model_data:
            hybrid_data = model_data['hybrid_agent']
            if model_cfg.hybrid_agent is None:
                model_cfg.hybrid_agent = ExtendedHybridAgentConfig()

            # Apply hybrid agent overrides using existing logic
            for key, value in hybrid_data.items():
                if hasattr(model_cfg.hybrid_agent, key):
                    setattr(model_cfg.hybrid_agent, key, value)
                    print(f"[CONFIG V2]: Applied hybrid_agent.{key} = {value}")

        # Handle flat model configuration
        flat_configs = {k: v for k, v in model_data.items()
                       if k not in ['actor', 'critic', 'hybrid_agent']}

        for key, value in flat_configs.items():
            if hasattr(model_cfg, key):
                old_value = getattr(model_cfg, key)
                setattr(model_cfg, key, value)
                print(f"[CONFIG V2]: Applied model.{key}: {old_value} -> {value}")
            else:
                print(f"\033[93m[CONFIG V2]: Warning: ExtendedModelConfig has no attribute '{key}' (skipping)\033[0m")

    # Validate configuration
    model_cfg.validate()

    return model_cfg
```

#### Step 3.2: Simplify Wrapper Configuration Loading
**File**: `configs/config_manager_v2.py` - Update `_create_wrapper_config` method
```python
@staticmethod
def _create_wrapper_config(yaml_config: Dict[str, Any], primary_cfg: PrimaryConfig) -> ExtendedWrapperConfig:
    """Create wrapper configuration with nested structure support."""

    # Start with Isaac Lab defaults
    wrapper_cfg = ExtendedWrapperConfig()

    if 'wrappers' in yaml_config:
        wrapper_data = yaml_config['wrappers']

        # Handle nested wrapper configurations
        if 'force_torque_sensor' in wrapper_data:
            fts_data = wrapper_data['force_torque_sensor']
            wrapper_cfg.force_torque_sensor = ForceTorqueSensorConfig(
                enabled=fts_data.get('enabled', wrapper_cfg.force_torque_sensor.enabled),
                use_tanh_scaling=fts_data.get('use_tanh_scaling', wrapper_cfg.force_torque_sensor.use_tanh_scaling),
                tanh_scale=fts_data.get('tanh_scale', wrapper_cfg.force_torque_sensor.tanh_scale)
            )
            print(f"[CONFIG V2]: Applied force_torque_sensor config: enabled={wrapper_cfg.force_torque_sensor.enabled}")

        if 'hybrid_control' in wrapper_data:
            hc_data = wrapper_data['hybrid_control']
            wrapper_cfg.hybrid_control = HybridControlConfig(
                enabled=hc_data.get('enabled', wrapper_cfg.hybrid_control.enabled),
                reward_type=hc_data.get('reward_type', wrapper_cfg.hybrid_control.reward_type)
            )
            print(f"[CONFIG V2]: Applied hybrid_control config: enabled={wrapper_cfg.hybrid_control.enabled}")

        if 'observation_noise' in wrapper_data:
            on_data = wrapper_data['observation_noise']
            wrapper_cfg.observation_noise = ObservationNoiseConfig(
                enabled=on_data.get('enabled', wrapper_cfg.observation_noise.enabled),
                global_scale=on_data.get('global_scale', wrapper_cfg.observation_noise.global_scale),
                apply_to_critic=on_data.get('apply_to_critic', wrapper_cfg.observation_noise.apply_to_critic),
                seed=on_data.get('seed', wrapper_cfg.observation_noise.seed)
            )
            print(f"[CONFIG V2]: Applied observation_noise config: enabled={wrapper_cfg.observation_noise.enabled}")

        if 'wandb_logging' in wrapper_data:
            wl_data = wrapper_data['wandb_logging']
            wrapper_cfg.wandb_logging = WandbLoggingConfig(
                enabled=wl_data.get('enabled', wrapper_cfg.wandb_logging.enabled),
                wandb_project=wl_data.get('wandb_project', wrapper_cfg.wandb_logging.wandb_project),
                wandb_entity=wl_data.get('wandb_entity', wrapper_cfg.wandb_logging.wandb_entity),
                wandb_name=wl_data.get('wandb_name', wrapper_cfg.wandb_logging.wandb_name),
                wandb_group=wl_data.get('wandb_group', wrapper_cfg.wandb_logging.wandb_group),
                wandb_tags=wl_data.get('wandb_tags', wrapper_cfg.wandb_logging.wandb_tags)
            )
            print(f"[CONFIG V2]: Applied wandb_logging config: enabled={wrapper_cfg.wandb_logging.enabled}")

        if 'action_logging' in wrapper_data:
            al_data = wrapper_data['action_logging']
            wrapper_cfg.action_logging = ActionLoggingConfig(
                enabled=al_data.get('enabled', wrapper_cfg.action_logging.enabled),
                track_selection=al_data.get('track_selection', wrapper_cfg.action_logging.track_selection),
                track_pos=al_data.get('track_pos', wrapper_cfg.action_logging.track_pos),
                track_rot=al_data.get('track_rot', wrapper_cfg.action_logging.track_rot),
                track_force=al_data.get('track_force', wrapper_cfg.action_logging.track_force),
                track_torque=al_data.get('track_torque', wrapper_cfg.action_logging.track_torque),
                force_size=al_data.get('force_size', wrapper_cfg.action_logging.force_size),
                logging_frequency=al_data.get('logging_frequency', wrapper_cfg.action_logging.logging_frequency)
            )
            print(f"[CONFIG V2]: Applied action_logging config: enabled={wrapper_cfg.action_logging.enabled}")

        # Handle flat wrapper configurations
        flat_configs = {k: v for k, v in wrapper_data.items()
                       if k not in ['force_torque_sensor', 'hybrid_control', 'observation_noise',
                                   'wandb_logging', 'action_logging']}

        for key, value in flat_configs.items():
            flat_attr_map = {
                'fragile_objects': 'fragile_objects_enabled',
                'efficient_reset': 'efficient_reset_enabled',
                'observation_manager': 'observation_manager_enabled',
                'factory_metrics': 'factory_metrics_enabled'
            }

            if key in flat_attr_map:
                attr_name = flat_attr_map[key]
                if isinstance(value, dict) and 'enabled' in value:
                    setattr(wrapper_cfg, attr_name, value['enabled'])
                    print(f"[CONFIG V2]: Applied {key}.enabled = {value['enabled']}")
            else:
                print(f"\033[93m[CONFIG V2]: Warning: Unknown wrapper config '{key}' (skipping)\033[0m")

    return wrapper_cfg
```

### Phase 4: Update CLI Override Handling

#### Step 4.1: Enhanced CLI Override Support
**File**: `configs/config_manager_v2.py` - Update `_apply_cli_overrides` method
```python
@staticmethod
def _apply_cli_overrides(config_bundle: ConfigBundle, cli_overrides: List[str]) -> None:
    """Apply CLI overrides with support for nested configurations."""

    for override in cli_overrides:
        if '=' not in override:
            print(f"\033[93m[CONFIG V2]: Warning: Invalid override format '{override}' (expected key=value)\033[0m")
            continue

        key, value = override.split('=', 1)
        parsed_value = ConfigManagerV2._parse_override_value(value)

        # Handle nested model overrides
        if key.startswith('model.actor.'):
            param = key.replace('model.actor.', '')
            if hasattr(config_bundle.model_cfg.actor, param):
                setattr(config_bundle.model_cfg.actor, param, parsed_value)
                print(f"[CONFIG V2]: CLI override model.actor.{param} = {parsed_value}")
            else:
                print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

        elif key.startswith('model.critic.'):
            param = key.replace('model.critic.', '')
            if hasattr(config_bundle.model_cfg.critic, param):
                setattr(config_bundle.model_cfg.critic, param, parsed_value)
                print(f"[CONFIG V2]: CLI override model.critic.{param} = {parsed_value}")
            else:
                print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

        elif key.startswith('model.hybrid_agent.'):
            param = key.replace('model.hybrid_agent.', '')
            if config_bundle.model_cfg.hybrid_agent is None:
                config_bundle.model_cfg.hybrid_agent = ExtendedHybridAgentConfig()
            if hasattr(config_bundle.model_cfg.hybrid_agent, param):
                setattr(config_bundle.model_cfg.hybrid_agent, param, parsed_value)
                print(f"[CONFIG V2]: CLI override model.hybrid_agent.{param} = {parsed_value}")
            else:
                print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

        elif key.startswith('wrappers.force_torque_sensor.'):
            param = key.replace('wrappers.force_torque_sensor.', '')
            if hasattr(config_bundle.wrapper_cfg.force_torque_sensor, param):
                setattr(config_bundle.wrapper_cfg.force_torque_sensor, param, parsed_value)
                print(f"[CONFIG V2]: CLI override wrappers.force_torque_sensor.{param} = {parsed_value}")
            else:
                print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

        # Add similar handling for other nested wrapper configs...

        # Handle existing flat overrides (primary, model flat attrs, etc.)
        else:
            # Use existing logic for flat overrides
            ConfigManagerV2._apply_flat_cli_override(config_bundle, key, parsed_value)
```

### Phase 5: Update Legacy Conversion

#### Step 5.1: Simplify Legacy Config Dict Generation
**File**: `configs/config_manager_v2.py` - Update `get_legacy_config_dict` method
```python
@staticmethod
def get_legacy_config_dict(config_bundle: ConfigBundle) -> Dict[str, Any]:
    """Convert configuration bundle to legacy dictionary format."""

    return {
        'primary': {
            'agents_per_break_force': config_bundle.primary_cfg.agents_per_break_force,
            'num_envs_per_agent': config_bundle.primary_cfg.num_envs_per_agent,
            'break_forces': config_bundle.primary_cfg.break_forces,
            'decimation': config_bundle.primary_cfg.decimation,
            'policy_hz': config_bundle.primary_cfg.policy_hz,
            'max_steps': config_bundle.primary_cfg.max_steps,
            'debug_mode': config_bundle.primary_cfg.debug_mode,
            'seed': config_bundle.primary_cfg.seed,
            'ckpt_tracker_path': config_bundle.primary_cfg.ckpt_tracker_path,
            'ctrl_torque': config_bundle.primary_cfg.ctrl_torque
        },
        'derived': {
            'total_agents': config_bundle.primary_cfg.total_agents,
            'total_num_envs': config_bundle.primary_cfg.total_num_envs,
            'rollout_steps': config_bundle.env_cfg.get_rollout_steps(),
            'max_steps': config_bundle.primary_cfg.max_steps,
            'sim_dt': config_bundle.primary_cfg.sim_dt,
            'ckpt_tracker_path': config_bundle.primary_cfg.ckpt_tracker_path,
            'seed': config_bundle.primary_cfg.seed
        },
        'environment': {
            'decimation': config_bundle.env_cfg.decimation,
            'episode_length_s': config_bundle.env_cfg.episode_length_s
        },
        'model': config_bundle.model_cfg.to_dict(),  # Now produces proper nested structure
        'wrappers': config_bundle.wrapper_cfg.to_dict(),  # Now produces proper nested structure
        'agent': config_bundle.agent_cfg.to_skrl_dict(config_bundle.env_cfg.episode_length_s),
        'experiment': {
            'name': config_bundle.agent_cfg.experiment_name,
            'tags': config_bundle.agent_cfg.wandb_tags,
            'group': config_bundle.agent_cfg.wandb_group,
            'wandb_project': config_bundle.agent_cfg.wandb_project
        }
    }
```

### Phase 6: Testing and Validation

#### Step 6.1: Update Unit Tests
**Files**: `tests/unit/test_extended_configs.py`
- Update tests to work with nested structure
- Add tests for nested dataclass validation
- Test that `to_dict()` produces expected nested structure

#### Step 6.2: Integration Testing
- Test that `model_config['actor']['n']` works in launch_utils
- Test CLI overrides like `model.actor.latent_size=512`
- Verify YAML configs load correctly with nested structure

#### Step 6.3: Migration Testing
- Test existing V2 config files work unchanged
- Verify all warnings are eliminated
- Test hybrid agent configuration works properly

## Expected Outcomes

### Before Implementation (Current Issues):
```
[CONFIG V2]: Warning: ExtendedModelConfig has no attribute 'actor' (skipping)
[CONFIG V2]: Warning: ExtendedModelConfig has no attribute 'critic' (skipping)
KeyError: 'actor'  # in launch_utils
```

### After Implementation (Expected Success):
```
[CONFIG V2]: Applied actor config: n=1, latent_size=256
[CONFIG V2]: Applied critic config: n=3, latent_size=1024
[CONFIG V2]: Applied force_torque_sensor config: enabled=True
[INFO]: Creating hybrid control agent models
[INFO]: Hybrid agent parameters configured
# No errors - launch_utils gets model_config['actor']['n'] successfully
```

## Benefits of This Solution

1. **Maintains Defaults-First**: Isaac Lab defaults built into dataclass constructors
2. **Natural YAML Structure**: Configs match intuitive nested YAML format
3. **Type Safety**: Full IDE support and validation for all nested configs
4. **No Conversion Errors**: `to_dict()` produces exactly what launch_utils expects
5. **Better CLI Support**: Natural paths like `model.actor.latent_size=512`
6. **Extensible**: Easy to add new nested sections
7. **Backward Compatible**: Existing V2 YAML files continue to work

## Implementation Priority

**High Priority**: Steps 2.1, 2.2, 3.1, 3.2 - Core dataclass and loading changes
**Medium Priority**: Step 4.1 - Enhanced CLI override support
**Low Priority**: Step 6.1, 6.2, 6.3 - Testing updates

The core fix (eliminating warnings and KeyError) will be achieved after completing the high-priority steps.