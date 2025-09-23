# Configuration System Refactor Plan

## Overview

This document outlines the complete plan for refactoring the configuration system from a reference-based approach to a defaults-first architecture. The goal is to create a robust, maintainable system where every parameter has clear defaults from Isaac Lab/SKRL, with explicit override hierarchy and no silent failures.

## Current Problems

- Reference resolution fails when values are missing (`${primary.episode_length_s}`)
- Complex reference resolution logic prone to errors
- No clear fallback to Isaac Lab/SKRL defaults
- Difficult to track where parameter values come from
- Silent failures and unclear error messages

## Target Architecture

### Defaults-First Loading Order
```
1. Isaac Lab defaults (task-specific configs)
2. SKRL PPO defaults
3. PrimaryConfig overrides
4. Local YAML overrides
5. CLI argument overrides
```

### Extended Configuration Classes
- Create extended versions of all Isaac Lab configs in `cfg_exts/`
- Use computed properties for derived values (total_agents, rollout_steps)
- Extended configs reference other extended configs by default
- SKRL config extensions live in `agents/` folder following SKRL patterns

### Error Handling Philosophy
- No quiet failures anywhere
- Throw explicit errors for unknown task names, invalid parameters
- Clear, actionable error messages
- Fail fast rather than using fallbacks

## Implementation Plan

### Phase 1: Foundation Setup & Architecture

#### 1.1 Rename and restructure folder
```bash
mv configs/isaac_lab_extensions/ configs/cfg_exts/
```
- Update all imports throughout codebase with proper error handling
- Add `__init__.py` with version checking and import validation

#### 1.2 Create PrimaryConfig dataclass
File: `cfg_exts/primary_cfg.py`

```python
@configclass
class PrimaryConfig:
    agents_per_break_force: int = 2
    num_envs_per_agent: int = 256
    break_forces: Union[int, List[int]] = -1
    decimation: int = 8
    policy_hz: int = 15

    @property
    def total_agents(self) -> int:
        num_forces = len(self.break_forces) if isinstance(self.break_forces, list) else 1
        return num_forces * self.agents_per_break_force

    @property
    def total_num_envs(self) -> int:
        return self.total_agents * self.num_envs_per_agent

    @property
    def rollout_steps(self) -> int:
        sim_dt = (1/self.policy_hz) / self.decimation
        return int((1/sim_dt) / self.decimation * episode_length_s)

    def __post_init__(self):
        # Validation logic with clear error messages
```

#### 1.3 Create extended Isaac Lab configs
Files to create:
- `cfg_exts/extended_factory_env_cfg.py` - extends `FactoryEnvCfg`
- `cfg_exts/extended_peg_insert_cfg.py` - extends `FactoryTaskPegInsertCfg`
- `cfg_exts/extended_gear_mesh_cfg.py` - extends `FactoryTaskGearMeshCfg`
- `cfg_exts/extended_nut_thread_cfg.py` - extends `FactoryTaskNutThreadCfg`

Each with:
- Computed properties for derived calculations
- Robust error handling for missing dependencies
- Type hints and validation

#### 1.4 Create extended model and wrapper configs
- `cfg_exts/extended_model_cfg.py` - SimBa architecture parameters
- `cfg_exts/extended_wrapper_cfg.py` - wrapper configurations

#### 1.5 Create SKRL config extension
File: `agents/extended_ppo_cfg.py`
- Wrapper around SKRL's PPO_DEFAULT_CONFIG
- Computed properties for rollout/batch calculations
- Block PPO specific parameters

### Phase 2: Robust ConfigManager Refactor

#### 2.1 Create task resolution system
```python
def resolve_task_name(yaml_cfg: Dict, cli_task: Optional[str]) -> str:
    # CLI takes precedence, then YAML, then error
    if cli_task:
        return cli_task
    if 'task_name' in yaml_cfg:
        return yaml_cfg['task_name']
    raise ValueError("No task name specified in CLI or YAML config")

def load_extended_cfg_for_task(task_name: str) -> Tuple[ExtendedEnvCfg, ExtendedTaskCfg]:
    # Map task names to extended config classes with validation
```

#### 2.2 Create robust merging system
```python
def merge_primary_cfg(target_cfg: Any, primary_cfg: PrimaryConfig) -> None:
    # Deep merge with type checking and validation

def apply_yaml_overrides(target_cfg: Any, yaml_dict: Dict, path: str = "") -> None:
    # Apply only specified values with nested object support
    # Validate all overrides against target config schema

def apply_cli_overrides(target_cfg: Any, cli_overrides: List[str]) -> None:
    # Support nested paths like "environment.decimation=4"
```

#### 2.3 Create comprehensive error handling
- Explicit error messages for every failure mode
- Validation of all parameter types and ranges
- Clear indication of where each parameter value came from

#### 2.4 Create new main loading method
```python
def load_defaults_first_cfg(cfg_path: str, cli_overrides: List[str], cli_task: str) -> ConfigBundle:
    # 1. Resolve task name (with error handling)
    # 2. Load extended configs with Isaac Lab/SKRL defaults
    # 3. Create and merge PrimaryConfig
    # 4. Apply YAML overrides (with validation)
    # 5. Apply CLI overrides (with validation)
    # 6. Return structured config bundle
```

### Phase 3: YAML Simplification & Validation

#### 3.1 Create YAML validation system
- Schema validation for each config section
- Automatic detection of remaining reference syntax
- Validation that all overrides target valid config attributes

#### 3.2 Update base configuration
File: `configs/base/factory_base.yaml` (simplified)
```yaml
task_name: "Isaac-Factory-PegInsert-Direct-v0"

# Direct Isaac Lab environment overrides only
environment:
  decimation: 8
  filter_collisions: true

# Model configuration (direct values only)
model:
  actor:
    latent_size: 256
  critic:
    latent_size: 1024

# Wrapper enables/disables and direct parameters
wrappers:
  force_torque_sensor:
    enabled: false
    tanh_scale: 0.03
```

#### 3.3 Update experiment configurations
- Remove ALL reference syntax (`${primary.episode_length_s}`)
- Add explicit task_name to each config
- Validate each config produces same results as current system

### Phase 4: Comprehensive Testing Strategy

#### 4.1 Create unit tests for each component
Test files:
- `tests/unit/test_extended_configs.py`
- `tests/unit/test_primary_cfg.py`
- `tests/unit/test_config_manager_v2.py`
- `tests/unit/test_yaml_validation.py`

Test coverage:
- Each extended config loads with proper defaults
- Computed properties with edge cases
- Merging logic with various input types
- Error handling for all failure modes

#### 4.2 Create integration tests
File: `tests/integration/test_config_equivalence.py`
- Load same config with old and new systems
- Compare all final parameter values
- Ensure derived values match exactly
- Test with all existing config files

#### 4.3 Create performance tests
- Ensure computed properties are efficient
- Compare loading times old vs new system
- Memory usage analysis

### Phase 5: Production Integration

#### 5.1 Update factory runner with feature flag
```python
# Allow switching between old and new systems for validation
use_new_config_system = os.getenv('USE_NEW_CONFIG', 'false').lower() == 'true'
```

#### 5.2 Create migration validation script
File: `scripts/validate_config_migration.py`
- Load same config with both systems, compare all outputs
- Run on all existing config files
- Generate migration report

#### 5.3 Gradual rollout
- Test with simple configs first
- Validate training runs produce same results
- Full migration only after validation passes

### Phase 6: Cleanup & Documentation

#### 6.1 Remove deprecated code
- Delete old reference resolution methods only after validation
- Remove unused imports and methods
- Clean up any temporary compatibility code

#### 6.2 Create comprehensive documentation
- Architecture overview with diagrams
- Migration guide for new configs
- Troubleshooting guide
- Performance characteristics

#### 6.3 Create developer tools
- Config validation CLI tool
- Config comparison utility
- Config generation templates

## Implementation Quality Standards

### Error Handling
- Every potential failure point has explicit error handling
- Clear, actionable error messages
- No silent failures anywhere in the system

### Type Safety
- Complete type hints throughout
- Runtime type validation for critical paths
- mypy compliance

### Performance
- Computed properties cached where appropriate
- Efficient merging algorithms
- Memory-conscious design

### Testing
- >95% code coverage
- All edge cases tested
- Performance regression testing
- Error case testing

### Documentation
- Every public method documented
- Architecture decisions explained
- Examples for common use cases

## File Structure After Refactor

```
configs/
├── cfg_exts/
│   ├── __init__.py
│   ├── primary_cfg.py
│   ├── extended_factory_env_cfg.py
│   ├── extended_peg_insert_cfg.py
│   ├── extended_gear_mesh_cfg.py
│   ├── extended_nut_thread_cfg.py
│   ├── extended_ctrl_cfg.py (existing)
│   ├── extended_model_cfg.py
│   ├── extended_wrapper_cfg.py
│   └── version_compat.py (existing)
├── base/
│   └── factory_base.yaml (simplified)
├── experiments/
│   ├── hybrid_control_exp.yaml (updated)
│   └── debug_exp.yaml (updated)
└── config_manager.py (refactored)

agents/
├── block_ppo.py (existing)
└── extended_ppo_cfg.py (new)

tests/
├── unit/
│   ├── test_extended_configs.py
│   ├── test_primary_cfg.py
│   ├── test_config_manager_v2.py
│   └── test_yaml_validation.py
└── integration/
    └── test_config_equivalence.py

scripts/
└── validate_config_migration.py
```

## Success Criteria

1. **Functional equivalence**: New system produces identical results to current system
2. **Error transparency**: All failure modes produce clear, actionable error messages
3. **Performance**: Loading times equal to or better than current system
4. **Maintainability**: Adding new parameters requires minimal code changes
5. **Robustness**: System handles edge cases and invalid inputs gracefully
6. **Documentation**: Complete documentation for maintenance and extension

## Risk Mitigation

- **Backward compatibility**: Not maintaining old YAML format - clean break
- **Validation**: Comprehensive testing before production deployment
- **Rollback**: Keep old system available during transition period
- **Performance**: Benchmark all changes against current system
- **Training continuity**: Ensure existing checkpoints and experiments continue working