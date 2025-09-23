# Configuration System Migration Guide

This guide helps you migrate from the legacy reference-based configuration system to the new ConfigManagerV2 defaults-first system.

## Quick Migration Checklist

- [ ] Remove all `${...}` reference syntax
- [ ] Add explicit `task_name` to configuration root
- [ ] Convert `defaults` section values to direct assignments
- [ ] Remove manual `derived` section calculations
- [ ] Update any custom code to use new API
- [ ] Test with validation script
- [ ] Verify training equivalence

## Step-by-Step Migration

### Step 1: Update Configuration Files

**Before (Legacy format):**
```yaml
defaults:
  task_name: "Isaac-Factory-PegInsert-Direct-v0"
  episode_length_s: 10.0

primary:
  agents_per_break_force: 2
  num_envs_per_agent: 256
  decimation: 8

derived:
  total_agents: ${primary.agents_per_break_force}
  total_num_envs: ${__calc_envs:}
  rollout_steps: ${__calc_rollout:}

environment:
  episode_length_s: ${defaults.episode_length_s}
  decimation: ${primary.decimation}

model:
  actor_latent_size: 256

wrappers:
  force_torque_sensor:
    enabled: ${model.use_hybrid_agent}
```

**After (V2 format):**
```yaml
task_name: "Isaac-Factory-PegInsert-Direct-v0"

primary:
  agents_per_break_force: 2
  num_envs_per_agent: 256
  decimation: 8
  # total_agents computed automatically
  # total_num_envs computed automatically
  # rollout_steps computed automatically

environment:
  # episode_length_s comes from Isaac Lab defaults (10.0s)
  # decimation applied automatically from primary

model:
  actor:
    latent_size: 256

wrappers:
  force_torque_sensor:
    enabled: false  # Set explicit value
```

### Step 2: Update Code Usage

**Before (Legacy API):**
```python
from configs.config_manager import ConfigManager

resolved_config = ConfigManager.load_and_resolve_config(
    config_path,
    cli_overrides
)

primary = resolved_config['primary']
derived = resolved_config['derived']
environment = resolved_config['environment']
agent_config = resolved_config['agent']

print(f"Total agents: {derived['total_agents']}")
print(f"Episode length: {environment['episode_length_s']}")
```

**After (V2 API):**
```python
from configs.config_manager_v2 import ConfigManagerV2

# Load with new system
bundle = ConfigManagerV2.load_defaults_first_config(
    config_path=config_path,
    cli_overrides=cli_overrides
)

# Access typed configuration objects
primary = bundle.primary_cfg
environment = bundle.env_cfg
agent_config = bundle.agent_cfg

print(f"Total agents: {primary.total_agents}")  # Computed property
print(f"Episode length: {environment.episode_length_s}")

# OR convert to legacy format for compatibility
legacy_dict = ConfigManagerV2.get_legacy_config_dict(bundle)
primary = legacy_dict['primary']
derived = legacy_dict['derived']  # Contains computed values
environment = legacy_dict['environment']
```

### Step 3: Remove Reference Patterns

Find and replace these patterns:

| Legacy Pattern | V2 Replacement |
|----------------|-----------------|
| `${primary.decimation}` | Move to `primary:` section, auto-applied |
| `${defaults.task_name}` | Use root-level `task_name:` |
| `${defaults.episode_length_s}` | Remove (comes from Isaac Lab defaults) |
| `${derived.total_agents}` | Use computed property `primary_cfg.total_agents` |
| `${__calc_*:}` | Remove (computed automatically) |

### Step 4: Handle Special Cases

#### Hybrid Agent Configuration
**Before:**
```yaml
model:
  use_hybrid_agent: ${wrappers.hybrid_control.enabled}

wrappers:
  hybrid_control:
    enabled: true
  force_torque_sensor:
    enabled: ${model.use_hybrid_agent}
```

**After:**
```yaml
model:
  use_hybrid_agent: true
  hybrid_agent:
    ctrl_torque: false
    pos_scale: 1.0

wrappers:
  hybrid_control:
    enabled: true
  force_torque_sensor:
    enabled: true  # Explicit value
```

#### Multi-Agent Training
**Before:**
```yaml
primary:
  agents_per_break_force: 2
  break_forces: [50, 100, 150]

derived:
  total_agents: ${__calc_agents:}

wrappers:
  fragile_objects:
    num_agents: ${derived.total_agents}
    break_force: ${primary.break_forces}
```

**After:**
```yaml
primary:
  agents_per_break_force: 2
  break_forces: [50, 100, 150]
  # total_agents = 2 * 3 = 6 (computed automatically)

wrappers:
  fragile_objects:
    enabled: true
    # num_agents and break_force applied automatically
```

## Migration Tools

### Validation Script

Test that your migrated configuration produces the same results:

```bash
# Validate single config
python scripts/validate_config_migration.py --config your_config_v2.yaml

# Validate all configs
python scripts/validate_config_migration.py --all-configs

# Generate report
python scripts/validate_config_migration.py --config your_config_v2.yaml --output report.txt
```

### Feature Flag Testing

Test both systems side-by-side:

```bash
# Test legacy system
USE_NEW_CONFIG=false python learning/factory_runnerv2.py --config config.yaml

# Test new system
USE_NEW_CONFIG=true python learning/factory_runnerv2.py --config config_v2.yaml

# Compare results
```

## Common Migration Issues

### Issue 1: Missing task_name
**Error:** `ValueError: No task name specified in CLI or YAML config`

**Solution:** Add explicit task name to config root:
```yaml
task_name: "Isaac-Factory-PegInsert-Direct-v0"  # Add this line
```

### Issue 2: Invalid parameter values
**Error:** `ConfigurationError: decimation must be positive, got: 0`

**Solution:** Check all parameter values in your config:
```yaml
primary:
  decimation: 8  # Must be positive integer
  agents_per_break_force: 2  # Must be positive
```

### Issue 3: Hybrid agent config missing
**Error:** `ValueError: Model config specifies use_hybrid_agent=True but no hybrid config provided`

**Solution:** Add hybrid agent configuration:
```yaml
model:
  use_hybrid_agent: true
  hybrid_agent:  # Add this section
    ctrl_torque: false
    pos_scale: 1.0
    force_scale: 1.0
```

### Issue 4: Wrapper configuration format
**Error:** Warnings about missing wrapper attributes

**Solution:** Use flat attribute names for wrapper config:
```yaml
wrappers:
  force_torque_sensor:
    enabled: true
    tanh_scale: 0.05
  # Instead of nested wrapper objects
```

## Testing Your Migration

### 1. Configuration Loading Test
```python
from configs.config_manager_v2 import ConfigManagerV2

try:
    bundle = ConfigManagerV2.load_defaults_first_config("your_config_v2.yaml")
    print("✅ Configuration loads successfully")
    print(f"Task: {bundle.task_name}")
    print(f"Total agents: {bundle.primary_cfg.total_agents}")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

### 2. Training Pipeline Test
```bash
# Short training run to verify pipeline works
USE_NEW_CONFIG=true python learning/factory_runnerv2.py \
  --config your_config_v2.yaml \
  --override "primary.max_steps=1000"
```

### 3. Comparison Test
```bash
# Run validation script
python scripts/validate_config_migration.py \
  --config your_config_v2.yaml \
  --output validation_report.txt

# Check the report for differences
cat validation_report.txt
```

## Rollout Strategy

### Phase 1: Individual Config Migration
1. Migrate one config file at a time
2. Test with validation script
3. Fix any issues found
4. Verify training equivalence

### Phase 2: Environment Testing
1. Test in development environment
2. Run short training experiments
3. Compare metrics with legacy system
4. Test all wrapper combinations

### Phase 3: Production Migration
1. Update production configs
2. Monitor training closely
3. Have rollback plan ready
4. Document any issues

## Rollback Plan

If you encounter issues with the new system:

```bash
# Immediate rollback to legacy system
export USE_NEW_CONFIG=false

# Or revert configuration files to legacy format
git checkout legacy_configs/
```

## Support

- **Documentation**: See `docs/CONFIGURATION_SYSTEM_V2.md`
- **Validation**: Use `scripts/validate_config_migration.py`
- **Issues**: Report issues with detailed error messages
- **Examples**: Check `configs/test/simple_migration_test.yaml`

## Success Checklist

Migration is complete when:

- [ ] All reference syntax (`${...}`) removed
- [ ] Configuration loads without errors
- [ ] Validation script reports success
- [ ] Training pipeline works correctly
- [ ] Computed values match expected results
- [ ] All team members trained on new system
- [ ] Documentation updated

Your migration is successful when the new system produces identical training results with improved maintainability and error handling!