# Configuration System Gradual Rollout Guide

This document outlines the strategy for gradually rolling out the new ConfigManagerV2 system to replace the legacy configuration system.

## Rollout Phases

### Phase 1: Initial Testing (Current Phase)
**Status**: Ready for testing
**Timeline**: 1-2 days

**Objectives:**
- Validate that the new system loads without errors
- Confirm basic functionality with simple configurations
- Verify the feature flag system works correctly

**Test Configurations:**
1. `configs/test/simple_migration_test.yaml` - Minimal test config
2. `configs/base/factory_base_v2.yaml` - Standard base config
3. `configs/experiments/hybrid_control_exp_v2.yaml` - Complex hybrid config

**Testing Steps:**
```bash
# Test 1: Legacy system (baseline)
python learning/factory_runnerv2.py --config configs/test/simple_migration_test.yaml

# Test 2: New system (comparison)
USE_NEW_CONFIG=true python learning/factory_runnerv2.py --config configs/test/simple_migration_test.yaml

# Test 3: Validation script
python scripts/validate_config_migration.py --config configs/test/simple_migration_test.yaml
```

**Success Criteria:**
- ✅ New system loads configuration without errors
- ✅ All unit tests pass
- ✅ Basic training loop can start (first few steps)
- ⚠️ Configuration differences are documented and acceptable

### Phase 2: Extended Validation
**Status**: Next phase
**Timeline**: 3-5 days

**Objectives:**
- Run short training experiments with both systems
- Compare training metrics and convergence
- Test with various CLI overrides
- Validate checkpoint loading/saving

**Test Configurations:**
- All configs in `configs/base/` directory
- Key experiment configs
- Configs with CLI overrides

**Testing Steps:**
```bash
# Run parallel training experiments
USE_NEW_CONFIG=false python learning/factory_runnerv2.py --config CONFIG_FILE &
USE_NEW_CONFIG=true python learning/factory_runnerv2.py --config CONFIG_FILE &

# Compare results after 1000 steps
python scripts/compare_training_results.py
```

**Success Criteria:**
- Training metrics are within 5% variance
- Checkpoint files are compatible
- No performance regression (>10% slower)

### Phase 3: Production Pilot
**Status**: Future phase
**Timeline**: 1-2 weeks

**Objectives:**
- Run complete training experiments with new system
- Monitor for any issues in production environment
- Gather performance data

**Test Configurations:**
- Production experiment configurations
- Multi-agent training setups
- Long-running experiments (>1M steps)

**Success Criteria:**
- No training failures or errors
- Performance equal to or better than legacy system
- All features work as expected

### Phase 4: Full Migration
**Status**: Future phase
**Timeline**: 1 week

**Objectives:**
- Switch default to new system
- Update documentation
- Remove legacy system

**Actions:**
- Change default `USE_NEW_CONFIG=true`
- Update all scripts and documentation
- Archive legacy configuration code

## Testing Checklist

### Pre-Rollout Validation
- [ ] All unit tests pass (Phase 4 complete)
- [ ] Integration tests pass
- [ ] Migration validation script runs successfully
- [ ] Feature flag system works correctly
- [ ] Documentation is updated

### Phase 1 Validation
- [ ] New system loads configurations without errors
- [ ] Basic training loop starts successfully
- [ ] CLI overrides work correctly
- [ ] Environment creation succeeds
- [ ] Agent initialization succeeds

### Phase 2 Validation
- [ ] Training metrics match legacy system (±5%)
- [ ] Checkpoint saving/loading works
- [ ] Wandb logging functions correctly
- [ ] Multi-agent training works
- [ ] Performance is acceptable

### Phase 3 Validation
- [ ] Long-running experiments complete successfully
- [ ] No memory leaks or performance degradation
- [ ] All wrapper functionality works
- [ ] Production environment compatibility

## Rollback Plan

If critical issues are discovered during any phase:

1. **Immediate Actions:**
   - Set `USE_NEW_CONFIG=false` to revert to legacy system
   - Document the issue and error details
   - Stop any running experiments using new system

2. **Investigation:**
   - Reproduce the issue in isolated environment
   - Compare behavior with legacy system
   - Identify root cause

3. **Resolution:**
   - Fix the issue in new system
   - Re-run validation tests
   - Resume rollout from appropriate phase

## Environment Variables

Control the rollout using these environment variables:

```bash
# Use new configuration system (default: false)
export USE_NEW_CONFIG=true

# Enable verbose logging for debugging (default: false)
export CONFIG_DEBUG=true

# Force validation on every config load (default: false)
export CONFIG_VALIDATE=true
```

## Monitoring and Metrics

During rollout, monitor these key metrics:

1. **Functionality:**
   - Configuration loading success rate
   - Training initialization success rate
   - Error rates and types

2. **Performance:**
   - Configuration loading time
   - Memory usage
   - Training step time
   - Overall training performance

3. **Compatibility:**
   - Checkpoint compatibility
   - Multi-agent training compatibility
   - Wrapper integration compatibility

## Communication Plan

### Internal Team Updates
- Daily status updates during Phase 1-2
- Weekly status updates during Phase 3-4
- Immediate notification of any critical issues

### Documentation Updates
- Update README.md with new configuration system
- Update training guides and examples
- Create migration guide for users

### Training Team Coordination
- Coordinate with active experiments
- Plan migration windows for minimal disruption
- Provide fallback instructions

## Success Metrics

The rollout is considered successful when:

1. **All phases complete without critical issues**
2. **Performance is equal to or better than legacy system**
3. **All functionality works as expected**
4. **Team is confident in the new system**
5. **Documentation is complete and accurate**

## Current Status

✅ **Phase 1: Foundation Setup** - Complete
✅ **Phase 2: Core ConfigManager Refactor** - Complete
✅ **Phase 3: YAML Simplification & Validation** - Complete
✅ **Phase 4: Unit Testing** - Complete
✅ **Phase 5.1: Factory runner with feature flag** - Complete
✅ **Phase 5.2: Migration validation script** - Complete
🔄 **Phase 5.3: Gradual rollout preparation** - In Progress

**Next Steps:**
1. Run Phase 1 testing with simple configurations
2. Document any configuration differences found
3. Address any critical issues before Phase 2
4. Begin extended validation testing

**Contact:**
For questions or issues during rollout, contact the configuration system team.