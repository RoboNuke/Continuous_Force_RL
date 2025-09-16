# Factory Environment Refactoring Plan

## Overview
This document outlines the plan for eliminating modified files after successful wrapper extraction and testing. The goal is to return to using Isaac Lab's original factory environment with modular wrappers providing all custom functionality.

## Phase 5: File Elimination and Integration

### Files to Eliminate (after successful testing)

#### 1. Custom Environment Files
- **`envs/factory/factory_env.py`** - Modified factory environment
  - **Reason**: Contains custom force-torque, fragile objects, and efficient reset logic
  - **Replacement**: Use Isaac Lab's original factory environment with wrappers
  - **Verification**: Ensure all custom functionality is replicated in wrappers

- **`envs/factory/obs_factory_env.py`** - Observation-enhanced factory environment
  - **Reason**: Subclassing approach being replaced with wrapper composition
  - **Replacement**: Use `HistoryObservationWrapper` + `ObservationManagerWrapper`
  - **Verification**: Test observation format and history functionality

#### 2. Custom Control Files (if applicable)
- **`envs/factory/factory_control.py`** - Modified control functions
  - **Status**: Keep if still needed by other code, or extract remaining functions to wrappers
  - **Action**: Review dependencies and create migration plan
  - **Note**: Core functions already extracted to `factory_control_utils.py`

#### 3. Legacy Logging Files
- **`logging/MultiWandbLoggerPPO.py`** - Custom multi-agent logger
  - **Reason**: Functionality replaced by `WandbLoggingWrapper`
  - **Replacement**: Use generic wandb wrapper with any RL framework
  - **Verification**: Test multi-agent logging and episode tracking

### Files to Keep and Modify

#### 1. Configuration Files
- **`envs/factory/factory_env_cfg.py`** - Keep but clean up
  - **Action**: Remove configuration for eliminated functionality
  - **Keep**: Core Isaac Lab configuration structure
  - **Update**: Add wrapper-specific configuration sections

#### 2. Training Scripts
- **Update training entry points to use new factory utility**
  - Replace custom environment creation with `create_factory_environment()`
  - Update imports to use wrapper factory
  - Test with different presets

### Integration Steps

#### Step 1: Create Migration Examples
```python
# Before (custom environment)
from envs.factory.obs_factory_env import ObsFactoryEnv
env = ObsFactoryEnv(cfg=config)

# After (wrapper composition)
from wrappers.factory import create_factory_environment
env = create_factory_environment(
    env_cfg=config,
    preset="research",
    num_agents=1
)
```

#### Step 2: Validation Testing
- **Unit Tests**: Test each wrapper individually
- **Integration Tests**: Test wrapper combinations
- **Performance Tests**: Compare memory and speed vs original
- **Functionality Tests**: Verify all custom features work

#### Step 3: Gradual Migration
1. **Phase 5a**: Test wrappers alongside existing code
2. **Phase 5b**: Update one training script to use wrappers
3. **Phase 5c**: Migrate all training scripts
4. **Phase 5d**: Remove custom environment files
5. **Phase 5e**: Clean up imports and dependencies

### Verification Checklist

#### Functionality Verification
- [ ] Force-torque sensor data matches original implementation
- [ ] Fragile object break forces work correctly
- [ ] Efficient reset preserves training performance
- [ ] Hybrid control produces expected behavior
- [ ] Multi-agent assignment works correctly
- [ ] Wandb logging captures all metrics
- [ ] History observations maintain temporal relationships
- [ ] Observation format stays consistent

#### Performance Verification
- [ ] Training speed comparable to original
- [ ] Memory usage within acceptable bounds
- [ ] GPU utilization efficient
- [ ] Episode reset times acceptable

#### Integration Verification
- [ ] All wrapper combinations work together
- [ ] Configuration overrides function correctly
- [ ] Multi-agent scenarios stable
- [ ] Error handling robust

### Risk Assessment

#### High Risk
- **Observation format incompatibility**: Could break existing models
  - **Mitigation**: Extensive testing with saved models
- **Performance degradation**: Wrapper overhead could slow training
  - **Mitigation**: Benchmark against original implementation

#### Medium Risk
- **Missing functionality**: Some edge cases might not be covered
  - **Mitigation**: Comprehensive test suite and gradual migration
- **Configuration complexity**: New wrapper system might confuse users
  - **Mitigation**: Clear documentation and examples

#### Low Risk
- **Import path changes**: Code updates needed
  - **Mitigation**: Clear migration guide and examples

### Success Criteria

#### Primary Goals
1. **Functionality Parity**: All custom features work as before
2. **Performance Parity**: Training speed within 5% of original
3. **Code Reduction**: Eliminate at least 3 custom environment files
4. **Modularity**: Each wrapper usable independently

#### Secondary Goals
1. **Documentation**: Complete API documentation for all wrappers
2. **Examples**: Working examples for common use cases
3. **Testing**: 90%+ test coverage for wrapper functionality
4. **Maintenance**: Reduced code complexity and maintenance burden

### Implementation Timeline

#### Week 1: Testing and Validation
- Unit tests for all wrappers
- Integration testing with factory environments
- Performance benchmarking

#### Week 2: Migration Implementation
- Update training scripts to use factory utility
- Create migration examples and documentation
- Test with existing models and checkpoints

#### Week 3: File Elimination
- Remove custom environment files
- Clean up imports and dependencies
- Final integration testing

#### Week 4: Documentation and Polish
- Complete API documentation
- Create usage examples
- Performance optimization if needed

### Rollback Plan

If issues are discovered during migration:

1. **Immediate**: Keep original files as backup during testing
2. **Short-term**: Git branches for each migration phase
3. **Long-term**: Tagged releases before major changes

### Communication Plan

- Document all breaking changes
- Provide clear migration examples
- Update README with new usage patterns
- Create troubleshooting guide for common issues

## Next Steps

1. **Unit Testing**: Create comprehensive tests for each wrapper
2. **Integration Testing**: Test wrapper combinations and edge cases
3. **Performance Testing**: Benchmark against original implementation
4. **Migration**: Update training scripts to use new factory system
5. **Cleanup**: Remove custom files after successful validation

This plan ensures a safe, systematic transition from custom environment files to a modular wrapper-based architecture while maintaining all functionality and performance characteristics.