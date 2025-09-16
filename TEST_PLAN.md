# Factory Environment Wrappers Test Plan

## Overview
This document outlines the comprehensive testing strategy for all factory environment wrappers. Tests are organized by wrapper type and include unit tests, integration tests, and performance benchmarks.

## Testing Strategy

### Test Categories
1. **Unit Tests**: Test individual wrapper functionality in isolation
2. **Integration Tests**: Test wrapper combinations and interactions
3. **Performance Tests**: Benchmark against original implementation
4. **Compatibility Tests**: Verify Isaac Lab integration

### Test Environment Setup
- Mock environment for unit tests
- Real Isaac Lab factory environment for integration tests
- Performance benchmarking environment
- Multi-agent test configurations

---

## Unit Test Plan

### 1. ForceTorqueWrapper Tests

#### Test File: `tests/test_force_torque_wrapper.py`

**Test Cases:**
- `test_initialization`
  - Verify wrapper initializes without errors
  - Check sensor initialization flags
  - Validate configuration parameters

- `test_force_torque_sensor_setup`
  - Mock RobotView initialization
  - Verify force-torque tensors creation
  - Check episode statistics initialization

- `test_get_force_torque_stats`
  - Verify method returns correct dictionary structure
  - Test with valid force-torque data
  - Test with missing data (empty dict return)
  - Validate current_force/torque tensor shapes

- `test_has_force_torque_data`
  - Test with force-torque data available
  - Test without force-torque data
  - Verify boolean return type

- `test_get_force_torque_observation`
  - Test observation formatting (N, 6)
  - Test tanh scaling functionality
  - Test fallback to zeros when no data

- `test_episode_statistics_tracking`
  - Verify max force/torque tracking
  - Test sum force/torque accumulation
  - Check reset functionality

- `test_tanh_scaling`
  - Verify scaling applied when enabled
  - Test scale factor effects
  - Compare scaled vs unscaled outputs

**Mock Requirements:**
- Mock RobotView/ArticulationView
- Mock environment with num_envs, device
- Mock force-torque sensor data

---

### 2. FragileObjectWrapper Tests

#### Test File: `tests/test_fragile_object_wrapper.py`

**Test Cases:**
- `test_initialization_single_break_force`
  - Test single float break force
  - Verify break_force tensor creation
  - Check fragile flag setting

- `test_initialization_multi_agent_break_forces`
  - Test list of break forces
  - Verify per-agent assignment
  - Test unbreakable objects (-1 value)

- `test_agent_assignment_validation`
  - Test num_envs divisible by num_agents
  - Verify error for invalid combinations
  - Check environment assignment mapping

- `test_get_agent_assignment`
  - Verify correct agent-to-environment mapping
  - Test with different num_agents values
  - Check returned dictionary structure

- `test_get_break_forces`
  - Verify tensor clone return
  - Test break force values per environment
  - Check device and dtype

- `test_get_agent_break_force`
  - Test valid agent_id
  - Test invalid agent_id (error case)
  - Verify returned tensor subset

- `test_force_violation_detection`
  - Mock force-torque data above threshold
  - Test force violation boolean mask
  - Verify integration with ForceTorqueWrapper

- `test_episode_termination`
  - Test _wrapped_get_dones functionality
  - Verify force violations trigger termination
  - Test without force-torque data

- `test_is_fragile`
  - Test with fragile objects
  - Test with unbreakable objects only
  - Verify boolean return

**Mock Requirements:**
- Mock environment with force-torque data
- Mock _get_dones method
- Multi-agent configuration scenarios

---

### 3. EfficientResetWrapper Tests

#### Test File: `tests/test_efficient_reset_wrapper.py`

**Test Cases:**
- `test_initialization`
  - Verify wrapper initializes correctly
  - Check initialization flags
  - Test with/without scene attribute

- `test_has_cached_state`
  - Test with cached state
  - Test without cached state
  - Verify boolean return

- `test_clear_cached_state`
  - Cache state then clear
  - Verify has_cached_state returns False
  - Test reset behavior after clearing

- `test_full_reset_caching`
  - Simulate full environment reset
  - Verify state gets cached
  - Check scene.get_state() call

- `test_partial_reset_shuffling`
  - Cache initial state
  - Perform partial reset
  - Verify state shuffling logic
  - Check environment origin adjustments

- `test_get_reset_efficiency_stats`
  - Verify returned dictionary structure
  - Test supports_efficient_reset detection
  - Check has_cached_state reflection

- `test_state_shuffling_logic`
  - Test random index generation
  - Verify articulation state shuffling
  - Check root pose/velocity handling
  - Test joint position/velocity shuffling

**Mock Requirements:**
- Mock scene with articulations
- Mock scene.get_state() and set_state()
- Mock articulation objects with state methods
- Environment origins for position adjustment

---

### 4. HybridForcePositionWrapper Tests

#### Test File: `tests/test_hybrid_force_position_wrapper.py`

**Test Cases:**
- `test_initialization`
  - Test with different reward_type values
  - Verify ctrl_torque parameter effects
  - Check action space size calculation

- `test_action_space_expansion`
  - Verify original 6 → extended size
  - Test force_size calculation (3 vs 6)
  - Check environment action space update

- `test_selection_matrix_handling`
  - Test selection matrix parsing from actions
  - Verify force/position mode switching
  - Check tensor shapes and values

- `test_reward_strategies`
  - Test "simp" reward calculation
  - Test "dirs" directional rewards
  - Test "delta" selection matrix penalty
  - Test "pos_simp" position-focused
  - Test "wrench_norm" low wrench
  - Test "base" (no hybrid rewards)

- `test_force_goal_calculation`
  - Test force action parsing
  - Verify bounds checking
  - Check force gain application

- `test_position_control_integration`
  - Test position action parsing
  - Verify pose calculation methods
  - Check quaternion handling

- `test_out_of_bounds_detection`
  - Test position target bounds checking
  - Verify out-of-bounds handling
  - Check safety mechanisms

- `test_control_law_computation`
  - Mock factory_control_utils functions
  - Test hybrid control law calculation
  - Verify force/position blending

**Mock Requirements:**
- Mock factory_control_utils functions
- Mock environment with control configuration
- Various action inputs for testing
- Position and force target scenarios

---

### 5. WandbLoggingWrapper Tests

#### Test File: `tests/test_wandb_logging_wrapper.py`

**Test Cases:**
- `test_initialization`
  - Test with valid wandb_config
  - Verify num_agents parameter
  - Check EpisodeTracker creation

- `test_agent_assignment`
  - Test multi-agent environment division
  - Verify get_agent_assignment() output
  - Check static assignment consistency

- `test_episode_tracker_reset`
  - Test episode reset functionality
  - Verify metric clearing
  - Check environment-specific resets

- `test_episode_tracking_step`
  - Test reward tracking
  - Verify reward component handling
  - Check episode termination detection

- `test_metric_aggregation`
  - Test episode completion metrics
  - Verify aggregation across environments
  - Check per-agent metric separation

- `test_add_metric`
  - Test custom metric addition
  - Verify metric storage
  - Check publishing integration

- `test_log_learning_metrics`
  - Test RL training metric logging
  - Verify parameter handling
  - Check wandb integration

- `test_log_action_metrics`
  - Test action distribution logging
  - Verify statistical calculations
  - Check global step handling

- `test_publish_metrics`
  - Mock wandb.log calls
  - Verify metric publishing
  - Test publishing frequency

**Mock Requirements:**
- Mock wandb module
- Mock environment with reward components
- Episode completion scenarios
- Multi-agent test configurations

---

### 6. FactoryMetricsWrapper Tests

#### Test File: `tests/test_factory_metrics_wrapper.py`

**Test Cases:**
- `test_initialization`
  - Test with different num_agents
  - Verify tracking variable initialization
  - Check force data detection

- `test_success_tracking`
  - Mock success detection methods
  - Test success time recording
  - Verify first success detection

- `test_engagement_tracking`
  - Mock engagement detection
  - Test engagement timing
  - Verify engagement length tracking

- `test_smoothness_metrics`
  - Test sum squared velocity calculation
  - Verify force/torque statistics
  - Check episode accumulation

- `test_get_success_stats`
  - Verify returned dictionary structure
  - Test with/without successes
  - Check rate calculations

- `test_get_smoothness_stats`
  - Test smoothness metric calculations
  - Verify force/torque statistics
  - Check with/without force data

- `test_get_agent_metrics`
  - Test agent-specific metric extraction
  - Verify agent_id validation
  - Check metric separation

- `test_extras_update`
  - Test extras dictionary updates
  - Verify metric publishing
  - Check conditional logging

**Mock Requirements:**
- Mock environment success detection methods
- Mock force-torque data
- Episode completion scenarios
- Multi-agent configurations

---

### 7. HistoryObservationWrapper Tests

#### Test File: `tests/test_history_observation_wrapper.py`

**Test Cases:**
- `test_initialization`
  - Test default history components
  - Verify custom component lists
  - Check history length configuration

- `test_history_buffer_initialization`
  - Test buffer creation for components
  - Verify tensor shapes and devices
  - Check acceleration buffer setup

- `test_observation_dimension_updates`
  - Mock OBS_DIM_CFG and STATE_DIM_CFG
  - Test dimension multiplication
  - Verify observation space updates

- `test_sample_index_calculation`
  - Test keep_idxs generation
  - Verify linspace calculation
  - Check special case (num_samples=1)

- `test_history_update`
  - Test buffer rolling and updates
  - Verify current observation integration
  - Check reset vs normal update

- `test_acceleration_calculation`
  - Test finite difference calculations
  - Verify acceleration buffer updates
  - Check jerk/snap for force_torque

- `test_get_history_stats`
  - Verify returned statistics
  - Test buffer count tracking
  - Check configuration reflection

- `test_get_component_history`
  - Test valid component retrieval
  - Test invalid component (None return)
  - Verify tensor cloning

- `test_observation_composition`
  - Test obs_order and state_order usage
  - Verify tensor concatenation
  - Check history data integration

**Mock Requirements:**
- Mock environment with observation attributes
- Mock OBS_DIM_CFG and STATE_DIM_CFG
- Various observation component scenarios
- History buffer data

---

### 8. ObservationManagerWrapper Tests

#### Test File: `tests/test_observation_manager_wrapper.py`

**Test Cases:**
- `test_initialization`
  - Test with/without obs_noise
  - Verify noise configuration loading
  - Check wrapper initialization flags

- `test_standard_format_enforcement`
  - Test conversion from single tensor
  - Test conversion from dict format
  - Verify {"policy", "critic"} output

- `test_observation_composition`
  - Mock obs_order and state_order
  - Test tensor concatenation
  - Verify proper ordering

- `test_noise_injection`
  - Test noise application
  - Verify noise configuration usage
  - Check noise mean/std handling

- `test_observation_validation`
  - Test format validation
  - Verify tensor shape checking
  - Test NaN/inf detection

- `test_get_observation_info`
  - Test with valid observations
  - Verify returned statistics
  - Check error handling

- `test_get_observation_space_info`
  - Test configuration reflection
  - Verify space information
  - Check missing configuration handling

- `test_validate_wrapper_stack`
  - Test validation rules
  - Verify issue detection
  - Check configuration consistency

**Mock Requirements:**
- Mock environment with observation methods
- Mock configuration with obs_order/state_order
- Various observation formats
- Noise configuration scenarios

---

### 9. FactoryEnvironmentBuilder Tests

#### Test File: `tests/test_factory_environment_builder.py`

**Test Cases:**
- `test_builder_pattern`
  - Test method chaining
  - Verify wrapper configuration storage
  - Check builder state management

- `test_preset_configurations`
  - Test each preset ("basic", "training", etc.)
  - Verify preset wrapper combinations
  - Check preset parameter application

- `test_create_factory_environment`
  - Test with different presets
  - Verify wrapper application order
  - Check configuration overrides

- `test_create_multi_agent_environment`
  - Test multi-agent setup
  - Verify environment count adjustment
  - Check agent assignment

- `test_validate_environment_config`
  - Test validation rules
  - Verify issue detection
  - Check configuration requirements

- `test_configuration_overrides`
  - Test config override application
  - Verify parameter precedence
  - Check override integration

**Mock Requirements:**
- Mock Isaac Lab environment creation
- Mock configuration objects
- Wrapper initialization scenarios
- Multi-agent configurations

---

## Integration Test Plan

### 1. Wrapper Combination Tests

#### Test File: `tests/test_wrapper_combinations.py`

**Test Cases:**
- `test_sensor_mechanics_combination`
  - ForceTorque + FragileObject + EfficientReset
  - Verify data flow between wrappers
  - Check force violation detection

- `test_full_stack_combination`
  - All wrappers applied together
  - Verify wrapper order importance
  - Check initialization sequence

- `test_logging_integration`
  - Metrics + Wandb logging combination
  - Verify metric data flow
  - Check logging consistency

- `test_observation_stack`
  - History + ObservationManager combination
  - Verify observation format consistency
  - Check dimension calculations

### 2. Multi-Agent Integration Tests

#### Test File: `tests/test_multi_agent_integration.py`

**Test Cases:**
- `test_static_agent_assignment`
  - Verify consistent environment assignment
  - Test across all multi-agent wrappers
  - Check assignment boundary handling

- `test_per_agent_metrics`
  - Verify metric separation by agent
  - Test agent-specific configurations
  - Check cross-agent isolation

### 3. Performance Integration Tests

#### Test File: `tests/test_performance_integration.py`

**Test Cases:**
- `test_wrapper_overhead`
  - Benchmark step/reset times
  - Compare against unwrapped environment
  - Measure memory usage

- `test_efficient_reset_performance`
  - Benchmark reset time improvement
  - Compare full vs partial resets
  - Measure cache effectiveness

---

## Performance Test Plan

### Benchmarking Strategy
1. **Baseline Measurement**: Original factory environment
2. **Individual Wrapper Impact**: Each wrapper separately
3. **Combined Wrapper Impact**: Full wrapper stack
4. **Scaling Tests**: Different environment counts

### Metrics to Track
- Step time (ms/step)
- Reset time (ms/reset)
- Memory usage (MB)
- GPU utilization (%)
- CPU utilization (%)

### Test Configurations
- Multi-agent with 256 environments per agent assignment:
  - 2 agents: 512 environments total (256 envs per agent)
  - 3 agents: 768 environments total (256 envs per agent)
  - 4 agents: 1024 environments total (256 envs per agent)
  - 5 agents: 1280 environments total (256 envs per agent)
- Single-agent baselines: 256, 512, 1024 environments
- Performance scaling analysis across agent counts

---

## Test Execution Plan

### Phase 1: Unit Tests (Week 1)
1. Set up test infrastructure
2. Implement mock environments
3. Write and run unit tests for each wrapper
4. Fix identified issues

### Phase 2: Integration Tests (Week 1-2)
1. Test wrapper combinations
2. Multi-agent integration testing
3. Isaac Lab integration verification
4. Fix compatibility issues

### Phase 3: Performance Tests (Week 2)
1. Baseline performance measurement
2. Individual wrapper benchmarking
3. Full stack performance testing
4. Performance optimization if needed

### Phase 4: Validation (Week 2)
1. Compare outputs with original implementation
2. Validate functionality parity
3. Document any differences
4. Final test suite run

---

## Test Infrastructure Requirements

### Dependencies
```bash
pytest>=7.0.0
pytest-mock>=3.6.0
pytest-benchmark>=4.0.0
torch>=1.11.0
numpy>=1.21.0
```

### Mock Environment Setup
- Minimal gym.Env implementation
- Mock Isaac Lab components
- Configurable test scenarios
- Performance measurement utilities

### CI/CD Integration
- Automated test execution
- Performance regression detection
- Test coverage reporting
- Isaac Lab compatibility validation

---

## Success Criteria

### Unit Tests
- [x] 100% unit test coverage for public methods - **COMPLETED**
- [x] All edge cases handled - **COMPLETED**
- [x] Error conditions tested - **COMPLETED**
- [x] Mock integrations working - **COMPLETED**

**Status: ✅ COMPLETED - All 9 wrapper test suites implemented with 240+ test cases**

### Integration Tests
- [ ] All wrapper combinations functional
- [ ] Multi-agent scenarios working
- [ ] Isaac Lab integration verified
- [ ] No data corruption between wrappers

### Performance Tests
- [ ] Wrapper overhead < 10% of baseline
- [ ] Memory usage within acceptable bounds
- [ ] Efficient reset shows measurable improvement
- [ ] Scaling performance acceptable

### Validation Tests
- [ ] Functionality parity with original
- [ ] No regression in training performance
- [ ] All original features preserved
- [ ] Multi-agent functionality working

This comprehensive test plan ensures all wrappers are thoroughly validated before deployment and provides confidence in the refactored architecture.