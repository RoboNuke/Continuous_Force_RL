# WandB Data Structure Reference

This document describes the structure and conventions used in the wandb logging for the Continuous Force RL project. Use this as a reference when creating data analysis notebooks.

## Connection Details

- **Entity:** `hur`
- **Project:** `SG_Exps`

## Run Organization

### Method Tags
Each experimental method has its own tag. Runs are tagged with the method name.
Example placeholder tags:
- `pose_control`
- `hybrid_basic`
- `lclop`

### Evaluation Tags
Each seed generates multiple runs, further tagged by evaluation type:
- `eval_performance` - Nominal evaluation at training noise level (typically 1mm)
- `eval_noise` - Evaluation across multiple noise levels

### Seeds
- Typically 5 seeds per method (do not hardcode this value)
- Each seed produces separate runs for each evaluation type

## Metric Naming Conventions

### Performance Evaluation Metrics
For `eval_performance` tagged runs:
```
Eval_Core/{metric_name}
```

### Noise Evaluation Metrics
For `eval_noise` tagged runs:
```
Noise_Eval({min_noise}mm-{max_noise}mm)_Core/{metric_name}
```

### Noise Level Ranges
Noise levels are cumulative ranges where the previous max becomes the next min:
| Display Label | Metric Range |
|---------------|--------------|
| 1mm | `0mm-1mm` |
| 2.5mm | `1mm-2.5mm` |
| 5mm | `2.5mm-5mm` |
| 7.5mm | `5mm-7.5mm` |

## Available Metrics

All metrics are available under both `Eval_Core/` and `Noise_Eval(...)_Core/` prefixes:

| Metric Name | Description |
|-------------|-------------|
| `num_successful_completions` | Number of successful task completions |
| `num_breaks` | Number of breaks/failures |
| `total_episodes` | Total number of episodes evaluated |
| `num_failed_timeouts` | Number of episodes that timed out |
| `ssv` | Sum of squared velocities |
| `ssjv` | Sum of squared joint velocities |
| `max_force` | Maximum force applied |
| `episode_length` | Average episode length (time in seconds) |
| `avg_force_in_contact` | Average force while in contact |
| `energy` | Energy consumption metric |

## Checkpoint Identification

- Checkpoints are identified by `total_steps`
- This value corresponds to the number of training steps completed at that checkpoint
- Each row in wandb history represents data from a specific checkpoint

## Best Policy Selection

To determine the best policy for a given run:
1. Query the `eval_performance` tagged run
2. For each checkpoint (row), calculate: `num_successful_completions - num_breaks`
3. Select the checkpoint with the highest score
4. Use this `total_steps` value to identify the best checkpoint

## Computed Metrics

### Success Rate
```python
success_rate = (num_successful_completions / total_episodes) * 100
```

### Break Rate
```python
break_rate = (num_breaks / total_episodes) * 100
```

## Notebook Template Structure

When creating data analysis notebooks, follow this structure:

### Block 1: Imports & Constants
- All imports (wandb, matplotlib, numpy, pandas)
- Entity, project constants
- Method tags
- Noise level mappings
- Metric key names

### Block 2: Determine Best Policy
- For each method tag, query runs with both method tag AND `eval_performance`
- Calculate best checkpoint per run based on success - breaks
- Store best `total_steps` value per run

### Block 3: Download Data
- Download data for best checkpoints from relevant eval runs
- Format into pandas DataFrames
- Print human-readable summary

### Block 4+: Individual Plot Blocks
Each plot block must be **completely self-contained** (except for raw data):
- Define all styling constants within the block (figure size, fonts, colors, etc.)
- Only dependency is the DataFrame from Block 3
- Single block rerun = updated plot

## LaTeX Table Formatting

For publication tables:
- Use mean ± std format: `{mean:.1f} ± {std:.1f}`
- Bold the best value in each column
- Use `\textbf{}` for bolding in LaTeX
- Include directional indicators: ↑ for higher-is-better, ↓ for lower-is-better
