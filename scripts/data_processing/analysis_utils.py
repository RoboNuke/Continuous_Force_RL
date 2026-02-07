"""
Analysis Utilities for Data Processing Notebooks

This module provides shared functionality for all data analysis notebooks including:
- WandB data fetching
- Statistical computations (rates, confidence intervals)
- Standardized plotting functions
- Global constants for consistent styling

Usage:
    from analysis_utils import (
        get_best_checkpoint_per_run,
        download_eval_data,
        plot_rate_figure,
        plot_rate_figure_by_method,
        COLORS,
        NOISE_LEVEL_COLORS,
    )
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon, Rectangle, Circle, Ellipse
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

# =============================================================================
# WANDB CONFIGURATION
# =============================================================================
ENTITY = "hur"
PROJECT = "SG_Exps"

# =============================================================================
# EVALUATION TAGS
# =============================================================================
TAG_EVAL_PERFORMANCE = "eval_performance"
TAG_EVAL_NOISE = "eval_forge_noise" #"eval_noise"
TAG_EVAL_DYNAMICS = "eval_dynamics"
TAG_EVAL_GAIN = "eval_gain"
TAG_EVAL_YAW = "eval_yaw"
TAG_EVAL_FRAGILE = "eval_fragile"
TAG_OLD_NOISE_EVAL = "old_noise_eval"

# =============================================================================
# METRICS
# =============================================================================
METRIC_SUCCESS = "num_successful_completions"
METRIC_BREAKS = "num_breaks"
METRIC_TOTAL = "total_episodes"

# =============================================================================
# COLORS - MUST REMAIN CONSISTENT ACROSS ALL ANALYSES
# DO NOT MODIFY WITHOUT EXPLICIT PERMISSION
# =============================================================================
COLORS = {
    # Base method names
    "Pose": "#2ca02c",           # Green
    "Hybrid-Basic": "#ff7f0e",   # Orange
    "MATCH": "#1f77b4",          # Blue
    "SWISH": "#1f77b4",          # Blue (alias)
    "LCLoP": "#1f77b4",          # Blue (alias)

    # 1mm noise variants
    "Pose(1mm)": "#2ca02c",
    "MATCH(1mm)": "#1f77b4",
    "Hybrid-Basic(1mm)": "#ff7f0e",

    # 2.5mm noise variants
    "Pose(2.5mm)": "#781fb4",    # Purple
    "SWISH(2.5mm)": "#b41f1f",   # Red
    "Hybrid-Basic(2.5mm)": "#b4aa1f",  # Yellow-green

    # state-std variants
    "Pose(state-std)": "#2ca02c",
    "MATCH(state-std)": "#1f77b4",
    "Hybrid-Basic(state-std)": "#ff7f0e",

    # Yaw/angle variants
    "Pose(3deg)": "#781fb4",
    "Pose(15deg)": "#2ca02c",
    "MATCH(3deg)": "#b41f1f",
    "MATCH(15deg)": "#1f77b4",
    "Hybrid-Basic(3deg)": "#b4aa1f",
    "Hybrid-Basic(15deg)": "#ff7f0e",
}

# Noise level colors (for plot_rate_figure_by_method)
NOISE_LEVEL_COLORS = {
    "1mm": "#2ca02c",      # Green (low noise)
    "2.5mm": "#1f77b4",    # Blue
    "5mm": "#ff7f0e",      # Orange
    "7.5mm": "#d62728",    # Red (high noise)
}

# =============================================================================
# DEFAULT PLOT STYLING
# =============================================================================
DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (6, 4.5)
DEFAULT_FIGSIZE_GRID_CELL = (4.25, 3)

# Font sizes
FONT_SUPTITLE = 16
FONT_TITLE = 14
FONT_AXIS_LABEL = 12
FONT_TICK = 10
FONT_LEGEND = 10
FONT_BAR_LABEL = 7
FONT_NA = 12

# Error bar defaults
ERROR_BAR_CAPSIZE = 3
ERROR_BAR_COLOR = "black"
ERROR_BAR_LINEWIDTH = 1.5

# Highlight styling (for multi-panel grids)
HIGHLIGHT_COLOR = "gold"
HIGHLIGHT_LINEWIDTH = 3

# Bar plot defaults
DEFAULT_GROUP_WIDTH = 0.8

# =============================================================================
# SHAPE ICON CONFIGURATION (for shape_comparison plots)
# =============================================================================
SHAPE_ICON_SIZE = 0.06
SHAPE_ICON_Y_OFFSET = 0.09
SHAPE_TEXT_OFFSET = -0.03
SHAPE_ICON_COLOR = "#333333"
SHAPE_ICON_EDGE_COLOR = "black"
SHAPE_ICON_LINEWIDTH = 1.5

SHAPE_ICONS = {
    "circle": ("circle", {}),
    "square": ("square", {}),
    "rectangle": ("rectangle", {"aspect": 0.5}),
    "hexagon": ("polygon", {"num_sides": 6}),
    "oval": ("oval", {"aspect": 0.6}),
    "arch": ("arch", {}),
}


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def get_best_checkpoint_per_run(
    api,
    method_tag: str,
    eval_tag: str = TAG_EVAL_PERFORMANCE,
    max_checkpoint: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Find the best checkpoint for each run based on score = success - breaks.

    Args:
        api: wandb.Api() instance
        method_tag: Tag identifying the method/experiment
        eval_tag: Tag for evaluation runs (default: eval_performance)
        max_checkpoint: If set, only consider checkpoints with total_steps <= this value.
                       If None, consider all checkpoints.
        verbose: Print progress information

    Returns:
        dict[run_id] -> {run_name, best_step, score}
    """
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"$and": [{"tags": method_tag}, {"tags": eval_tag}]}
    )

    best_checkpoints = {}
    for run in runs:
        history = run.history()
        if history.empty:
            if verbose:
                print(f"Warning: Run {run.name} has no history data")
            continue

        # Filter to checkpoints at or below max_checkpoint if specified
        if max_checkpoint is not None:
            history = history[history["total_steps"] <= max_checkpoint]
            if history.empty:
                if verbose:
                    print(f"Warning: Run {run.name} has no checkpoints <= {max_checkpoint}")
                continue

        # Calculate score: successes - breaks
        history["score"] = history[f"Eval_Core/{METRIC_SUCCESS}"] - history[f"Eval_Core/{METRIC_BREAKS}"]
        best_idx = history["score"].idxmax()
        best_step = int(history.loc[best_idx, "total_steps"])

        best_checkpoints[run.id] = {
            "run_name": run.name,
            "best_step": best_step,
            "score": history.loc[best_idx, "score"],
        }
        if verbose:
            print(f"  {run.name}: best checkpoint at step {best_step} (score: {history.loc[best_idx, 'score']:.0f})")

    return best_checkpoints


def download_eval_data(
    api,
    method_tag: str,
    best_checkpoints: Dict,
    level_mapping: Dict[str, str],
    prefix_template: str,
    level_col_name: str = "level",
    eval_tag: str = TAG_EVAL_NOISE,
    old_noise_filter: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generic download function for level-based eval data (noise, gain, yaw, etc.).

    Args:
        api: wandb.Api() instance
        method_tag: Tag identifying the method/experiment
        best_checkpoints: Output from get_best_checkpoint_per_run()
        level_mapping: Dict mapping display label -> metric key
                       e.g., {"1mm": "0mm-1mm", "2.5mm": "1mm-2.5mm"}
        prefix_template: Format string for metric prefix, use {level} as placeholder
                         e.g., "Noise_Eval({level})_Core"
        level_col_name: Name for the level column in output DataFrame
        eval_tag: Tag for evaluation runs
        old_noise_filter: None, "include", or "exclude" for old_noise_eval tag
        verbose: Print progress information

    Returns:
        DataFrame with columns: [run_id, run_name, checkpoint, {level_col_name},
                                 success, breaks, total]
    """
    # Build filter
    base_filters = [{"tags": method_tag}, {"tags": eval_tag}]

    if old_noise_filter == "include":
        base_filters.append({"tags": TAG_OLD_NOISE_EVAL})
    elif old_noise_filter == "exclude":
        base_filters.append({"tags": {"$ne": TAG_OLD_NOISE_EVAL}})

    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"$and": base_filters}
    )

    # Build lookup by agent number from best_checkpoints
    checkpoint_by_agent = {}
    for run_id, info in best_checkpoints.items():
        agent_num = info["run_name"].rsplit("_", 1)[-1]
        checkpoint_by_agent[agent_num] = info["best_step"]

    data = []
    for run in runs:
        # Extract agent number from run name
        agent_num = run.name.rsplit("_", 1)[-1]

        if agent_num not in checkpoint_by_agent:
            if verbose:
                print(f"Warning: No matching performance run for agent {agent_num} ({run.name})")
            continue

        best_step = checkpoint_by_agent[agent_num]
        history = run.history()

        # Check for empty history or missing total_steps column
        if history.empty or "total_steps" not in history.columns:
            if verbose:
                print(f"Warning: Run {run.name} has no history data")
            continue

        if best_step not in history["total_steps"].values:
            if verbose:
                print(f"Warning: Checkpoint {best_step} not found in {run.name}")
            continue

        row = history[history["total_steps"] == best_step].iloc[0]

        for level_label, level_value in level_mapping.items():
            prefix = prefix_template.format(level=level_value)
            data.append({
                "run_id": run.id,
                "run_name": run.name,
                "checkpoint": best_step,
                level_col_name: level_label,
                "success": row[f"{prefix}/{METRIC_SUCCESS}"],
                "breaks": row[f"{prefix}/{METRIC_BREAKS}"],
                "total": row[f"{prefix}/{METRIC_TOTAL}"],
            })

    return pd.DataFrame(data)


def download_eval_performance_data(
    api,
    method_tag: str,
    best_checkpoints: Dict,
    extra_metrics: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download eval_performance data for best checkpoints.

    Args:
        api: wandb.Api() instance
        method_tag: Tag identifying the method/experiment
        best_checkpoints: Output from get_best_checkpoint_per_run()
        extra_metrics: List of additional metric names to fetch (without Eval_Core/ prefix)
        verbose: Print progress information

    Returns:
        DataFrame with columns: [run_id, run_name, checkpoint, success, breaks, total,
                                 + extra_metrics]
    """
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"$and": [{"tags": method_tag}, {"tags": TAG_EVAL_PERFORMANCE}]}
    )

    data = []
    for run in runs:
        if run.id not in best_checkpoints:
            continue

        best_step = best_checkpoints[run.id]["best_step"]
        history = run.history()

        # Check for empty history or missing total_steps column
        if history.empty or "total_steps" not in history.columns:
            if verbose:
                print(f"Warning: Run {run.name} has no history data")
            continue

        if best_step not in history["total_steps"].values:
            if verbose:
                print(f"Warning: Checkpoint {best_step} not found in {run.name}")
            continue

        row = history[history["total_steps"] == best_step].iloc[0]

        record = {
            "run_id": run.id,
            "run_name": run.name,
            "checkpoint": best_step,
            "success": row[f"Eval_Core/{METRIC_SUCCESS}"],
            "breaks": row[f"Eval_Core/{METRIC_BREAKS}"],
            "total": row[f"Eval_Core/{METRIC_TOTAL}"],
        }

        if extra_metrics:
            for metric in extra_metrics:
                record[metric] = row[f"Eval_Core/{metric}"]

        data.append(record)

    return pd.DataFrame(data)


def download_training_history(
    api,
    method_tag: str,
    metric_col: str = "Episode/success_rate",
    scale_factor: float = 100.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download training history (success rate over time) from training runs.

    Args:
        api: wandb.Api() instance
        method_tag: Tag identifying the method/experiment
        metric_col: Column name for the metric to extract
        scale_factor: Multiply metric by this value (e.g., 100 to convert 0-1 to percentage)
        verbose: Print progress information

    Returns:
        DataFrame with columns: [run_id, run_name, total_steps, value]
    """
    # Get training runs (exclude eval_performance tag)
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"$and": [{"tags": method_tag}, {"tags": {"$ne": TAG_EVAL_PERFORMANCE}}]}
    )

    all_data = []
    for run in runs:
        history = run.history()
        if history.empty:
            if verbose:
                print(f"Warning: Run {run.name} has no history data")
            continue

        for _, row in history.iterrows():
            if pd.notna(row.get("total_steps")) and pd.notna(row.get(metric_col)):
                all_data.append({
                    "run_id": run.id,
                    "run_name": run.name,
                    "total_steps": int(row["total_steps"]),
                    "value": row[metric_col] * scale_factor,
                })

        if verbose:
            print(f"  {run.name}: {len(history)} checkpoints")

    return pd.DataFrame(all_data)


# =============================================================================
# DATA FILTERING & STATISTICS FUNCTIONS
# =============================================================================

def filter_top_n_runs(
    df: pd.DataFrame,
    best_checkpoints: Dict,
    top_n: Optional[int],
    match_by: str = "agent_num",
) -> pd.DataFrame:
    """
    Filter dataframe to top N runs by score.

    Args:
        df: DataFrame to filter
        best_checkpoints: Output from get_best_checkpoint_per_run()
        top_n: Number of top runs to keep, or None for all
        match_by: "agent_num" (extract from run_name suffix) or "run_id"

    Returns:
        Filtered DataFrame
    """
    if df.empty or top_n is None or len(best_checkpoints) <= top_n:
        return df

    # Sort by score and get top N
    sorted_runs = sorted(best_checkpoints.items(), key=lambda x: x[1]["score"], reverse=True)
    top_entries = sorted_runs[:top_n]

    if match_by == "run_id":
        top_ids = {run_id for run_id, _ in top_entries}
        return df[df["run_id"].isin(top_ids)]
    else:  # agent_num
        top_run_names = {info["run_name"] for _, info in top_entries}
        top_agent_nums = {name.rsplit("_", 1)[-1] for name in top_run_names}
        return df[df["run_name"].apply(lambda x: x.rsplit("_", 1)[-1] in top_agent_nums)]


def compute_rates(
    df: pd.DataFrame,
    level_labels: List[str],
    level_col: str,
    metric: str = "success",
    total_col: str = "total",
    error_type: str = "ci",
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute rates with error estimates for each level.

    Args:
        df: DataFrame with data
        level_labels: List of level values to compute for
        level_col: Column name containing level values
        metric: Column name for the metric (e.g., "success", "breaks")
        total_col: Column name for total count
        error_type: "ci" (95% CI from per-seed std), "binary_se", or "sem"

    Returns:
        (means, errors, errors_lower, errors_upper)
        - means: mean rate for each level
        - errors: raw error values (CI, SE, or SEM)
        - errors_lower/upper: clipped for asymmetric error bars on [0, 100]
    """
    if df.empty:
        zeros = [0.0] * len(level_labels)
        return zeros, zeros, zeros, zeros

    means = []
    errors = []
    errors_lower = []
    errors_upper = []

    for level_label in level_labels:
        subset = df[df[level_col] == level_label]

        if subset.empty:
            means.append(0.0)
            errors.append(0.0)
            errors_lower.append(0.0)
            errors_upper.append(0.0)
            continue

        if error_type == "binary_se":
            # Binary standard error: sqrt(p*(1-p)/n) on pooled data
            total_metric = subset[metric].sum()
            total_episodes = subset[total_col].sum()
            p = total_metric / total_episodes
            mean_pct = p * 100
            err = np.sqrt(p * (1 - p) / total_episodes) * 100
        else:
            # Per-seed statistics
            subset = subset.copy()
            subset["rate"] = 100 * subset[metric] / subset[total_col]
            mean_pct = subset["rate"].mean()
            std = subset["rate"].std()
            n_seeds = len(subset)
            sem = std / np.sqrt(n_seeds) if n_seeds > 0 else 0

            if error_type == "ci":
                err = 1.96 * sem  # 95% CI
            else:  # sem
                err = sem

        means.append(mean_pct)
        errors.append(err)
        # Clip error bars to stay within [0, 100]
        errors_lower.append(min(err, mean_pct))
        errors_upper.append(min(err, 100 - mean_pct))

    return means, errors, errors_lower, errors_upper


def compute_stats_by_step(
    df: pd.DataFrame,
    value_col: str = "value",
    step_col: str = "total_steps",
) -> pd.DataFrame:
    """
    Compute mean and 95% CI for each step (for training curves).

    Args:
        df: DataFrame with training data
        value_col: Column containing the metric values
        step_col: Column containing step numbers

    Returns:
        DataFrame with columns: [total_steps, mean, std, count, sem, ci_95, lower, upper]
    """
    if df.empty:
        return pd.DataFrame()

    stats = df.groupby(step_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    stats.columns = [step_col, "mean", "std", "count"]

    # Calculate SEM and 95% CI
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci_95"] = 1.96 * stats["sem"]

    # Calculate bounds clipped to [0, 100]
    stats["lower"] = (stats["mean"] - stats["ci_95"]).clip(0, 100)
    stats["upper"] = (stats["mean"] + stats["ci_95"]).clip(0, 100)

    return stats.sort_values(step_col)


# =============================================================================
# COLOR UTILITIES
# =============================================================================

def get_method_color(
    method_name: str,
    colors: Optional[Dict[str, str]] = None,
    default: str = "#333333",
) -> str:
    """
    Get color for a method, trying exact match then base name extraction.

    Args:
        method_name: Full method name (e.g., "Pose(1mm)")
        colors: Color dictionary to use (default: COLORS)
        default: Fallback color if not found

    Returns:
        Color hex string
    """
    if colors is None:
        colors = COLORS

    # Try exact match
    if method_name in colors:
        return colors[method_name]

    # Try base name (everything before parenthesis)
    base_name = method_name.split("(")[0]
    if base_name in colors:
        return colors[base_name]

    return default


# =============================================================================
# SHAPE ICON DRAWING (for shape_comparison plots)
# =============================================================================

def draw_shape_icon(
    ax,
    shape_key: str,
    x: float,
    y: float,
    size: float = SHAPE_ICON_SIZE,
    shape_icons_config: Optional[Dict] = None,
) -> Optional[plt.Axes]:
    """
    Draw a shape icon at the specified position in figure coordinates.

    Args:
        ax: Parent axes (used to create inset)
        shape_key: Key into shape_icons_config (e.g., "circle", "hexagon")
        x, y: Position in figure coordinates
        size: Size of the icon
        shape_icons_config: Shape configuration dict (default: SHAPE_ICONS)

    Returns:
        Inset axes containing the shape, or None if shape_key not found
    """
    if shape_icons_config is None:
        shape_icons_config = SHAPE_ICONS

    if shape_key not in shape_icons_config:
        return None

    shape_type, kwargs = shape_icons_config[shape_key]

    # Adjust size for certain shapes
    if shape_type in ["oval", "rectangle"]:
        size = 1.5 * size

    # Create inset axes
    inset_ax = ax.inset_axes(
        [x - size/2, y - size/2, size, size],
        transform=ax.figure.transFigure
    )
    ax_size = 1.2
    inset_ax.set_xlim(-ax_size, ax_size)
    inset_ax.set_ylim(-ax_size, ax_size)
    inset_ax.axis("off")

    patch = None

    if shape_type == "circle":
        inset_ax.set_aspect("equal")
        patch = Circle((0, 0), 1, facecolor=SHAPE_ICON_COLOR,
                       edgecolor=SHAPE_ICON_EDGE_COLOR, linewidth=SHAPE_ICON_LINEWIDTH)

    elif shape_type == "square":
        inset_ax.set_aspect("equal")
        patch = Rectangle((-0.9, -0.9), 1.8, 1.8, facecolor=SHAPE_ICON_COLOR,
                         edgecolor=SHAPE_ICON_EDGE_COLOR, linewidth=SHAPE_ICON_LINEWIDTH)

    elif shape_type == "rectangle":
        aspect = kwargs.get("aspect", 0.5)
        base_size = 2.0
        patch = Rectangle((-base_size/2, -base_size/2 * aspect), base_size, base_size * aspect,
                         facecolor=SHAPE_ICON_COLOR, edgecolor=SHAPE_ICON_EDGE_COLOR,
                         linewidth=SHAPE_ICON_LINEWIDTH)

    elif shape_type == "oval":
        aspect = kwargs.get("aspect", 0.6)
        base_size = 2.0
        patch = Ellipse((0, 0), base_size, base_size * aspect, facecolor=SHAPE_ICON_COLOR,
                       edgecolor=SHAPE_ICON_EDGE_COLOR, linewidth=SHAPE_ICON_LINEWIDTH)

    elif shape_type == "polygon":
        inset_ax.set_aspect("equal")
        num_sides = kwargs.get("num_sides", 6)
        patch = RegularPolygon((0, 0), num_sides, radius=1, facecolor=SHAPE_ICON_COLOR,
                              edgecolor=SHAPE_ICON_EDGE_COLOR, linewidth=SHAPE_ICON_LINEWIDTH)

    elif shape_type == "arch":
        inset_ax.set_aspect("equal")
        rect_width = 1.8
        rect_height = 1.2
        arc_radius = 0.55

        half_w = rect_width / 2
        half_h = rect_height / 2
        bottom_y = -half_h
        top_y = half_h

        n_arc_points = 30
        theta = np.linspace(0, np.pi, n_arc_points)
        arc_x = arc_radius * np.cos(theta)
        arc_y = bottom_y + arc_radius * np.sin(theta)

        verts = [(-half_w, top_y), (half_w, top_y), (half_w, bottom_y), (arc_radius, bottom_y)]
        for ax_pt, ay_pt in zip(arc_x, arc_y):
            verts.append((ax_pt, ay_pt))
        verts.extend([(-arc_radius, bottom_y), (-half_w, bottom_y), (-half_w, top_y)])

        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor=SHAPE_ICON_COLOR,
                         edgecolor=SHAPE_ICON_EDGE_COLOR, linewidth=SHAPE_ICON_LINEWIDTH)

    if patch is not None:
        inset_ax.add_patch(patch)

    return inset_ax


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_grouped_bars(
    ax: plt.Axes,
    x: np.ndarray,
    data_by_method: Dict[str, Tuple[List[float], List[float], List[float]]],
    method_names: List[str],
    colors: Optional[Dict[str, str]] = None,
    group_width: float = DEFAULT_GROUP_WIDTH,
    show_labels: bool = True,
    label_fontsize: int = FONT_BAR_LABEL,
    label_decimal: int = 1,
    capsize: int = ERROR_BAR_CAPSIZE,
    y_lim: Tuple[float, float] = (0, 100),
) -> List:
    """
    Plot grouped bar chart on given axes.

    This is the core bar-drawing function used by higher-level plotting functions.

    Args:
        ax: Matplotlib axes to plot on
        x: X-axis positions (numpy array)
        data_by_method: dict[method_name] -> (means, errors_lower, errors_upper)
        method_names: List of method names (determines bar order)
        colors: Color dictionary (default: COLORS)
        group_width: Total width for bar group at each x position
        show_labels: Whether to show rotated value labels on bars
        label_fontsize: Font size for bar labels
        label_decimal: Decimal places for label values
        capsize: Error bar cap size
        y_lim: Y-axis limits (used for label placement calculation)

    Returns:
        List of bar container objects
    """
    if colors is None:
        colors = COLORS

    n_methods = len(method_names)
    bar_width = group_width / n_methods
    all_bars = []

    for i, method_name in enumerate(method_names):
        if method_name not in data_by_method:
            continue

        means, errors_lower, errors_upper = data_by_method[method_name]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        color = get_method_color(method_name, colors)

        bars = ax.bar(
            x + offset, means, bar_width,
            label=method_name, color=color,
            yerr=[errors_lower, errors_upper],
            capsize=capsize, zorder=3
        )
        all_bars.append(bars)

        # Add rotated labels
        if show_labels:
            y_range = y_lim[1] - y_lim[0]
            label_height_estimate = y_range * 0.20

            # Get the raw errors for label text (use lower as proxy for symmetric display)
            for bar, mean, err_lo, err_hi in zip(bars, means, errors_lower, errors_upper):
                if mean > 0:
                    # Use average of lower/upper for display (they should be similar)
                    err_display = (err_lo + err_hi) / 2

                    # Determine label position (inside bar if fits, above otherwise)
                    space_inside = bar.get_height() - err_lo
                    threshold = label_height_estimate + (8 if y_lim[1] >= 50 else 0.5)

                    if space_inside > threshold:
                        label_y = bar.get_height() - err_lo - (1 if y_lim[1] >= 50 else 0.1)
                        va = "top"
                    else:
                        label_y = bar.get_height() + err_hi + (3 if y_lim[1] >= 50 else 0.2)
                        va = "bottom"

                    ax.text(
                        bar.get_x() + bar.get_width() / 2, label_y,
                        f"{mean:.{label_decimal}f}±{err_display:.{label_decimal}f}",
                        ha="center", va=va, rotation=90,
                        fontsize=label_fontsize, color="black", fontweight="bold"
                    )

    return all_bars


def plot_rate_figure(
    data: Dict[str, pd.DataFrame],
    method_names: List[str],
    level_labels: List[str],
    level_col: str,
    metric: str = "success",
    total_col: str = "total",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    y_lim: Tuple[float, float] = (0, 100),
    y_ticks: Optional[List[float]] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    error_type: str = "ci",
    colors: Optional[Dict[str, str]] = None,
    show_labels: bool = True,
    label_decimal: int = 1,
    filter_top_n: Optional[int] = None,
    best_checkpoints: Optional[Dict[str, Dict]] = None,
    legend_loc: str = "best",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create complete rate comparison figure (success rate, break rate, etc.).

    This is a high-level convenience function that:
    1. Creates figure and axes
    2. Computes rates from raw DataFrames
    3. Calls plot_grouped_bars() for rendering
    4. Configures all decorations

    Args:
        data: dict[method_name] -> DataFrame with level_col and metric columns
        method_names: List of method names (determines bar order and legend)
        level_labels: List of level values for x-axis
        level_col: Column name in DataFrames containing level values
        metric: Column name for the metric ("success" or "breaks")
        total_col: Column name for total count
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        y_lim: Y-axis limits
        y_ticks: Y-axis tick values (auto if None)
        figsize: Figure size
        dpi: Figure DPI
        error_type: "ci" (95% CI) or "binary_se"
        colors: Color dictionary (default: COLORS)
        show_labels: Show rotated value labels on bars
        label_decimal: Decimal places for labels
        filter_top_n: Filter to top N runs by score (requires best_checkpoints)
        best_checkpoints: dict[method] -> checkpoints dict (required if filter_top_n set)
        legend_loc: Legend location

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Compute rates for each method
    data_by_method = {}
    for method_name in method_names:
        if method_name not in data:
            continue

        df = data[method_name]

        # Filter if requested
        if filter_top_n is not None and best_checkpoints is not None:
            if method_name in best_checkpoints:
                df = filter_top_n_runs(df, best_checkpoints[method_name], filter_top_n)

        means, errors, errors_lower, errors_upper = compute_rates(
            df, level_labels, level_col, metric, total_col, error_type
        )
        data_by_method[method_name] = (means, errors_lower, errors_upper)

    # Plot bars
    x = np.arange(len(level_labels))
    plot_grouped_bars(
        ax, x, data_by_method, method_names, colors,
        show_labels=show_labels, label_decimal=label_decimal, y_lim=y_lim
    )

    # Configure axes
    ax.set_xlabel(x_label, fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_AXIS_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, fontsize=FONT_TICK)
    ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc=legend_loc)

    plt.tight_layout()
    return fig, ax


def plot_rate_figure_by_method(
    data: Dict[str, pd.DataFrame],
    method_names: List[str],
    level_labels: List[str],
    level_col: str,
    metric: str = "success",
    total_col: str = "total",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    y_lim: Tuple[float, float] = (0, 100),
    y_ticks: Optional[List[float]] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    error_type: str = "ci",
    level_colors: Optional[Dict[str, str]] = None,
    show_labels: bool = True,
    label_decimal: int = 1,
    filter_top_n: Optional[int] = None,
    best_checkpoints: Optional[Dict[str, Dict]] = None,
    legend_loc: str = "best",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create rate comparison figure with methods on x-axis and levels as grouped bars.

    This is a transposed view of plot_rate_figure():
    - X-axis: method names
    - Grouped bars: levels (e.g., noise levels), colored by level_colors

    Args:
        data: dict[method_name] -> DataFrame with level_col and metric columns
        method_names: List of method names (determines x-axis order)
        level_labels: List of level values (determines bar grouping and legend)
        level_col: Column name in DataFrames containing level values
        metric: Column name for the metric ("success" or "breaks")
        total_col: Column name for total count
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        y_lim: Y-axis limits
        y_ticks: Y-axis tick values (auto if None)
        figsize: Figure size
        dpi: Figure DPI
        error_type: "ci" (95% CI) or "binary_se"
        level_colors: Color dictionary for levels (default: NOISE_LEVEL_COLORS)
        show_labels: Show rotated value labels on bars
        label_decimal: Decimal places for labels
        filter_top_n: Filter to top N runs by score (requires best_checkpoints)
        best_checkpoints: dict[method] -> checkpoints dict (required if filter_top_n set)
        legend_loc: Legend location

    Returns:
        (fig, ax)
    """
    if level_colors is None:
        level_colors = NOISE_LEVEL_COLORS

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Reorganize data: for each level, compute rates across all methods
    # data_by_level[level_label] = (means_per_method, errors_lower, errors_upper)
    data_by_level = {}

    for level_label in level_labels:
        means = []
        errors_lower = []
        errors_upper = []

        for method_name in method_names:
            if method_name not in data:
                means.append(0.0)
                errors_lower.append(0.0)
                errors_upper.append(0.0)
                continue

            df = data[method_name]

            # Filter if requested
            if filter_top_n is not None and best_checkpoints is not None:
                if method_name in best_checkpoints:
                    df = filter_top_n_runs(df, best_checkpoints[method_name], filter_top_n)

            # Compute rate for this single level
            level_means, level_errors, level_errors_lower, level_errors_upper = compute_rates(
                df, [level_label], level_col, metric, total_col, error_type
            )
            means.append(level_means[0])
            errors_lower.append(level_errors_lower[0])
            errors_upper.append(level_errors_upper[0])

        data_by_level[level_label] = (means, errors_lower, errors_upper)

    # Plot bars using plot_grouped_bars
    # Here, x positions are methods, and "methods" in plot_grouped_bars are actually levels
    x = np.arange(len(method_names))
    plot_grouped_bars(
        ax, x, data_by_level, level_labels, level_colors,
        show_labels=show_labels, label_decimal=label_decimal, y_lim=y_lim
    )

    # Configure axes
    ax.set_xlabel(x_label, fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_AXIS_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=FONT_TICK)
    ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc=legend_loc)

    plt.tight_layout()
    return fig, ax


def plot_multi_panel_grid(
    data: Dict[str, Dict[str, pd.DataFrame]],
    panel_keys: List[str],
    panel_display_names: Dict[str, str],
    method_names: List[str],
    level_labels: List[str],
    level_col: str,
    metric: str = "success",
    total_col: str = "total",
    n_cols: int = 2,
    suptitle: str = "",
    x_label: str = "",
    y_label: str = "",
    y_lim: Tuple[float, float] = (0, 100),
    y_ticks: Optional[List[float]] = None,
    figsize_per_cell: Tuple[float, float] = DEFAULT_FIGSIZE_GRID_CELL,
    dpi: int = DEFAULT_DPI,
    error_type: str = "ci",
    colors: Optional[Dict[str, str]] = None,
    show_labels: bool = True,
    label_decimal: int = 1,
    highlight_panel: Optional[str] = None,
    na_panels: Optional[List[str]] = None,
    na_text: str = "N/A",
    filter_top_n: Optional[int] = None,
    best_checkpoints: Optional[Dict[str, Dict[str, Dict]]] = None,
    show_shape_icons: bool = False,
    shape_icons_config: Optional[Dict] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create 2D grid of bar plots (for fragility, clearance, shape comparisons).

    This function creates a grid of subplots, each using plot_grouped_bars()
    for consistent rendering.

    Args:
        data: dict[panel_key][method_name] -> DataFrame
        panel_keys: List of panel keys (determines panel order)
        panel_display_names: dict[panel_key] -> display string for titles
        method_names: List of method names
        level_labels: List of level values for x-axis
        level_col: Column name containing level values
        metric: Column name for the metric
        total_col: Column name for total count
        n_cols: Number of columns in grid
        suptitle: Figure super title
        x_label: X-axis label (only on bottom row)
        y_label: Y-axis label (only on left column)
        y_lim: Y-axis limits
        y_ticks: Y-axis tick values
        figsize_per_cell: Size per subplot cell
        dpi: Figure DPI
        error_type: "ci" or "binary_se"
        colors: Color dictionary
        show_labels: Show rotated value labels
        label_decimal: Decimal places for labels
        highlight_panel: Panel key to highlight with gold border
        na_panels: List of panel keys to show "N/A" instead of plot
        na_text: Text to display in N/A panels
        filter_top_n: Filter to top N runs
        best_checkpoints: dict[panel_key][method] -> checkpoints
        show_shape_icons: Whether to draw shape icons above panels
        shape_icons_config: Shape icon configuration

    Returns:
        (fig, axes)
    """
    import math

    if na_panels is None:
        na_panels = []

    n_panels = len(panel_keys)
    n_rows = math.ceil(n_panels / n_cols)

    figsize = (figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    axes = np.atleast_2d(axes)

    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, y=0.99)

    x = np.arange(len(level_labels))

    for idx, panel_key in enumerate(panel_keys):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        display_name = panel_display_names.get(panel_key, panel_key)
        is_na_panel = panel_key in na_panels

        if is_na_panel:
            # Show N/A text
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, na_text, transform=ax.transAxes,
                    fontsize=FONT_NA, ha="center", va="center",
                    style="italic", color="gray")
            ax.set_title(display_name, fontsize=FONT_TITLE - 2)
            for spine in ax.spines.values():
                spine.set_visible(True)
        else:
            # Compute rates for each method
            data_by_method = {}
            for method_name in method_names:
                if panel_key not in data or method_name not in data[panel_key]:
                    continue

                df = data[panel_key][method_name]

                # Filter if requested
                if filter_top_n is not None and best_checkpoints is not None:
                    if panel_key in best_checkpoints and method_name in best_checkpoints[panel_key]:
                        df = filter_top_n_runs(df, best_checkpoints[panel_key][method_name], filter_top_n)

                means, errors, errors_lower, errors_upper = compute_rates(
                    df, level_labels, level_col, metric, total_col, error_type
                )
                data_by_method[method_name] = (means, errors_lower, errors_upper)

            # Plot bars
            plot_grouped_bars(
                ax, x, data_by_method, method_names, colors,
                show_labels=show_labels, label_decimal=label_decimal, y_lim=y_lim
            )

            # Configure subplot
            ax.set_title(display_name, fontsize=FONT_TITLE - 2)
            ax.set_xticks(x)
            ax.set_xticklabels(level_labels, fontsize=FONT_TICK)
            ax.set_ylim(y_lim)
            if y_ticks is not None:
                ax.set_yticks(y_ticks)
            ax.tick_params(axis="y", labelsize=FONT_TICK)

            # Only show legend on first plot
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                # Strip "(1mm)" suffix from legend labels for cleaner display
                labels = [l.replace("(1mm)", "").strip() for l in labels]
                ax.legend(handles, labels, fontsize=FONT_LEGEND - 1, loc="upper right")

        # X-axis label only on bottom row
        if row == n_rows - 1:
            ax.set_xlabel(x_label, fontsize=FONT_AXIS_LABEL)

        # Y-axis label only on left column
        if col == 0:
            ax.set_ylabel(y_label, fontsize=FONT_AXIS_LABEL)

        # Highlight border
        if highlight_panel is not None and panel_key == highlight_panel:
            for spine in ["top", "left", "right", "bottom"]:
                ax.spines[spine].set_color(HIGHLIGHT_COLOR)
                ax.spines[spine].set_linewidth(HIGHLIGHT_LINEWIDTH)

    # Hide unused subplots
    for idx in range(n_panels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)

    # Draw shape icons if requested
    if show_shape_icons:
        plt.subplots_adjust(hspace=0.45)
        fig.canvas.draw()

        for idx, panel_key in enumerate(panel_keys):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            display_name = panel_display_names.get(panel_key, panel_key)

            bbox = ax.get_position()
            center_x = bbox.x0 + bbox.width / 2
            icon_y = bbox.y1 + SHAPE_ICON_Y_OFFSET

            draw_shape_icon(ax, panel_key, center_x, icon_y, SHAPE_ICON_SIZE, shape_icons_config)

            text_y = icon_y + SHAPE_TEXT_OFFSET
            fig.text(center_x, text_y, display_name, ha="center", va="top",
                     fontsize=FONT_TITLE - 2, fontweight="bold")

    return fig, axes


def plot_training_curves(
    data_by_method: Dict[str, pd.DataFrame],
    colors: Optional[Dict[str, str]] = None,
    title: str = "",
    x_label: str = "Total Steps",
    y_label: str = "Success Rate (%)",
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Tuple[float, float] = (0, 100),
    y_ticks: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = DEFAULT_DPI,
    threshold: Optional[float] = None,
    show_threshold_annotations: bool = False,
    ci_alpha: float = 0.2,
    value_col: str = "value",
    legend_loc: str = "lower right",
) -> Tuple[plt.Figure, plt.Axes, Optional[Dict[str, int]]]:
    """
    Plot training curves with shaded confidence intervals.

    Args:
        data_by_method: dict[method_name] -> DataFrame with total_steps and value columns
        colors: Color dictionary
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        x_lim: X-axis limits (min, max), or None for auto
        y_lim: Y-axis limits
        y_ticks: Y-axis tick values
        figsize: Figure size
        dpi: Figure DPI
        threshold: Draw horizontal line at this value and find crossings
        show_threshold_annotations: Show vertical lines and labels at crossings
        ci_alpha: Alpha for confidence interval shading
        value_col: Column name for values in DataFrames
        legend_loc: Legend location

    Returns:
        (fig, ax, threshold_crossings) - crossings is dict[method] -> step, or None if no threshold
    """
    if colors is None:
        colors = COLORS

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    threshold_crossings = {} if threshold is not None else None

    for method_name, df in data_by_method.items():
        stats = compute_stats_by_step(df, value_col=value_col)

        if stats.empty:
            continue

        color = get_method_color(method_name, colors)

        # Plot mean line
        ax.plot(stats["total_steps"], stats["mean"],
                color=color, label=method_name, linewidth=2)

        # Plot CI shaded region
        ax.fill_between(stats["total_steps"], stats["lower"], stats["upper"],
                        color=color, alpha=ci_alpha)

        # Find threshold crossing
        if threshold is not None:
            crossing = stats[stats["mean"] >= threshold]
            if not crossing.empty:
                threshold_crossings[method_name] = int(crossing.iloc[0]["total_steps"])

    # Plot threshold line
    if threshold is not None and show_threshold_annotations:
        ax.axhline(y=threshold, color="black", linestyle="--", linewidth=1.5,
                   label=f"{threshold}% Threshold")

        # Add crossing annotations
        for i, (method_name, step) in enumerate(sorted(threshold_crossings.items(), key=lambda x: x[1])):
            ax.axvline(x=step, color="black", linestyle=":", alpha=0.5, linewidth=1)
            y_pos = 5 + (i * 6)
            ax.text(step, y_pos, f"{method_name}: {step:,}",
                    fontsize=9, color="black", ha="left", va="bottom")

    # Configure axes
    ax.set_xlabel(x_label, fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_AXIS_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(6, 6))
    ax.legend(fontsize=FONT_LEGEND, loc=legend_loc)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax, threshold_crossings


# =============================================================================
# DATA SUMMARY UTILITIES
# =============================================================================

def print_data_summary(
    data: Dict[str, pd.DataFrame],
    level_labels: List[str],
    level_col: str,
    metric: str = "success",
    total_col: str = "total",
    title: str = "DATA SUMMARY",
):
    """
    Print a summary of rates for each method and level.

    Args:
        data: dict[method_name] -> DataFrame
        level_labels: List of level values
        level_col: Column name containing level values
        metric: Column name for the metric
        total_col: Column name for total count
        title: Header title
    """
    print("=" * 60)
    print(title)
    print("=" * 60)

    for method_name, df in data.items():
        print(f"\n{method_name}:")
        if df.empty:
            print("  No data")
            continue
        for level in level_labels:
            subset = df[df[level_col] == level]
            if not subset.empty:
                total = subset[total_col].sum()
                rate = 100 * subset[metric].sum() / total
                print(f"  {level}: {rate:.1f}%")
            else:
                print(f"  {level}: No data")
