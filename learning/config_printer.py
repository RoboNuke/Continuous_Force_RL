"""
Configuration Pretty-Printer Utility

Simple utility to print configuration objects in a readable YAML-like format
for easier debugging and inspection.
"""

import inspect
from typing import Any, Dict, List, Union, Optional


# ANSI Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Colors for config sources
    BLUE = '\033[94m'        # Local overrides
    GREEN = '\033[92m'       # CLI overrides
    YELLOW = '\033[93m'      # Base values
    RED = '\033[91m'         # Errors

    # Additional colors
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'


def print_config(title: str, config_obj: Any, max_depth: int = 3, current_depth: int = 0, color_map: Optional[Dict[str, str]] = None, path_prefix: str = "") -> None:
    """
    Print a configuration object in a readable YAML-like format.

    Args:
        title: Title/name for the configuration section
        config_obj: The configuration object to print
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth (internal use)
        color_map: Dictionary mapping config paths to colors (for source tracking)
        path_prefix: Prefix to add to paths when looking up colors (e.g., "environment")
    """
    if current_depth == 0:
        print(f"\n{'='*60}")
        print(f"📋 {title}")
        print('='*60)

    _print_object(config_obj, indent_level=current_depth, max_depth=max_depth, current_depth=current_depth, color_map=color_map, path_prefix=path_prefix)

    if current_depth == 0:
        print('='*60)


def _print_object(obj: Any, indent_level: int = 0, max_depth: int = 3, current_depth: int = 0, color_map: Optional[Dict[str, str]] = None, path_prefix: str = "") -> None:
    """
    Recursively print an object with proper indentation.

    Args:
        obj: Object to print
        indent_level: Current indentation level
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        color_map: Dictionary mapping config paths to colors
        path_prefix: Current path in the configuration for color mapping
    """
    indent = "  " * indent_level

    # Prevent infinite recursion
    if current_depth >= max_depth:
        print(f"{indent}... (max depth reached)")
        return

    # Handle None
    if obj is None:
        print(f"{indent}null")
        return

    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        color = _get_color_for_path(path_prefix, color_map)
        formatted_value = _format_value(obj)
        if color:
            print(f"{indent}{color}{formatted_value}{Colors.RESET}")
        else:
            print(f"{indent}{formatted_value}")
        return

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            print(f"{indent}[]")
            return

        # If all items are simple types, print on one line
        if len(obj) <= 5 and all(isinstance(item, (str, int, float, bool, type(None))) for item in obj):
            formatted_items = [_format_value(item) for item in obj]
            print(f"{indent}[{', '.join(formatted_items)}]")
        else:
            for i, item in enumerate(obj):
                print(f"{indent}- ")
                item_path = f"{path_prefix}[{i}]" if path_prefix else f"[{i}]"
                _print_object(item, indent_level + 1, max_depth, current_depth + 1, color_map, item_path)
        return

    # Handle dictionaries
    if isinstance(obj, dict):
        if len(obj) == 0:
            print(f"{indent}{{}}")
            return

        for key, value in obj.items():
            key_path = f"{path_prefix}.{key}" if path_prefix else key
            color = _get_color_for_path(key_path, color_map)

            if color:
                print(f"{indent}{color}{key}{Colors.RESET}:", end="")
            else:
                print(f"{indent}{key}:", end="")

            # If value is simple, print on same line
            if isinstance(value, (str, int, float, bool, type(None))):
                value_color = _get_color_for_path(key_path, color_map)
                formatted_value = _format_value(value)
                if value_color:
                    print(f" {value_color}{formatted_value}{Colors.RESET}")
                else:
                    print(f" {formatted_value}")
            elif isinstance(value, (list, tuple)) and len(value) <= 3 and all(isinstance(item, (str, int, float, bool)) for item in value):
                formatted_items = [_format_value(item) for item in value]
                value_color = _get_color_for_path(key_path, color_map)
                formatted_list = f"[{', '.join(formatted_items)}]"
                if value_color:
                    print(f" {value_color}{formatted_list}{Colors.RESET}")
                else:
                    print(f" {formatted_list}")
            else:
                print()  # New line for complex objects
                _print_object(value, indent_level + 1, max_depth, current_depth + 1, color_map, key_path)
        return

    # Handle dataclass objects
    if hasattr(obj, '__dataclass_fields__'):
        class_name = obj.__class__.__name__
        print(f"{indent}# {class_name}")

        for field_name in obj.__dataclass_fields__:
            try:
                field_value = getattr(obj, field_name)
                print(f"{indent}{field_name}:", end="")

                # If value is simple, print on same line
                if isinstance(field_value, (str, int, float, bool, type(None))):
                    field_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
                    value_color = _get_color_for_path(field_path, color_map)
                    formatted_value = _format_value(field_value)
                    if value_color:
                        print(f" {value_color}{formatted_value}{Colors.RESET}")
                    else:
                        print(f" {formatted_value}")
                else:
                    print()  # New line for complex objects
                    field_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
                    _print_object(field_value, indent_level + 1, max_depth, current_depth + 1, color_map, field_path)
            except AttributeError:
                print(f"{indent}{field_name}: <not accessible>")
        return

    # Handle objects with attributes (general case)
    if hasattr(obj, '__dict__'):
        class_name = obj.__class__.__name__
        print(f"{indent}# {class_name}")

        # Get public attributes (not starting with _)
        public_attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}

        if not public_attrs:
            print(f"{indent}<no public attributes>")
            return

        for attr_name, attr_value in public_attrs.items():
            print(f"{indent}{attr_name}:", end="")

            # If value is simple, print on same line
            if isinstance(attr_value, (str, int, float, bool, type(None))):
                attr_path = f"{path_prefix}.{attr_name}" if path_prefix else attr_name
                value_color = _get_color_for_path(attr_path, color_map)
                formatted_value = _format_value(attr_value)
                if value_color:
                    print(f" {value_color}{formatted_value}{Colors.RESET}")
                else:
                    print(f" {formatted_value}")
            else:
                print()  # New line for complex objects
                attr_path = f"{path_prefix}.{attr_name}" if path_prefix else attr_name
                _print_object(attr_value, indent_level + 1, max_depth, current_depth + 1, color_map, attr_path)
        return

    # Fallback for other object types
    try:
        print(f"{indent}{repr(obj)}")
    except:
        print(f"{indent}<{type(obj).__name__} object>")


def _format_value(value: Any) -> str:
    """
    Format a basic value for display.

    Args:
        value: Value to format

    Returns:
        Formatted string representation
    """
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        # Quote strings that contain spaces or special characters
        if ' ' in value or any(char in value for char in '{}[](),'):
            return f'"{value}"'
        return value
    else:
        return str(value)


def print_config_summary(title: str, config_obj: Any) -> None:
    """
    Print a brief summary of a configuration object.

    Args:
        title: Title for the summary
        config_obj: Configuration object to summarize
    """
    print(f"\n📊 {title} Summary:")
    print("-" * 40)

    if hasattr(config_obj, '__dataclass_fields__'):
        field_count = len(config_obj.__dataclass_fields__)
        print(f"Type: {config_obj.__class__.__name__} (dataclass)")
        print(f"Fields: {field_count}")
    elif isinstance(config_obj, dict):
        print(f"Type: Dictionary")
        print(f"Keys: {len(config_obj)}")
        if config_obj:
            print(f"Sample keys: {list(config_obj.keys())[:5]}")
    elif hasattr(config_obj, '__dict__'):
        public_attrs = {k: v for k, v in config_obj.__dict__.items() if not k.startswith('_')}
        print(f"Type: {config_obj.__class__.__name__}")
        print(f"Public attributes: {len(public_attrs)}")
    else:
        print(f"Type: {type(config_obj).__name__}")
        print(f"Value: {repr(config_obj)}")


# Convenience functions for common configurations
def print_env_config(env_cfg: Any) -> None:
    """Print environment configuration in readable format."""
    print_config("Environment Configuration", env_cfg)


def print_agent_config(agent_cfg: Any) -> None:
    """Print agent configuration in readable format."""
    print_config("Agent Configuration", agent_cfg)


def print_model_config(model_cfg: Any) -> None:
    """Print model configuration in readable format."""
    print_config("Model Configuration", model_cfg)


def print_task_config(task_cfg: Any) -> None:
    """Print task configuration in readable format."""
    print_config("Task Configuration", task_cfg)


def print_control_config(ctrl_cfg: Any) -> None:
    """Print control configuration in readable format."""
    print_config("Control Configuration", ctrl_cfg)


def _get_color_for_path(path: str, color_map: Optional[Dict[str, str]]) -> Optional[str]:
    """
    Get the color for a configuration path based on source tracking.

    Args:
        path: The configuration path (e.g., "primary.num_envs", "agent.learning_epochs")
        color_map: Dictionary mapping paths to source types

    Returns:
        ANSI color code or None if no color mapping exists
    """
    if not color_map or not path:
        return None

    # Check exact path match first
    if path in color_map:
        source = color_map[path]
        return _get_color_for_source(source)

    # Check for prefix matches (for nested structures)
    for config_path, source in color_map.items():
        if path.startswith(config_path):
            return _get_color_for_source(source)

    # Check if we need to add common prefixes (for when printing sub-configs)
    common_prefixes = ['environment', 'primary', 'agent', 'model', 'wrappers']
    for prefix in common_prefixes:
        full_path = f"{prefix}.{path}"
        if full_path in color_map:
            source = color_map[full_path]
            return _get_color_for_source(source)

    return None


def _get_color_for_source(source: str) -> Optional[str]:
    """
    Get ANSI color code for a configuration source type.

    Args:
        source: Source type ('base', 'local_override', 'cli_override')

    Returns:
        ANSI color code
    """
    source_colors = {
        'base': None,  # Default color (no override)
        'local_override': Colors.BLUE,
        'cli_override': Colors.GREEN
    }
    return source_colors.get(source)