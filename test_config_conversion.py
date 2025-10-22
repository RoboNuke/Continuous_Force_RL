"""
Test script to verify config-to-dict conversion for wandb
Uses actual config classes from the project
"""

import pprint
from configs.config_manager_v3 import ConfigManagerV3


def convert_to_dict(obj, max_depth=10, _current_depth=0):
    """
    Recursively convert Python objects to plain dictionaries for wandb.

    Args:
        obj: Object to convert
        max_depth: Maximum recursion depth to prevent infinite loops
        _current_depth: Internal parameter for tracking recursion depth

    Returns:
        Plain dictionary, list, or primitive type suitable for wandb
    """
    # Prevent infinite recursion
    if _current_depth > max_depth:
        return str(obj)

    # Handle None
    if obj is None:
        return None

    # Handle primitive types
    if isinstance(obj, (int, float, str, bool)):
        return obj

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [convert_to_dict(item, max_depth, _current_depth + 1) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        return {
            key: convert_to_dict(value, max_depth, _current_depth + 1)
            for key, value in obj.items()
        }

    # Handle special types that need string conversion
    # torch.device, torch.dtype, pathlib.Path, etc.
    if type(obj).__name__ in ['device', 'dtype', 'PosixPath', 'WindowsPath', 'Path']:
        return str(obj)

    # Handle objects with __dict__ (dataclasses, custom classes, etc.)
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            # Skip callables (methods, functions, classes)
            if callable(value):
                continue
            # Skip class types
            if isinstance(value, type):
                continue
            # Convert the value recursively
            result[key] = convert_to_dict(value, max_depth, _current_depth + 1)
        return result

    # Fallback: convert to string representation
    return str(obj)


def main():
    print("="*80)
    print("Testing Config Conversion for Wandb")
    print("="*80)

    # Load the config
    config_path = "configs/experiments/hard_fPiH/hybrid_control_exp.yaml"
    config_manager = ConfigManagerV3(verbose=False)
    configs = config_manager.process_config(config_path)

    # Simulate what happens in training - create agent configs
    import learning.launch_utils_v3 as lUtils
    lUtils.define_agent_configs(configs)

    print("\n" + "="*80)
    print("BEFORE CONVERSION - Raw Config Objects")
    print("="*80)
    print("\nConfig sections available:")
    for key in configs.keys():
        if key not in ['config_paths', 'cli_overrides']:
            print(f"  - {key}: {type(configs[key])}")

    print("\n" + "-"*80)
    print("Sample: 'primary' config object attributes:")
    print("-"*80)
    if 'primary' in configs:
        for attr_name in dir(configs['primary']):
            if not attr_name.startswith('_') and not callable(getattr(configs['primary'], attr_name)):
                attr_value = getattr(configs['primary'], attr_name)
                print(f"  {attr_name}: {attr_value} (type: {type(attr_value).__name__})")

    # Simulate what the wrapper SHOULD do - combine ALL configs
    print("\n" + "="*80)
    print("SIMULATING WHAT WRAPPER SHOULD DO - COMBINE ALL CONFIGS")
    print("="*80)

    if 'agent' in configs and hasattr(configs['agent'], 'agent_exp_cfgs'):
        agent_config = configs['agent'].agent_exp_cfgs[0]

        # Combine ALL configs, not just agent
        all_configs_to_send = {}

        # Add all main config sections
        for key in ['primary', 'environment', 'model', 'wrappers', 'experiment', 'agent']:
            if key in configs:
                print(f"Converting {key} config...")
                all_configs_to_send[key] = convert_to_dict(configs[key])

        # Add the agent-specific config
        all_configs_to_send['agent_specific'] = convert_to_dict(agent_config)

        # Convert to dict
        print("\n" + "="*80)
        print("AFTER CONVERSION - Plain Dictionary")
        print("="*80)

        combined_dict = all_configs_to_send

        print("\nCombined config that would be sent to wandb:")
        print("-"*80)
        pprint.pprint(combined_dict, width=100, compact=False)

        print("\n" + "="*80)
        print(f"Total top-level keys: {len(combined_dict)}")
        print("="*80)

        # Show just the top-level keys
        print("\nTop-level keys that will appear in wandb Config tab:")
        for key in sorted(combined_dict.keys()):
            print(f"  - {key}")

    else:
        print("ERROR: Could not find agent_exp_cfgs in config")

    print("\n✓ Conversion test complete!")


if __name__ == "__main__":
    main()
