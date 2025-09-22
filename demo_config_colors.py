#!/usr/bin/env python3
"""
Standalone demo of the color-coded configuration system.
Shows how the system tracks and displays configuration sources.
"""

import sys
import os
sys.path.append('/home/hunter/Continuous_Force_RL')

from configs.config_manager import ConfigManager

def main():
    print("🎨 Color-Coded Configuration Demo")
    print("=" * 50)

    # Load base configuration
    print("\n1️⃣  Loading BASE configuration (factory_base.yaml)...")
    base_config = ConfigManager.load_and_resolve_config('configs/base/factory_base.yaml')

    # Show sample of base config
    primary_base = base_config.get('primary', {})
    print(f"   Sample base values:")
    print(f"   • episode_length_s: {primary_base.get('episode_length_s')}")
    print(f"   • max_steps: {primary_base.get('max_steps')}")
    print(f"   • break_forces: {primary_base.get('break_forces')}")

    # Load configuration with local overrides
    print("\n2️⃣  Loading with LOCAL OVERRIDES (hybrid_control_exp.yaml)...")
    hybrid_config = ConfigManager.load_and_resolve_config('configs/experiments/hybrid_control_exp.yaml')

    # Show what got overridden
    primary_hybrid = hybrid_config.get('primary', {})
    print(f"   Local overrides applied:")
    print(f"   • episode_length_s: {primary_base.get('episode_length_s')} → {primary_hybrid.get('episode_length_s')}")
    print(f"   • max_steps: {primary_base.get('max_steps')} → {primary_hybrid.get('max_steps')}")

    # Load with CLI overrides (including environment overrides)
    print("\n3️⃣  Loading with CLI OVERRIDES...")
    cli_config = ConfigManager.load_and_resolve_config(
        'configs/experiments/hybrid_control_exp.yaml',
        overrides=[
            'primary.max_steps=99999999',
            'agent.learning_epochs=15',
            'environment.decimation=8',
            'environment.filter_collisions=false'
        ]
    )

    # Show CLI overrides
    primary_cli = cli_config.get('primary', {})
    agent_cli = cli_config.get('agent', {})
    env_cli = cli_config.get('environment', {})
    print(f"   CLI overrides applied:")
    print(f"   • max_steps: {primary_hybrid.get('max_steps')} → {primary_cli.get('max_steps')}")
    print(f"   • learning_epochs: {hybrid_config.get('agent', {}).get('learning_epochs')} → {agent_cli.get('learning_epochs')}")
    print(f"   • environment.decimation: {hybrid_config.get('environment', {}).get('decimation')} → {env_cli.get('decimation')}")
    print(f"   • environment.filter_collisions: {hybrid_config.get('environment', {}).get('filter_collisions')} → {env_cli.get('filter_collisions')}")

    # Show color-coded output
    print("\n4️⃣  COLOR-CODED OUTPUT:")
    print("   (Blue = Local overrides, Green = CLI overrides, Default = Base)")
    print("-" * 50)

    # Environment config with mixed sources
    sample_environment = {
        'episode_length_s': env_cli.get('episode_length_s'),      # Local override (from hybrid config reference)
        'decimation': env_cli.get('decimation'),                  # CLI override
        'filter_collisions': env_cli.get('filter_collisions'),    # CLI override
        'component_dims': env_cli.get('component_dims', {}),      # Base value (nested)
        'task': env_cli.get('task', {})                           # Base value (nested)
    }

    ConfigManager.print_env_config(sample_environment)

    # Sample primary config with colors
    sample_primary = {
        'episode_length_s': primary_cli.get('episode_length_s'),  # Local override
        'max_steps': primary_cli.get('max_steps'),                # CLI override
        'decimation': primary_cli.get('decimation'),              # Base value
        'policy_hz': primary_cli.get('policy_hz')                 # Base value
    }

    from learning.config_printer import print_config
    print_config("Primary Configuration", sample_primary, color_map=ConfigManager._config_sources)

    # Sample agent config with colors
    sample_agent = {
        'learning_epochs': agent_cli.get('learning_epochs'),          # CLI override
        'policy_learning_rate': agent_cli.get('policy_learning_rate'), # Local override
        'discount_factor': agent_cli.get('discount_factor'),          # Base value
        'lambda': agent_cli.get('lambda')                             # Base value
    }

    print_config("Agent Learning Parameters", sample_agent, color_map=ConfigManager._config_sources)

    # Show source tracking summary
    sources = ConfigManager._config_sources
    base_count = sum(1 for v in sources.values() if v == 'base')
    local_count = sum(1 for v in sources.values() if v == 'local_override')
    cli_count = sum(1 for v in sources.values() if v == 'cli_override')

    print(f"\n5️⃣  SOURCE TRACKING SUMMARY:")
    print(f"   📊 Total parameters tracked: {len(sources)}")
    print(f"   🟨 Base values: {base_count}")
    print(f"   🔵 Local overrides: {local_count}")
    print(f"   🟢 CLI overrides: {cli_count}")

    print(f"\n✅ Demo complete! The color system is ready for use in your training pipeline.")

if __name__ == "__main__":
    main()