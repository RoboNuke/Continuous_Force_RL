#!/usr/bin/env python3
"""
Configuration Comparison Utility

Compare two configuration files or a configuration file with different CLI overrides
to identify differences and ensure consistency across experiments.

Usage:
    python scripts/config_compare.py config1.yaml config2.yaml
    python scripts/config_compare.py config.yaml --overrides1 "key=val1" --overrides2 "key=val2"
    python scripts/config_compare.py config1.yaml config2.yaml --output comparison.json
    python scripts/config_compare.py config.yaml config.yaml --overrides2 "primary.decimation=16"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config_manager_v2 import ConfigManagerV2
except ImportError as e:
    print(f"Error: Could not import configuration system: {e}")
    sys.exit(1)


class ConfigComparer:
    """Utility to compare configurations and identify differences."""

    def __init__(self):
        self.differences = []
        self.config1_info = {}
        self.config2_info = {}

    def compare_configs(self,
                       config1_path: str,
                       config2_path: str,
                       overrides1: Optional[List[str]] = None,
                       overrides2: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare two configurations and return detailed differences.

        Args:
            config1_path: Path to first configuration file
            config2_path: Path to second configuration file
            overrides1: CLI overrides for first config
            overrides2: CLI overrides for second config

        Returns:
            Dictionary with comparison results
        """
        overrides1 = overrides1 or []
        overrides2 = overrides2 or []

        try:
            # Load both configurations
            print(f"Loading configuration 1: {config1_path}")
            if overrides1:
                print(f"  With overrides: {overrides1}")
            bundle1 = ConfigManagerV2.load_defaults_first_config(
                config_path=config1_path,
                cli_overrides=overrides1
            )

            print(f"Loading configuration 2: {config2_path}")
            if overrides2:
                print(f"  With overrides: {overrides2}")
            bundle2 = ConfigManagerV2.load_defaults_first_config(
                config_path=config2_path,
                cli_overrides=overrides2
            )

            # Convert to legacy format for comparison
            config1_dict = ConfigManagerV2.get_legacy_config_dict(bundle1)
            config2_dict = ConfigManagerV2.get_legacy_config_dict(bundle2)

            # Store basic info
            self.config1_info = {
                'path': config1_path,
                'overrides': overrides1,
                'task_name': bundle1.task_name,
                'total_agents': bundle1.primary_cfg.total_agents,
                'total_envs': bundle1.primary_cfg.total_num_envs
            }

            self.config2_info = {
                'path': config2_path,
                'overrides': overrides2,
                'task_name': bundle2.task_name,
                'total_agents': bundle2.primary_cfg.total_agents,
                'total_envs': bundle2.primary_cfg.total_num_envs
            }

            # Compare configurations
            self._compare_dicts(config1_dict, config2_dict, "")

            return {
                'config1': self.config1_info,
                'config2': self.config2_info,
                'differences': self.differences,
                'num_differences': len(self.differences),
                'are_identical': len(self.differences) == 0
            }

        except Exception as e:
            return {
                'error': str(e),
                'config1': self.config1_info,
                'config2': self.config2_info,
                'differences': [],
                'num_differences': 0,
                'are_identical': False
            }

    def _compare_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = "") -> None:
        """Recursively compare two dictionaries and record differences."""

        # Get all keys from both dictionaries
        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in sorted(all_keys):
            current_path = f"{path}.{key}" if path else key

            # Check if key exists in both
            if key not in dict1:
                self.differences.append({
                    'path': current_path,
                    'type': 'missing_in_config1',
                    'config1_value': None,
                    'config2_value': dict2[key]
                })
                continue

            if key not in dict2:
                self.differences.append({
                    'path': current_path,
                    'type': 'missing_in_config2',
                    'config1_value': dict1[key],
                    'config2_value': None
                })
                continue

            # Both keys exist, compare values
            val1 = dict1[key]
            val2 = dict2[key]

            if isinstance(val1, dict) and isinstance(val2, dict):
                # Recursively compare nested dictionaries
                self._compare_dicts(val1, val2, current_path)
            elif self._values_differ(val1, val2):
                # Values are different
                self.differences.append({
                    'path': current_path,
                    'type': 'value_difference',
                    'config1_value': val1,
                    'config2_value': val2
                })

    def _values_differ(self, val1: Any, val2: Any) -> bool:
        """Check if two values are different, handling floating point precision."""

        # Handle None values
        if val1 is None or val2 is None:
            return val1 != val2

        # Handle floating point comparison with tolerance
        if isinstance(val1, float) and isinstance(val2, float):
            return abs(val1 - val2) > 1e-10

        # Handle lists
        if isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return True
            for a, b in zip(val1, val2):
                if self._values_differ(a, b):
                    return True
            return False

        # Direct comparison
        return val1 != val2

    def generate_report(self, comparison_result: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate a detailed comparison report."""

        if comparison_result.get('error'):
            report_lines = [
                "=" * 80,
                "CONFIGURATION COMPARISON REPORT - ERROR",
                "=" * 80,
                "",
                f"Error: {comparison_result['error']}",
                ""
            ]
        else:
            config1 = comparison_result['config1']
            config2 = comparison_result['config2']
            differences = comparison_result['differences']
            num_diffs = comparison_result['num_differences']

            # Header
            status = "IDENTICAL" if comparison_result['are_identical'] else f"{num_diffs} DIFFERENCES"
            report_lines = [
                "=" * 80,
                f"CONFIGURATION COMPARISON REPORT - {status}",
                "=" * 80,
                "",
                "CONFIGURATION 1:",
                f"  File: {config1['path']}",
                f"  Overrides: {config1.get('overrides', [])}",
                f"  Task: {config1.get('task_name', 'Unknown')}",
                f"  Total agents: {config1.get('total_agents', 'Unknown')}",
                f"  Total envs: {config1.get('total_envs', 'Unknown')}",
                "",
                "CONFIGURATION 2:",
                f"  File: {config2['path']}",
                f"  Overrides: {config2.get('overrides', [])}",
                f"  Task: {config2.get('task_name', 'Unknown')}",
                f"  Total agents: {config2.get('total_agents', 'Unknown')}",
                f"  Total envs: {config2.get('total_envs', 'Unknown')}",
                ""
            ]

            if comparison_result['are_identical']:
                report_lines.extend([
                    "✅ CONFIGURATIONS ARE IDENTICAL",
                    "",
                    "Both configurations produce exactly the same parameter values.",
                    "This is expected when comparing the same experiment configuration."
                ])
            else:
                report_lines.extend([
                    f"DIFFERENCES FOUND ({num_diffs}):",
                    "-" * 40,
                    ""
                ])

                # Group differences by type
                value_diffs = [d for d in differences if d['type'] == 'value_difference']
                missing_in_1 = [d for d in differences if d['type'] == 'missing_in_config1']
                missing_in_2 = [d for d in differences if d['type'] == 'missing_in_config2']

                # Value differences
                if value_diffs:
                    report_lines.append("VALUE DIFFERENCES:")
                    for diff in value_diffs:
                        report_lines.extend([
                            f"  📍 {diff['path']}",
                            f"     Config 1: {diff['config1_value']}",
                            f"     Config 2: {diff['config2_value']}",
                            ""
                        ])

                # Missing in config 1
                if missing_in_1:
                    report_lines.append("MISSING IN CONFIG 1:")
                    for diff in missing_in_1:
                        report_lines.extend([
                            f"  ➖ {diff['path']}",
                            f"     Value: {diff['config2_value']}",
                            ""
                        ])

                # Missing in config 2
                if missing_in_2:
                    report_lines.append("MISSING IN CONFIG 2:")
                    for diff in missing_in_2:
                        report_lines.extend([
                            f"  ➖ {diff['path']}",
                            f"     Value: {diff['config1_value']}",
                            ""
                        ])

                # Analysis
                report_lines.extend([
                    "ANALYSIS:",
                    "-" * 40,
                ])

                if value_diffs:
                    critical_diffs = [d for d in value_diffs if self._is_critical_difference(d['path'])]
                    if critical_diffs:
                        report_lines.append(f"⚠️  {len(critical_diffs)} critical differences found that may affect training")
                    else:
                        report_lines.append("ℹ️  All differences appear to be non-critical parameter variations")

                if missing_in_1 or missing_in_2:
                    report_lines.append("⚠️  Missing parameters may indicate configuration schema differences")

        report_text = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

            # Also save JSON data
            json_file = output_file.replace('.txt', '.json')
            with open(json_file, 'w') as f:
                json.dump(comparison_result, f, indent=2)

            print(f"Report saved to: {output_file}")
            print(f"JSON data saved to: {json_file}")

        return report_text

    def _is_critical_difference(self, path: str) -> bool:
        """Determine if a parameter difference is critical for training."""
        critical_params = [
            'primary.agents_per_break_force',
            'primary.num_envs_per_agent',
            'primary.decimation',
            'primary.policy_hz',
            'environment.episode_length_s',
            'model.actor',
            'model.critic',
            'agent.learning_rate',
            'agent.policy_learning_rate',
            'agent.critic_learning_rate',
            'agent.learning_epochs'
        ]

        return any(critical in path for critical in critical_params)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Compare two configuration files for differences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two different config files
  python scripts/config_compare.py config1.yaml config2.yaml

  # Compare same config with different overrides
  python scripts/config_compare.py config.yaml config.yaml \\
    --overrides2 "primary.decimation=16"

  # Compare with overrides on both sides
  python scripts/config_compare.py config1.yaml config2.yaml \\
    --overrides1 "model.use_hybrid_agent=true" \\
    --overrides2 "model.use_hybrid_agent=false"

  # Save detailed report
  python scripts/config_compare.py config1.yaml config2.yaml \\
    --output comparison_report.txt
        """
    )

    parser.add_argument(
        'config1',
        help="Path to first configuration file"
    )

    parser.add_argument(
        'config2',
        help="Path to second configuration file"
    )

    parser.add_argument(
        '--overrides1',
        action='append',
        help="CLI overrides for first config (can be used multiple times)"
    )

    parser.add_argument(
        '--overrides2',
        action='append',
        help="CLI overrides for second config (can be used multiple times)"
    )

    parser.add_argument(
        '--output', '-o',
        help="Save detailed report to file"
    )

    args = parser.parse_args()

    # Create comparer and run comparison
    comparer = ConfigComparer()
    result = comparer.compare_configs(
        config1_path=args.config1,
        config2_path=args.config2,
        overrides1=args.overrides1,
        overrides2=args.overrides2
    )

    # Generate and display report
    report = comparer.generate_report(result, args.output)
    print("\n" + report)

    # Exit with appropriate code
    if result.get('error'):
        print(f"\n❌ Comparison failed")
        sys.exit(1)
    elif result['are_identical']:
        print(f"\n✅ Configurations are identical")
        sys.exit(0)
    else:
        print(f"\n⚠️  Found {result['num_differences']} differences")
        sys.exit(0)


if __name__ == "__main__":
    main()