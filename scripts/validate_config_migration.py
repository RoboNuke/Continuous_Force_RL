#!/usr/bin/env python3
"""
Configuration Migration Validation Script

This script validates that the new ConfigManagerV2 system produces identical
results to the legacy ConfigManager system for all existing configuration files.

Usage:
    python scripts/validate_config_migration.py
    python scripts/validate_config_migration.py --config configs/base/factory_base_v2.yaml
    python scripts/validate_config_migration.py --all-configs
    python scripts/validate_config_migration.py --report-only

Features:
- Load same config with both old and new systems
- Compare all final parameter values with detailed diff
- Ensure derived values match exactly
- Test with CLI overrides
- Generate comprehensive migration report
- Validate all config files in the repository
"""

import argparse
import os
import sys
import json
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import both configuration systems
try:
    from configs.config_manager import ConfigManager as LegacyConfigManager
    from configs.config_manager_v2 import ConfigManagerV2
    BOTH_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import both config systems: {e}")
    BOTH_SYSTEMS_AVAILABLE = False


class ConfigValidationResult:
    """Results of validating a single configuration."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.success = False
        self.error_message = ""
        self.legacy_config = None
        self.new_config = None
        self.differences = []
        self.legacy_load_time = 0.0
        self.new_load_time = 0.0

    def add_difference(self, path: str, legacy_value: Any, new_value: Any):
        """Add a difference between legacy and new configs."""
        self.differences.append({
            'path': path,
            'legacy_value': legacy_value,
            'new_value': new_value
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            'config_path': self.config_path,
            'success': self.success,
            'error_message': self.error_message,
            'num_differences': len(self.differences),
            'differences': self.differences,
            'legacy_load_time': self.legacy_load_time,
            'new_load_time': self.new_load_time,
            'performance_improvement': (
                (self.legacy_load_time - self.new_load_time) / self.legacy_load_time * 100
                if self.legacy_load_time > 0 else 0
            )
        }


class ConfigMigrationValidator:
    """Validates configuration migration between old and new systems."""

    def __init__(self):
        self.results = []
        self.test_cli_overrides = [
            [],  # No overrides
            ['primary.decimation=16'],  # Simple override
            ['primary.decimation=12', 'primary.debug_mode=true'],  # Multiple overrides
            ['model.actor_latent_size=512', 'agent.learning_rate=1e-5'],  # Complex overrides
        ]

    def find_all_config_files(self) -> List[str]:
        """Find all YAML configuration files in the repository."""
        config_files = []

        # Search in configs directory
        configs_dir = project_root / 'configs'
        if configs_dir.exists():
            for yaml_file in configs_dir.rglob('*.yaml'):
                # Skip certain patterns
                if any(skip in str(yaml_file) for skip in ['__pycache__', '.git', 'temp']):
                    continue
                config_files.append(str(yaml_file))

        return sorted(config_files)

    def normalize_config_dict(self, config: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """
        Normalize configuration dictionary for comparison.

        Handles differences in representation between old and new systems
        that don't affect functionality.
        """
        if not isinstance(config, dict):
            return config

        normalized = {}
        for key, value in config.items():
            new_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                normalized[key] = self.normalize_config_dict(value, new_path)
            elif isinstance(value, list):
                # Normalize list values
                normalized[key] = [
                    self.normalize_config_dict(item, f"{new_path}[{i}]")
                    if isinstance(item, dict) else item
                    for i, item in enumerate(value)
                ]
            elif isinstance(value, float):
                # Round floats to avoid precision differences
                normalized[key] = round(value, 10)
            else:
                normalized[key] = value

        return normalized

    def compare_configs(self, legacy_config: Dict[str, Any], new_config: Dict[str, Any],
                       result: ConfigValidationResult, path: str = "") -> None:
        """
        Deep compare two configuration dictionaries and record differences.
        """
        # Normalize both configs
        legacy_norm = self.normalize_config_dict(legacy_config)
        new_norm = self.normalize_config_dict(new_config)

        # Find keys in legacy but not in new
        legacy_keys = set(legacy_norm.keys())
        new_keys = set(new_norm.keys())

        missing_in_new = legacy_keys - new_keys
        missing_in_legacy = new_keys - legacy_keys

        for key in missing_in_new:
            result.add_difference(
                f"{path}.{key}" if path else key,
                legacy_norm[key],
                "[MISSING]"
            )

        for key in missing_in_legacy:
            result.add_difference(
                f"{path}.{key}" if path else key,
                "[MISSING]",
                new_norm[key]
            )

        # Compare common keys
        common_keys = legacy_keys & new_keys
        for key in common_keys:
            current_path = f"{path}.{key}" if path else key
            legacy_val = legacy_norm[key]
            new_val = new_norm[key]

            if isinstance(legacy_val, dict) and isinstance(new_val, dict):
                self.compare_configs(legacy_val, new_val, result, current_path)
            elif legacy_val != new_val:
                result.add_difference(current_path, legacy_val, new_val)

    def validate_single_config(self, config_path: str, cli_overrides: List[str] = None) -> ConfigValidationResult:
        """
        Validate a single configuration file with both systems.
        """
        result = ConfigValidationResult(config_path)
        cli_overrides = cli_overrides or []

        if not BOTH_SYSTEMS_AVAILABLE:
            result.error_message = "Both configuration systems not available"
            return result

        try:
            # Test legacy system
            import time
            start_time = time.time()
            try:
                legacy_config = LegacyConfigManager.load_and_resolve_config(
                    config_path, cli_overrides
                )
                result.legacy_load_time = time.time() - start_time
                result.legacy_config = legacy_config
            except Exception as e:
                result.error_message = f"Legacy system failed: {str(e)}"
                return result

            # Test new system
            start_time = time.time()
            try:
                config_bundle = ConfigManagerV2.load_defaults_first_config(
                    config_path=config_path,
                    cli_overrides=cli_overrides
                )
                new_config = ConfigManagerV2.get_legacy_config_dict(config_bundle)
                result.new_load_time = time.time() - start_time
                result.new_config = new_config
            except Exception as e:
                result.error_message = f"New system failed: {str(e)}"
                return result

            # Compare the configurations
            self.compare_configs(legacy_config, new_config, result)

            # Mark as successful if no critical differences
            result.success = len(result.differences) == 0

        except Exception as e:
            result.error_message = f"Validation failed: {str(e)}\n{traceback.format_exc()}"

        return result

    def validate_config_with_overrides(self, config_path: str) -> List[ConfigValidationResult]:
        """
        Validate a configuration file with various CLI override combinations.
        """
        results = []

        for i, overrides in enumerate(self.test_cli_overrides):
            print(f"  Testing with overrides {i+1}/{len(self.test_cli_overrides)}: {overrides}")
            result = self.validate_single_config(config_path, overrides)
            result.config_path = f"{config_path} [overrides: {overrides}]"
            results.append(result)

        return results

    def validate_all_configs(self, config_files: List[str], test_overrides: bool = True) -> None:
        """
        Validate all provided configuration files.
        """
        print(f"Validating {len(config_files)} configuration files...")

        for i, config_file in enumerate(config_files, 1):
            print(f"\n[{i}/{len(config_files)}] Validating: {config_file}")

            if test_overrides:
                file_results = self.validate_config_with_overrides(config_file)
                self.results.extend(file_results)
            else:
                result = self.validate_single_config(config_file)
                self.results.append(result)

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report.
        """
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests

        # Calculate performance statistics
        performance_improvements = [
            r.performance_improvement for r in self.results
            if r.success and r.legacy_load_time > 0
        ]
        avg_performance_improvement = (
            sum(performance_improvements) / len(performance_improvements)
            if performance_improvements else 0
        )

        report_lines = [
            "=" * 80,
            "CONFIGURATION MIGRATION VALIDATION REPORT",
            "=" * 80,
            "",
            f"Total tests: {total_tests}",
            f"Successful: {successful_tests}",
            f"Failed: {failed_tests}",
            f"Success rate: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A",
            "",
            f"Average performance improvement: {avg_performance_improvement:.1f}%",
            "",
        ]

        if failed_tests > 0:
            report_lines.extend([
                "FAILED TESTS:",
                "-" * 40,
                ""
            ])

            for result in self.results:
                if not result.success:
                    report_lines.extend([
                        f"❌ {result.config_path}",
                        f"   Error: {result.error_message}",
                        ""
                    ])

                    if result.differences:
                        report_lines.append(f"   Differences ({len(result.differences)}):")
                        for diff in result.differences[:5]:  # Show first 5 differences
                            report_lines.append(f"     • {diff['path']}: {diff['legacy_value']} → {diff['new_value']}")
                        if len(result.differences) > 5:
                            report_lines.append(f"     ... and {len(result.differences) - 5} more")
                        report_lines.append("")

        if successful_tests > 0:
            report_lines.extend([
                "SUCCESSFUL TESTS:",
                "-" * 40,
                ""
            ])

            for result in self.results:
                if result.success:
                    perf_improvement = result.performance_improvement
                    perf_str = f" ({perf_improvement:+.1f}%)" if perf_improvement != 0 else ""
                    report_lines.append(f"✅ {result.config_path}{perf_str}")

        # Performance summary
        if performance_improvements:
            report_lines.extend([
                "",
                "PERFORMANCE SUMMARY:",
                "-" * 40,
                f"Best improvement: {max(performance_improvements):.1f}%",
                f"Worst performance: {min(performance_improvements):.1f}%",
                f"Average improvement: {avg_performance_improvement:.1f}%",
            ])

        report_text = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")

            # Also save detailed JSON report
            json_file = output_file.replace('.txt', '.json')
            with open(json_file, 'w') as f:
                json.dump([r.to_dict() for r in self.results], f, indent=2)
            print(f"Detailed JSON report saved to: {json_file}")

        return report_text


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate configuration migration between old and new systems"
    )
    parser.add_argument(
        '--config', type=str,
        help="Single configuration file to validate"
    )
    parser.add_argument(
        '--all-configs', action='store_true',
        help="Validate all configuration files in the repository"
    )
    parser.add_argument(
        '--no-overrides', action='store_true',
        help="Skip testing with CLI overrides (faster)"
    )
    parser.add_argument(
        '--report-only', action='store_true',
        help="Generate report from existing results without running validation"
    )
    parser.add_argument(
        '--output', type=str, default='migration_validation_report.txt',
        help="Output file for the report"
    )

    args = parser.parse_args()

    if not BOTH_SYSTEMS_AVAILABLE:
        print("Error: Both configuration systems must be available for validation")
        sys.exit(1)

    validator = ConfigMigrationValidator()

    if not args.report_only:
        if args.config:
            # Validate single config
            config_files = [args.config]
        elif args.all_configs:
            # Find all config files
            config_files = validator.find_all_config_files()
            if not config_files:
                print("No configuration files found!")
                sys.exit(1)
        else:
            # Default: validate a few key config files
            config_files = [
                'configs/base/factory_base_v2.yaml',
                'configs/experiments/hybrid_control_exp_v2.yaml'
            ]
            config_files = [f for f in config_files if os.path.exists(f)]

        if not config_files:
            print("No valid configuration files to validate!")
            sys.exit(1)

        print(f"Configuration Migration Validation")
        print(f"Config files to validate: {len(config_files)}")
        print(f"Testing with overrides: {not args.no_overrides}")

        validator.validate_all_configs(config_files, test_overrides=not args.no_overrides)

    # Generate and display report
    report = validator.generate_report(args.output)
    print("\n" + report)

    # Exit with appropriate code
    failed_tests = sum(1 for r in validator.results if not r.success)
    if failed_tests > 0:
        print(f"\n❌ Validation failed: {failed_tests} test(s) failed")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()