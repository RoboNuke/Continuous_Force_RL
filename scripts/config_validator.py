#!/usr/bin/env python3
"""
Configuration Validation CLI Tool

A standalone tool for validating configuration files using the new ConfigManagerV2 system.
Provides detailed validation, error reporting, and configuration analysis.

Usage:
    python scripts/config_validator.py config.yaml
    python scripts/config_validator.py config.yaml --verbose
    python scripts/config_validator.py config.yaml --check-overrides "primary.decimation=16"
    python scripts/config_validator.py config.yaml --output validation_report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config_manager_v2 import ConfigManagerV2
    from configs.cfg_exts.primary_cfg import PrimaryConfig
except ImportError as e:
    print(f"Error: Could not import configuration system: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class ConfigValidator:
    """Configuration validation tool with detailed analysis."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'validation_time': None,
            'config_path': None,
            'success': False,
            'error': None,
            'warnings': [],
            'config_info': {},
            'performance': {},
            'override_tests': []
        }

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def validate_config_file(self, config_path: str, cli_overrides: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate a configuration file and return detailed results.

        Args:
            config_path: Path to configuration file
            cli_overrides: Optional CLI overrides to test

        Returns:
            Dictionary with validation results
        """
        self.results['config_path'] = config_path
        cli_overrides = cli_overrides or []

        start_time = time.time()

        try:
            self.log(f"Validating configuration: {config_path}")

            # Check if file exists
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            # Load and validate configuration
            self.log("Loading configuration with new system...")
            bundle = ConfigManagerV2.load_defaults_first_config(
                config_path=config_path,
                cli_overrides=cli_overrides
            )

            # Record basic info
            self.results['config_info'] = {
                'task_name': bundle.task_name,
                'total_agents': bundle.primary_cfg.total_agents,
                'total_envs': bundle.primary_cfg.total_num_envs,
                'episode_length': bundle.env_cfg.episode_length_s,
                'decimation': bundle.env_cfg.decimation,
                'use_hybrid_agent': bundle.model_cfg.use_hybrid_agent,
                'hybrid_config_provided': bundle.hybrid_cfg is not None
            }

            # Performance metrics
            load_time = time.time() - start_time
            self.results['performance'] = {
                'load_time_ms': round(load_time * 1000, 2),
                'config_size_lines': self._count_config_lines(config_path)
            }

            # Validation checks
            self._perform_validation_checks(bundle)

            self.results['success'] = True
            self.log("✅ Configuration validation successful!")

        except Exception as e:
            self.results['error'] = str(e)
            self.results['success'] = False
            self.log(f"❌ Configuration validation failed: {e}", "ERROR")

        self.results['validation_time'] = time.time() - start_time
        return self.results

    def _count_config_lines(self, config_path: str) -> int:
        """Count non-empty lines in configuration file."""
        try:
            with open(config_path, 'r') as f:
                return len([line for line in f if line.strip() and not line.strip().startswith('#')])
        except:
            return 0

    def _perform_validation_checks(self, bundle) -> None:
        """Perform detailed validation checks on the configuration bundle."""

        # Check 1: Primary configuration consistency
        primary = bundle.primary_cfg
        if primary.agents_per_break_force <= 0:
            self.results['warnings'].append("agents_per_break_force should be positive")

        if primary.num_envs_per_agent <= 0:
            self.results['warnings'].append("num_envs_per_agent should be positive")

        if primary.decimation <= 0:
            self.results['warnings'].append("decimation should be positive")

        # Check 2: Computed properties
        total_agents = primary.total_agents
        total_envs = primary.total_num_envs

        if total_agents != total_envs // primary.num_envs_per_agent:
            self.results['warnings'].append("Computed total_agents doesn't match expected calculation")

        # Check 3: Hybrid agent consistency
        if bundle.model_cfg.use_hybrid_agent and bundle.hybrid_cfg is None:
            self.results['warnings'].append("use_hybrid_agent=True but no hybrid config provided")

        if not bundle.model_cfg.use_hybrid_agent and bundle.hybrid_cfg is not None:
            self.results['warnings'].append("Hybrid config provided but use_hybrid_agent=False")

        # Check 4: Wrapper consistency
        if bundle.model_cfg.use_hybrid_agent:
            if not bundle.wrapper_cfg.force_torque_sensor_enabled:
                self.results['warnings'].append("Hybrid agent enabled but force-torque sensor disabled")
            if not bundle.wrapper_cfg.hybrid_control_enabled:
                self.results['warnings'].append("Hybrid agent enabled but hybrid control wrapper disabled")

        # Check 5: Performance considerations
        if total_envs > 10000:
            self.results['warnings'].append(f"Large number of environments ({total_envs}) may impact performance")

        if primary.num_envs_per_agent > 1000:
            self.results['warnings'].append(f"High envs per agent ({primary.num_envs_per_agent}) may impact memory")

        self.log(f"Validation checks complete: {len(self.results['warnings'])} warnings found")

    def test_cli_overrides(self, config_path: str, override_sets: List[List[str]]) -> None:
        """Test configuration with various CLI override combinations."""
        self.log(f"Testing {len(override_sets)} CLI override combinations...")

        for i, overrides in enumerate(override_sets):
            override_result = {
                'overrides': overrides,
                'success': False,
                'error': None,
                'load_time_ms': 0
            }

            try:
                start_time = time.time()
                bundle = ConfigManagerV2.load_defaults_first_config(
                    config_path=config_path,
                    cli_overrides=overrides
                )
                override_result['load_time_ms'] = round((time.time() - start_time) * 1000, 2)
                override_result['success'] = True
                self.log(f"  ✅ Override set {i+1}: {overrides}")

            except Exception as e:
                override_result['error'] = str(e)
                self.log(f"  ❌ Override set {i+1}: {overrides} - {e}", "ERROR")

            self.results['override_tests'].append(override_result)

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a detailed validation report."""

        # Success/failure summary
        status = "✅ PASSED" if self.results['success'] else "❌ FAILED"

        report_lines = [
            "=" * 80,
            f"CONFIGURATION VALIDATION REPORT - {status}",
            "=" * 80,
            "",
            f"Configuration: {self.results['config_path']}",
            f"Validation time: {self.results['validation_time']:.3f}s",
            ""
        ]

        # Error information
        if self.results['error']:
            report_lines.extend([
                "ERROR DETAILS:",
                "-" * 40,
                self.results['error'],
                ""
            ])

        # Configuration information
        if self.results['config_info']:
            info = self.results['config_info']
            report_lines.extend([
                "CONFIGURATION DETAILS:",
                "-" * 40,
                f"Task: {info.get('task_name', 'Unknown')}",
                f"Total agents: {info.get('total_agents', 'Unknown')}",
                f"Total environments: {info.get('total_envs', 'Unknown')}",
                f"Episode length: {info.get('episode_length', 'Unknown')}s",
                f"Decimation: {info.get('decimation', 'Unknown')}",
                f"Hybrid agent: {info.get('use_hybrid_agent', 'Unknown')}",
                ""
            ])

        # Performance metrics
        if self.results['performance']:
            perf = self.results['performance']
            report_lines.extend([
                "PERFORMANCE METRICS:",
                "-" * 40,
                f"Load time: {perf.get('load_time_ms', 0)}ms",
                f"Config size: {perf.get('config_size_lines', 0)} lines",
                ""
            ])

        # Warnings
        if self.results['warnings']:
            report_lines.extend([
                "WARNINGS:",
                "-" * 40,
            ])
            for warning in self.results['warnings']:
                report_lines.append(f"⚠️  {warning}")
            report_lines.append("")

        # CLI override test results
        if self.results['override_tests']:
            successful_overrides = sum(1 for test in self.results['override_tests'] if test['success'])
            total_overrides = len(self.results['override_tests'])

            report_lines.extend([
                "CLI OVERRIDE TESTS:",
                "-" * 40,
                f"Successful: {successful_overrides}/{total_overrides}",
                ""
            ])

            for test in self.results['override_tests']:
                status_icon = "✅" if test['success'] else "❌"
                overrides_str = " ".join(test['overrides']) if test['overrides'] else "none"

                if test['success']:
                    report_lines.append(f"{status_icon} {overrides_str} ({test['load_time_ms']}ms)")
                else:
                    report_lines.append(f"{status_icon} {overrides_str} - {test['error']}")

            report_lines.append("")

        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 40,
        ])

        if self.results['success']:
            report_lines.append("✅ Configuration is valid and ready for use")

            if len(self.results['warnings']) == 0:
                report_lines.append("✅ No warnings or issues found")
            else:
                report_lines.append(f"⚠️  Consider addressing {len(self.results['warnings'])} warnings above")
        else:
            report_lines.append("❌ Fix configuration errors before using")
            report_lines.append("📚 See docs/CONFIGURATION_SYSTEM_V2.md for help")

        report_text = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

            # Also save JSON report
            json_file = output_file.replace('.txt', '.json')
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2)

            self.log(f"Report saved to: {output_file}")
            self.log(f"JSON data saved to: {json_file}")

        return report_text


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Configuration validation tool for ConfigManagerV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/config_validator.py config.yaml
  python scripts/config_validator.py config.yaml --verbose
  python scripts/config_validator.py config.yaml --check-overrides "primary.decimation=16"
  python scripts/config_validator.py config.yaml --test-common-overrides
  python scripts/config_validator.py config.yaml --output report.txt
        """
    )

    parser.add_argument(
        'config_path',
        help="Path to configuration file to validate"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )

    parser.add_argument(
        '--check-overrides',
        action='append',
        help="Test specific CLI overrides (can be used multiple times)"
    )

    parser.add_argument(
        '--test-common-overrides',
        action='store_true',
        help="Test with common CLI override patterns"
    )

    parser.add_argument(
        '--output', '-o',
        help="Save detailed report to file"
    )

    args = parser.parse_args()

    # Create validator
    validator = ConfigValidator(verbose=args.verbose)

    # Validate basic configuration
    cli_overrides = args.check_overrides if args.check_overrides else []
    results = validator.validate_config_file(args.config_path, cli_overrides)

    # Test additional override combinations if requested
    if args.test_common_overrides or args.check_overrides:
        override_sets = []

        if args.test_common_overrides:
            # Common override patterns for testing
            override_sets.extend([
                ["primary.decimation=16"],
                ["primary.debug_mode=true"],
                ["model.use_hybrid_agent=true"],
                ["primary.decimation=12", "primary.debug_mode=true"],
                ["model.actor_latent_size=512", "model.critic_latent_size=1024"],
                ["agent.learning_epochs=8", "agent.policy_learning_rate=5e-7"]
            ])

        if args.check_overrides:
            # Group provided overrides for testing
            override_sets.append(args.check_overrides)

        validator.test_cli_overrides(args.config_path, override_sets)

    # Generate and display report
    report = validator.generate_report(args.output)
    print("\n" + report)

    # Exit with appropriate code
    if not results['success']:
        sys.exit(1)
    elif results['warnings']:
        print(f"\n⚠️  Validation passed with {len(results['warnings'])} warnings")
        sys.exit(0)
    else:
        print("\n✅ Validation passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()