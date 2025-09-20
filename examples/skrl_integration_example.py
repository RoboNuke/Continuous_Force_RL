#!/usr/bin/env python3
"""
Complete SKRL Integration Example

This example demonstrates how to use all the new SKRL-compatible components
together for a complete training pipeline. It showcases:

1. Environment building with configurable wrappers
2. SKRL-compatible BlockSimBa models
3. Asymmetric actor-critic observation handling
4. Comprehensive logging and metrics tracking
5. Full training and evaluation pipeline

This serves as both a working example and a template for new projects.
"""

import os
import torch
import argparse
from pathlib import Path
from typing import Optional

# Import our new SKRL-compatible components
from learning.skrl_training_config import (
    SKRLTrainingSystem,
    create_factory_training_system,
    create_research_training_system,
    TrainingConfig,
    EnvironmentBuilderConfig,
    ModelConfig,
    PPOConfig
)
from wrappers.environment_builder import (
    EnvironmentBuilder,
    create_factory_training_env,
    create_research_env,
    create_minimal_env
)
from models.skrl_block_simba import (
    create_skrl_blockimba_models,
    create_symmetric_models
)
from agents.skrl_compatible_ppo import SkrlCompatiblePPO
# Note: LoggingConfigPresets and ActionLoggingConfig have been removed
# Enhanced action logging now uses simple boolean parameters


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SKRL Integration Example")

    # Training mode
    parser.add_argument("--mode", type=str, default="demo",
                      choices=["demo", "train", "evaluate", "test_env"],
                      help="Mode to run the example in")

    # Environment settings
    parser.add_argument("--num_envs", type=int, default=64,
                      help="Number of parallel environments")
    parser.add_argument("--num_agents", type=int, default=4,
                      help="Number of agents")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda/cpu)")

    # Training settings
    parser.add_argument("--total_timesteps", type=int, default=100000,
                      help="Total training timesteps")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                      help="Path to load checkpoint from")
    parser.add_argument("--save_path", type=str, default="./checkpoints",
                      help="Path to save checkpoints")

    # Logging settings
    parser.add_argument("--wandb_project", type=str, default="skrl_integration_demo",
                      help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                      help="Wandb entity")
    parser.add_argument("--disable_logging", action="store_true",
                      help="Disable Wandb logging")

    # Model settings
    parser.add_argument("--symmetric_obs", action="store_true",
                      help="Use symmetric actor-critic observations")

    return parser.parse_args()


class SKRLIntegrationDemo:
    """
    Complete demonstration of SKRL integration with our new components.

    This class provides a comprehensive example of how to use all the new
    SKRL-compatible components together in a real training scenario.
    """

    def __init__(self, args):
        """Initialize the demo with command line arguments."""
        self.args = args
        self.training_system = None

        # Setup paths
        self.checkpoint_dir = Path(args.save_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print("SKRL Integration Demo")
        print("=" * 50)
        print(f"Mode: {args.mode}")
        print(f"Device: {args.device}")
        print(f"Num envs: {args.num_envs}")
        print(f"Num agents: {args.num_agents}")
        print(f"Symmetric obs: {args.symmetric_obs}")
        print()

    def demo_environment_builder(self):
        """Demonstrate the environment builder system."""
        print("🔧 Demonstrating Environment Builder")
        print("-" * 40)

        # Example 1: Factory training environment
        print("1. Creating factory training environment...")
        try:
            factory_env = create_factory_training_env(
                num_envs=self.args.num_envs,
                num_agents=self.args.num_agents,
                device=self.args.device,
                wandb_project=self.args.wandb_project if not self.args.disable_logging else None,
                wandb_entity=self.args.wandb_entity,
                enable_action_logging=not self.args.disable_logging
            )

            print(f"   ✓ Factory env created: {type(factory_env).__name__}")
            print(f"   ✓ Num envs: {factory_env.unwrapped.num_envs}")
            print(f"   ✓ Device: {factory_env.unwrapped.device}")

            # Test observation format
            obs, info = factory_env.reset()
            if isinstance(obs, dict):
                print(f"   ✓ Asymmetric observations: policy={obs['policy'].shape}, critic={obs['critic'].shape}")
            else:
                print(f"   ✓ Single tensor observations: {obs.shape}")

            factory_env.close()

        except Exception as e:
            print(f"   ✗ Factory env failed: {e}")

        # Example 2: Research environment
        print("\n2. Creating research environment...")
        try:
            research_env = create_research_env(
                num_envs=min(self.args.num_envs, 32),  # Smaller for research
                device=self.args.device,
                wandb_project=f"{self.args.wandb_project}_research" if not self.args.disable_logging else None,
                track_observations=not self.args.disable_logging
            )

            print(f"   ✓ Research env created: {type(research_env).__name__}")
            research_env.close()

        except Exception as e:
            print(f"   ✗ Research env failed: {e}")

        # Example 3: Minimal environment (for testing)
        print("\n3. Creating minimal environment...")
        try:
            minimal_env = create_minimal_env(
                num_envs=16,
                device=self.args.device
            )

            print(f"   ✓ Minimal env created: {type(minimal_env).__name__}")
            minimal_env.close()

        except Exception as e:
            print(f"   ✗ Minimal env failed: {e}")

    def demo_models(self):
        """Demonstrate the SKRL-compatible model creation."""
        print("\n🧠 Demonstrating SKRL-Compatible Models")
        print("-" * 40)

        # Example 1: Asymmetric actor-critic models
        print("1. Creating asymmetric actor-critic models...")
        try:
            asymmetric_models = create_skrl_blockimba_models(
                policy_obs_dim=20,
                critic_obs_dim=25,
                action_dim=15,
                num_agents=self.args.num_agents,
                device=self.args.device
            )

            print(f"   ✓ Actor model: {type(asymmetric_models['policy']).__name__}")
            print(f"   ✓ Critic model: {type(asymmetric_models['value']).__name__}")

            # Test model forward pass
            batch_size = 16
            test_obs = {
                "policy": torch.randn(batch_size, 20, device=self.args.device),
                "critic": torch.randn(batch_size, 25, device=self.args.device)
            }

            # Test actor
            action_mean, log_std, _ = asymmetric_models["policy"].act({"states": test_obs}, role="policy")
            print(f"   ✓ Actor output: actions={action_mean.shape}, log_std={log_std.shape}")

            # Test critic
            values, _ = asymmetric_models["value"].act({"states": test_obs}, role="value")
            print(f"   ✓ Critic output: values={values.shape}")

        except Exception as e:
            print(f"   ✗ Asymmetric models failed: {e}")

        # Example 2: Symmetric models
        print("\n2. Creating symmetric actor-critic models...")
        try:
            symmetric_models = create_symmetric_models(
                obs_dim=20,
                action_dim=15,
                num_agents=self.args.num_agents,
                device=self.args.device
            )

            print(f"   ✓ Symmetric models created successfully")

            # Test with single tensor observation
            test_obs_tensor = torch.randn(batch_size, 20, device=self.args.device)
            action_mean, log_std, _ = symmetric_models["policy"].act({"states": test_obs_tensor}, role="policy")
            values, _ = symmetric_models["value"].act({"states": test_obs_tensor}, role="value")

            print(f"   ✓ Tensor input test: actions={action_mean.shape}, values={values.shape}")

        except Exception as e:
            print(f"   ✗ Symmetric models failed: {e}")

    def demo_training_system(self):
        """Demonstrate the complete training system."""
        print("\n🚀 Demonstrating Complete Training System")
        print("-" * 40)

        # Create training system
        print("1. Creating SKRL training system...")
        try:
            if self.args.symmetric_obs:
                self.training_system = create_research_training_system(
                    num_envs=self.args.num_envs,
                    total_timesteps=self.args.total_timesteps,
                    device=self.args.device,
                    wandb_project=self.args.wandb_project if not self.args.disable_logging else None,
                    symmetric_obs=True
                )
            else:
                self.training_system = create_factory_training_system(
                    num_envs=self.args.num_envs,
                    num_agents=self.args.num_agents,
                    total_timesteps=self.args.total_timesteps,
                    device=self.args.device,
                    wandb_project=self.args.wandb_project if not self.args.disable_logging else None,
                    wandb_entity=self.args.wandb_entity
                )

            print("   ✓ Training system created")

        except Exception as e:
            print(f"   ✗ Training system creation failed: {e}")
            return

        # Setup training system
        print("2. Setting up training components...")
        try:
            self.training_system.setup()
            print("   ✓ Training system setup complete")

        except Exception as e:
            print(f"   ✗ Training system setup failed: {e}")
            return

        # Load checkpoint if specified
        if self.args.checkpoint_path:
            print("3. Loading checkpoint...")
            try:
                self.training_system.load_checkpoint(self.args.checkpoint_path)
                print(f"   ✓ Checkpoint loaded from {self.args.checkpoint_path}")
            except Exception as e:
                print(f"   ✗ Checkpoint loading failed: {e}")

    def run_training(self):
        """Run the training process."""
        if self.training_system is None:
            print("Training system not set up. Cannot train.")
            return

        print("\n🏃 Running Training")
        print("-" * 40)

        try:
            # Run training
            self.training_system.train()

            # Save final checkpoint
            final_checkpoint = self.checkpoint_dir / "final_model.pt"
            self.training_system.save_checkpoint(str(final_checkpoint))
            print(f"✓ Final model saved to {final_checkpoint}")

        except Exception as e:
            print(f"✗ Training failed: {e}")

    def run_evaluation(self):
        """Run evaluation episodes."""
        if self.training_system is None:
            print("Training system not set up. Cannot evaluate.")
            return

        print("\n📊 Running Evaluation")
        print("-" * 40)

        try:
            # Run evaluation
            avg_reward = self.training_system.evaluate(num_episodes=10)
            print(f"✓ Evaluation complete: Average reward = {avg_reward:.2f}")

        except Exception as e:
            print(f"✗ Evaluation failed: {e}")

    def test_environment_only(self):
        """Test environment creation and basic functionality."""
        print("\n🧪 Testing Environment Only")
        print("-" * 40)

        try:
            # Create minimal environment for testing
            env = create_minimal_env(
                num_envs=self.args.num_envs,
                device=self.args.device
            )

            print(f"✓ Environment created: {type(env).__name__}")
            print(f"✓ Num envs: {env.unwrapped.num_envs}")
            print(f"✓ Device: {env.unwrapped.device}")

            # Test reset
            obs, info = env.reset()
            print(f"✓ Reset successful: obs type = {type(obs)}")

            if isinstance(obs, dict):
                for key, value in obs.items():
                    print(f"   {key}: {value.shape} ({value.dtype})")
            else:
                print(f"   obs: {obs.shape} ({obs.dtype})")

            # Test step
            action_dim = env.action_space.shape[-1]
            actions = torch.randn(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)

            obs, reward, terminated, truncated, info = env.step(actions)
            print(f"✓ Step successful: reward shape = {reward.shape}")

            env.close()
            print("✓ Environment test completed successfully")

        except Exception as e:
            print(f"✗ Environment test failed: {e}")

    def run(self):
        """Run the demo based on the specified mode."""
        if self.args.mode == "demo":
            # Run all demonstrations
            self.demo_environment_builder()
            self.demo_models()
            self.demo_training_system()
            print("\n✅ Demo completed successfully!")

        elif self.args.mode == "train":
            # Setup and run training
            self.demo_training_system()
            self.run_training()

        elif self.args.mode == "evaluate":
            # Setup and run evaluation
            self.demo_training_system()
            self.run_evaluation()

        elif self.args.mode == "test_env":
            # Test environment only
            self.test_environment_only()

        # Cleanup
        if self.training_system:
            self.training_system.close()


def main():
    """Main entry point."""
    args = parse_args()

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    # Run demo
    demo = SKRLIntegrationDemo(args)
    demo.run()


if __name__ == "__main__":
    # Example usage scenarios
    print("SKRL Integration Example - Usage Scenarios")
    print("=" * 50)
    print("1. Demo mode (show all components):")
    print("   python examples/skrl_integration_example.py --mode demo --num_envs 32")
    print()
    print("2. Training mode:")
    print("   python examples/skrl_integration_example.py --mode train --num_envs 1024 --total_timesteps 1000000")
    print()
    print("3. Evaluation mode:")
    print("   python examples/skrl_integration_example.py --mode evaluate --checkpoint_path ./checkpoints/model.pt")
    print()
    print("4. Environment test mode:")
    print("   python examples/skrl_integration_example.py --mode test_env --num_envs 64")
    print()
    print("5. Research setup (symmetric observations):")
    print("   python examples/skrl_integration_example.py --mode demo --symmetric_obs --num_agents 1")
    print()

    # Run if called directly
    main()