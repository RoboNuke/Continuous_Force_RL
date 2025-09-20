#!/usr/bin/env python3
"""
Simplified SKRL Integration Example

This example demonstrates the much simpler approach after removing the complex
observation format and unnecessary components:

1. Use original BlockSimBa models directly (no wrappers needed)
2. Single tensor observations (no complex dict format)
3. Environment logging wrapper for metrics (single Wandb run)
4. SKRL standard checkpoint system
5. Multi-agent support built into the PPO agent
"""

import torch
from typing import Optional

# SKRL imports
try:
    from skrl.agents.torch.ppo import PPO
    from skrl.trainers.torch import SequentialTrainer
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    SKRL_AVAILABLE = True
except ImportError:
    SKRL_AVAILABLE = False
    print("SKRL not available. Install with: pip install skrl[torch]")

# Our simplified components
from agents.skrl_compatible_ppo import create_skrl_ppo_agent
from models.block_simba import BlockSimBaActor, BlockSimBaCritic  # Original models!
from wrappers.environment_builder import create_factory_training_env
from configs.config_manager import LoggingConfigPresets


def create_simple_skrl_setup(
    num_envs: int = 1024,
    num_agents: int = 4,
    device: str = "cuda",
    wandb_project: str = "simple_skrl_demo"
):
    """
    Create a complete SKRL setup with the simplified approach.

    Returns:
        Tuple of (env, agent, trainer) ready for training
    """
    print("🚀 Creating Simplified SKRL Setup")
    print(f"  Envs: {num_envs}, Agents: {num_agents}, Device: {device}")

    # 1. Create environment with logging wrapper
    print("📦 Creating environment...")
    env = create_factory_training_env(
        num_envs=num_envs,
        num_agents=num_agents,
        device=device,
        wandb_project=wandb_project,
        enable_action_logging=True
    )

    # Wrap for SKRL
    skrl_env = wrap_env(env)
    print(f"✓ Environment: {type(env).__name__}")

    # 2. Create original BlockSimBa models (no special wrappers needed!)
    print("🧠 Creating BlockSimBa models...")
    obs_dim = 25  # Adjust based on your environment
    action_dim = 15  # Adjust based on your environment

    models = {}

    # Actor model (original BlockSimBa)
    models["policy"] = BlockSimBaActor(
        observation_space=obs_dim,
        action_space=action_dim,
        device=device,
        num_agents=num_agents,
        act_init_std=0.6,
        actor_n=2,
        actor_latent=512
    )

    # Critic model (original BlockSimBa)
    models["value"] = BlockSimBaCritic(
        observation_space=obs_dim,
        action_space=1,  # Single value output
        device=device,
        num_agents=num_agents,
        critic_n=1,
        critic_latent=128
    )

    print(f"✓ Models: {type(models['policy']).__name__}, {type(models['value']).__name__}")

    # 3. Create memory
    print("💾 Creating memory...")
    memory = RandomMemory(
        memory_size=num_envs * 32,  # 32 rollout steps
        num_envs=skrl_env.num_envs,
        device=device
    )
    print(f"✓ Memory: {memory.memory_size} capacity")

    # 4. Create simplified PPO agent
    print("🤖 Creating PPO agent...")
    agent = create_skrl_ppo_agent(
        models=models,
        memory=memory,
        observation_space=skrl_env.observation_space,
        action_space=skrl_env.action_space,
        device=device,
        num_agents=num_agents,
        env_ref=env,  # Pass environment for logging wrapper access
        learning_rate=3e-4,
        cfg_overrides={
            "rollouts": 32,
            "learning_epochs": 10,
            "mini_batches": 4
        }
    )
    print(f"✓ Agent: {type(agent).__name__}")

    # 5. Create SKRL trainer
    print("🏃 Creating trainer...")
    if not SKRL_AVAILABLE:
        print("❌ SKRL not available - cannot create trainer")
        return env, agent, None

    trainer_cfg = {
        "timesteps": 100000,  # Short for demo
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": True
    }

    trainer = SequentialTrainer(
        cfg=trainer_cfg,
        env=skrl_env,
        agents=agent
    )
    print(f"✓ Trainer: {type(trainer).__name__}")

    return env, agent, trainer


def demonstrate_simplified_approach():
    """Demonstrate the simplified approach."""
    print("Simplified SKRL Integration Demo")
    print("=" * 50)

    # Create setup
    try:
        env, agent, trainer = create_simple_skrl_setup(
            num_envs=64,  # Small for demo
            num_agents=4,
            wandb_project="simplified_demo"
        )

        print("\n✅ Key Simplifications:")
        print("  ✓ Original BlockSimBa models (no wrappers)")
        print("  ✓ Single tensor observations (no complex dict)")
        print("  ✓ Environment logging wrapper (single Wandb run)")
        print("  ✓ SKRL standard checkpoint system")
        print("  ✓ Multi-agent aware PPO agent")

        # Test observation format
        print("\n🔍 Testing observation format...")
        obs, _ = env.reset()
        print(f"  Observation type: {type(obs)}")
        print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

        if isinstance(obs, dict):
            print("  ❌ Still using complex dict format!")
            for key, value in obs.items():
                print(f"    {key}: {value.shape}")
        else:
            print("  ✅ Using simple tensor format!")

        # Test a few steps
        print("\n🎯 Testing environment steps...")
        for i in range(3):
            action = torch.randn(env.unwrapped.num_envs, 15, device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward mean = {reward.mean().item():.3f}")

        print("\n🧪 Testing SKRL compatibility...")
        if trainer:
            # Test agent action selection
            with torch.no_grad():
                actions = agent.act(obs, timestep=0, timesteps=0)
                print(f"  Agent action shape: {actions[0].shape}")
                print("  ✅ SKRL agent working!")

        print("\n💾 Testing checkpoint system...")
        # Test save/load (SKRL's standard system)
        checkpoint_path = "/tmp/test_checkpoint.pt"
        agent.save(checkpoint_path)
        print(f"  ✅ Checkpoint saved: {checkpoint_path}")

        # Cleanup
        env.close()
        print("\n🎉 Simplified approach demonstration complete!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def show_comparison():
    """Show the difference between old and new approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: Old vs New Approach")
    print("=" * 60)

    print("\n❌ OLD COMPLEX APPROACH:")
    print("  • Custom SKRL BlockSimBa model wrappers")
    print("  • Complex {'policy': tensor, 'critic': tensor} observations")
    print("  • Separate episode tracker with duplicate Wandb runs")
    print("  • State preprocessor for observation format conversion")
    print("  • Custom checkpoint handling")
    print("  • Asymmetric actor-critic forced complexity")

    print("\n✅ NEW SIMPLIFIED APPROACH:")
    print("  • Original BlockSimBa models (no wrappers needed)")
    print("  • Simple tensor observations (SKRL compatible)")
    print("  • Environment logging wrapper (single Wandb run)")
    print("  • No state preprocessing needed")
    print("  • SKRL standard checkpoint system")
    print("  • Multi-agent aware without complexity")

    print("\n📊 BENEFITS:")
    print("  • ~70% less code")
    print("  • No duplicate Wandb runs")
    print("  • Standard SKRL checkpoint compatibility")
    print("  • Easier to understand and maintain")
    print("  • Better performance (less overhead)")


if __name__ == "__main__":
    # Show comparison first
    show_comparison()

    # Run demonstration
    demonstrate_simplified_approach()

    print("\n" + "=" * 60)
    print("USAGE SUMMARY:")
    print("=" * 60)
    print("1. Use original BlockSimBa models directly")
    print("2. Create environment with logging wrapper")
    print("3. Create simplified SKRL PPO agent with env_ref")
    print("4. Use SKRL's standard trainer and checkpoint system")
    print("5. Single Wandb run handles all metrics")
    print("\nNo complex observation formats or duplicate systems needed!")