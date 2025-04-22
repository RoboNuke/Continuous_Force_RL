# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskGearMeshCfg, FactoryTaskNutThreadCfg, FactoryTaskPegInsertCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Factory-PegInsert-Local-v0",
    entry_point="envs.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "BroNet_cfg_entry_point": f"{agents.__name__}:BroNet_ppo_cfg.yaml",
        "SimBaNet_ppo_cfg_entry_point": f"{agents.__name__}:SimBaNet_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-GearMesh-Local-v0",
    entry_point="envs.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "SimBaNet_ppo_cfg_entry_point": f"{agents.__name__}:SimBaNet_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-NutThread-Local-v0",
    entry_point="envs.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "SimBaNet_ppo_cfg_entry_point": f"{agents.__name__}:SimBaNet_ppo_cfg.yaml",
    },
)

