"""
Extended Peg Insert Task Configuration

This module defines ExtendedFactoryTaskPegInsertCfg which extends Isaac Lab's
FactoryTaskPegInsertCfg with our custom parameters.
"""

import os
from pathlib import Path
from typing import Optional

from .version_compat import get_isaac_lab_task_imports


def _get_project_root() -> str:
    """
    Find project root by looking for .git directory.

    Returns:
        Absolute path to project root

    Raises:
        RuntimeError: If no .git directory found in parent hierarchy
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / '.git').exists():
            return str(parent)
    raise RuntimeError(
        "Could not find project root (no .git directory found). "
        "Ensure you are running from within the git repository."
    )

# Get Isaac Lab imports with version compatibility
configclass, PegInsert, _, _ = get_isaac_lab_task_imports()
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

try:
    from isaaclab.sensors import ContactSensorCfg
except:
    from omni.isaac.lab.sensors import ContactSensorCfg


def build_contact_sensor_cfg(peg_prim_name: str, hole_prim_name: str) -> ContactSensorCfg:
    """
    Build ContactSensorCfg with dynamic prim names.

    Args:
        peg_prim_name: The prim name of the peg asset (e.g., 'forge_round_peg_8mm')
        hole_prim_name: The prim name of the hole asset (e.g., 'forge_hole_8mm')

    Returns:
        ContactSensorCfg configured for the specified assets

    Note:
        Peg and hole USDs have different nesting structures:
        - Peg: /{prim_name}/mesh (1 level under HeldAsset)
        - Hole: /{hole_name}/{hole_name}/mesh (2 levels under FixedAsset)
        When spawned, these become:
        - Peg: /World/envs/env_.*/HeldAsset/{prim_name}
        - Hole: /World/envs/env_.*/FixedAsset/{hole_name}/{hole_name}
    """
    return ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/HeldAsset/{peg_prim_name}",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=[f"/World/envs/env_.*/FixedAsset/{hole_prim_name}/{hole_prim_name}"],
        track_air_time=True,
    )


@configclass
class ExtendedFactoryTaskPegInsertCfg(PegInsert):
    """
    Extended peg insert task configuration.

    Inherits from Isaac Lab's FactoryTaskPegInsertCfg and adds our custom parameters.
    """
    use_ft_sensor: bool = False
    obs_type: str = ''
    ctrl_type: str = ''
    agent_type: str = ''
    ctrl: ExtendedCtrlCfg = None

    # Asset variant configuration (None = use IsaacLab defaults)
    asset_variant: Optional[str] = None
    asset_manifest_path: Optional[str] = None

    # Goal offset for multi-hole plates (XY offset from plate origin to target hole, local frame)
    goal_offset: tuple = (0.0, 0.0)

    held_fixed_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/forge_round_peg_8mm",
        update_period=0.0,
        history_length=0,  # Track last 10 timesteps of contact history
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/FixedAsset/forge_hole_8mm/forge_hole_8mm"],
        track_air_time=True,  # Enable tracking of time in contact vs air time
    )

    def __post_init__(self):
        self.ctrl = ExtendedCtrlCfg()
        # Note: asset variant is applied later via apply_asset_variant_if_specified()
        # after all config overrides (YAML + CLI) have been applied

    def apply_asset_variant_if_specified(self) -> bool:
        """
        Apply asset variant if both asset_variant and asset_manifest_path are set.

        Call this AFTER all config overrides (YAML + CLI) have been applied.

        Returns:
            True if variant was applied, False if no variant specified
        """
        if self.asset_variant is not None:
            self._apply_asset_variant()
            return True
        return False

    def _apply_asset_variant(self) -> None:
        """
        Apply asset variant configuration.

        Raises:
            ValueError: If asset_variant is set but manifest_path is not
            FileNotFoundError: If manifest or USD files not found
            KeyError: If variant not in manifest
        """
        if self.asset_manifest_path is None:
            raise ValueError(
                f"asset_variant='{self.asset_variant}' specified but asset_manifest_path is None. "
                f"You must specify both asset_variant and asset_manifest_path in your config."
            )

        # Normalize path for backward compatibility:
        # Strip everything before "assets/" to handle old absolute paths from other machines
        manifest_path = self.asset_manifest_path
        if "assets/" in manifest_path:
            manifest_path = "assets/" + manifest_path.split("assets/", 1)[1]

        # Resolve relative paths against project root
        if not os.path.isabs(manifest_path):
            project_root = _get_project_root()
            manifest_path = os.path.join(project_root, manifest_path)

        from configs.asset_variant_loader import AssetVariantLoader

        loader = AssetVariantLoader(manifest_path)
        variant = loader.load_variant(self.asset_variant)

        # Update held asset config (peg) - used by factory_env.py for runtime calculations
        self.held_asset_cfg.usd_path = variant.peg.usd_path
        self.held_asset_cfg.diameter = variant.peg.diameter
        self.held_asset_cfg.height = variant.peg.height
        self.held_asset_cfg.mass = variant.peg.mass
        self.held_asset_cfg.friction = variant.peg.friction

        # Update fixed asset config (hole) - used by factory_env.py for runtime calculations
        self.fixed_asset_cfg.usd_path = variant.hole.usd_path
        self.fixed_asset_cfg.diameter = variant.hole.diameter
        self.fixed_asset_cfg.height = variant.hole.height
        self.fixed_asset_cfg.base_height = variant.hole.base_height
        self.fixed_asset_cfg.mass = variant.hole.mass
        self.fixed_asset_cfg.friction = variant.hole.friction

        # Update ArticulationCfg spawn paths - used by Isaac Sim to load the USD files
        self.held_asset.spawn.usd_path = variant.peg.usd_path
        self.fixed_asset.spawn.usd_path = variant.hole.usd_path

        # Update ArticulationCfg mass properties
        self.held_asset.spawn.mass_props.mass = variant.peg.mass
        self.fixed_asset.spawn.mass_props.mass = variant.hole.mass

        # Update contact sensor with correct prim paths
        self.held_fixed_contact_sensor = build_contact_sensor_cfg(
            peg_prim_name=variant.peg.prim_name,
            hole_prim_name=variant.hole.prim_name
        )

        # Adjust hand_init_pos[2] based on peg height difference from default
        # Default IsaacLab peg height is 0.050m (50mm)
        # Taller pegs need the hand to start higher to avoid spawning inside the hole
        DEFAULT_PEG_HEIGHT = 0.050
        peg_height_diff = variant.peg.height - DEFAULT_PEG_HEIGHT
        if peg_height_diff != 0:
            original_hand_z = self.hand_init_pos[2]
            # hand_init_pos is a list, so we need to modify it in place or reassign
            self.hand_init_pos = [
                self.hand_init_pos[0],
                self.hand_init_pos[1],
                self.hand_init_pos[2] + peg_height_diff
            ]
            print(f"[ASSET VARIANT] Adjusted hand_init_pos[2]: {original_hand_z:.4f} -> {self.hand_init_pos[2]:.4f} "
                  f"(+{peg_height_diff*1000:.1f}mm for taller peg)")

        # Set goal offset for multi-hole plates
        self.goal_offset = variant.goal_offset

        print(f"[ASSET VARIANT] Loaded variant '{variant.name}'")
        print(f"  Peg: {variant.peg.prim_name} (d={variant.peg.diameter*1000:.3f}mm, h={variant.peg.height*1000:.1f}mm)")
        print(f"  Hole: {variant.hole.prim_name} (d={variant.hole.diameter*1000:.3f}mm, h={variant.hole.height*1000:.1f}mm)")
        print(f"  Clearance: {variant.clearance*1000:.4f}mm")
        if variant.goal_offset != (0.0, 0.0):
            print(f"  Goal offset: ({variant.goal_offset[0]*1000:.1f}mm, {variant.goal_offset[1]*1000:.1f}mm)")

    def apply_primary_cfg(self, primary_cfg) -> None:
        """Apply primary configuration values to this task config."""
        self.decimation = primary_cfg.decimation
        if hasattr(self, 'scene') and self.scene is not None:
            if isinstance(self.scene, dict):
                self.scene['num_envs'] = primary_cfg.total_num_envs
            else:
                self.scene.num_envs = primary_cfg.total_num_envs
        self._primary_cfg = primary_cfg