"""
Asset Variant Loader

Loads and validates peg/hole USD variants from a JSON manifest.
Follows 'fail quick, fail loud' design - no silent fallbacks.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PegConfig:
    """Configuration for a peg asset."""
    usd_path: str          # Absolute path to USD file
    prim_name: str         # Prim name in USD (for contact sensor paths)
    diameter: float        # meters
    height: float          # meters
    mass: float            # kg
    friction: float


@dataclass
class HoleConfig:
    """Configuration for a hole asset."""
    usd_path: str          # Absolute path to USD file
    prim_name: str         # Prim name in USD (for contact sensor paths)
    diameter: float        # meters
    height: float          # meters
    base_height: float     # meters - height of base below hole
    mass: float            # kg
    friction: float


@dataclass
class PegHoleVariant:
    """Complete peg-hole variant configuration."""
    name: str
    peg: PegConfig
    hole: HoleConfig
    clearance: float       # Difference between hole and peg diameter
    goal_offset: tuple     # (x, y) offset from plate origin to target hole (local frame)


class AssetVariantLoader:
    """
    Loads peg/hole variants from a JSON manifest.

    Fails immediately with clear error messages if:
    - Manifest file not found
    - Manifest JSON is invalid
    - Requested variant not in manifest
    - USD files referenced in manifest don't exist
    - Required fields are missing
    """

    def __init__(self, manifest_path: str):
        """
        Initialize loader with manifest path.

        Args:
            manifest_path: Absolute path to manifest.json

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            json.JSONDecodeError: If manifest is not valid JSON
            ValueError: If manifest is missing required fields
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Asset manifest not found: {manifest_path}. "
                f"Create a manifest.json file in your assets directory."
            )

        self.manifest_path = manifest_path
        self.manifest_dir = os.path.dirname(manifest_path)

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Validate manifest structure
        self._validate_manifest_structure()

    def _validate_manifest_structure(self) -> None:
        """Validate manifest has required top-level structure."""
        if 'variants' not in self.manifest:
            raise ValueError(
                f"Manifest {self.manifest_path} missing 'variants' key. "
                f"Expected structure: {{'variants': {{'variant_name': {{...}}}}}}"
            )

        if not isinstance(self.manifest['variants'], dict):
            raise ValueError(
                f"Manifest 'variants' must be a dictionary, got {type(self.manifest['variants'])}"
            )

        if len(self.manifest['variants']) == 0:
            raise ValueError(
                f"Manifest {self.manifest_path} has no variants defined."
            )

    def get_available_variants(self) -> List[str]:
        """Return list of available variant names."""
        return list(self.manifest['variants'].keys())

    def load_variant(self, variant_name: str) -> PegHoleVariant:
        """
        Load a specific peg/hole variant.

        Args:
            variant_name: Name of variant to load (must be in manifest)

        Returns:
            PegHoleVariant with validated configuration

        Raises:
            KeyError: If variant_name not in manifest
            FileNotFoundError: If USD files don't exist
            ValueError: If variant config is missing required fields
        """
        if variant_name not in self.manifest['variants']:
            available = self.get_available_variants()
            raise KeyError(
                f"Variant '{variant_name}' not found in manifest. "
                f"Available variants: {available}"
            )

        variant_data = self.manifest['variants'][variant_name]

        # Validate and create peg config
        peg_config = self._parse_peg_config(variant_data, variant_name)

        # Validate and create hole config
        hole_config = self._parse_hole_config(variant_data, variant_name)

        # Get clearance (compute if not provided)
        clearance = variant_data.get('clearance', hole_config.diameter - peg_config.diameter)

        # Get goal offset (default to origin if not provided)
        goal_offset_list = variant_data.get('goal_offset', [0.0, 0.0])
        if len(goal_offset_list) != 2:
            raise ValueError(
                f"Variant '{variant_name}' goal_offset must be [x, y], got {goal_offset_list}"
            )
        goal_offset = tuple(goal_offset_list)

        return PegHoleVariant(
            name=variant_name,
            peg=peg_config,
            hole=hole_config,
            clearance=clearance,
            goal_offset=goal_offset
        )

    def _parse_peg_config(self, variant_data: Dict, variant_name: str) -> PegConfig:
        """Parse and validate peg configuration."""
        if 'peg' not in variant_data:
            raise ValueError(f"Variant '{variant_name}' missing 'peg' configuration")

        peg = variant_data['peg']
        required_fields = ['usd_path', 'prim_name', 'diameter', 'height']

        for field in required_fields:
            if field not in peg:
                raise ValueError(
                    f"Variant '{variant_name}' peg config missing required field: {field}"
                )

        # Convert relative path to absolute
        usd_path = os.path.join(self.manifest_dir, peg['usd_path'])

        if not os.path.exists(usd_path):
            raise FileNotFoundError(
                f"Peg USD file not found for variant '{variant_name}': {usd_path}"
            )

        return PegConfig(
            usd_path=usd_path,
            prim_name=peg['prim_name'],
            diameter=peg['diameter'],
            height=peg['height'],
            mass=peg.get('mass', 0.05),
            friction=peg.get('friction', 0.75)
        )

    def _parse_hole_config(self, variant_data: Dict, variant_name: str) -> HoleConfig:
        """Parse and validate hole configuration."""
        if 'hole' not in variant_data:
            raise ValueError(f"Variant '{variant_name}' missing 'hole' configuration")

        hole = variant_data['hole']
        required_fields = ['usd_path', 'prim_name', 'diameter', 'height']

        for field in required_fields:
            if field not in hole:
                raise ValueError(
                    f"Variant '{variant_name}' hole config missing required field: {field}"
                )

        # Convert relative path to absolute
        usd_path = os.path.join(self.manifest_dir, hole['usd_path'])

        if not os.path.exists(usd_path):
            raise FileNotFoundError(
                f"Hole USD file not found for variant '{variant_name}': {usd_path}"
            )

        return HoleConfig(
            usd_path=usd_path,
            prim_name=hole['prim_name'],
            diameter=hole['diameter'],
            height=hole['height'],
            base_height=hole.get('base_height', 0.0),
            mass=hole.get('mass', 0.05),
            friction=hole.get('friction', 0.75)
        )
