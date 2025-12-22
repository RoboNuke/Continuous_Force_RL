"""
OBJ to Factory USD Converter

Converts OBJ mesh files to USD files by using IsaacLab factory assets as templates.
This ensures all physics settings match exactly - we only replace the mesh geometry.

Usage:
    # For a peg asset:
    python scripts/obj_to_factory_usd.py input.obj output.usdc --prim-name my_peg --type peg

    # For a hole asset:
    python scripts/obj_to_factory_usd.py input.obj output.usdc --prim-name my_hole --type hole

    # With custom mass and scale:
    python scripts/obj_to_factory_usd.py input.obj output.usdc --prim-name my_peg --type peg --mass 0.05 --scale 0.001
"""

import argparse
import os
import sys

# Need to launch Isaac Sim app first to use USD properly
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Import conversion utilities (must be after Isaac Sim launch)
from obj_convert_utils import convert_from_template


def main():
    parser = argparse.ArgumentParser(
        description="Convert OBJ mesh to factory-compatible USD using IsaacLab templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert peg mesh:
  python scripts/obj_to_factory_usd.py meshes/my_peg.obj assets/pegs/my_peg.usdc --prim-name my_peg --type peg

  # Convert hole mesh with custom properties:
  python scripts/obj_to_factory_usd.py meshes/my_hole.obj assets/holes/my_hole.usdc --prim-name my_hole --type hole --mass 0.1 --friction 0.8

  # Convert mesh from mm to meters (scale by 0.001):
  python scripts/obj_to_factory_usd.py meshes/peg_mm.obj assets/pegs/peg.usdc --prim-name peg --type peg --scale 0.001
        """
    )

    parser.add_argument("input", type=str, help="Path to input OBJ file")
    parser.add_argument("output", type=str, help="Path to output USD file (.usd or .usdc)")
    parser.add_argument("--prim-name", type=str, required=True,
                        help="Name for the USD prim (e.g., 'my_peg', 'small_hole')")
    parser.add_argument("--type", type=str, required=True, choices=["peg", "hole"],
                        help="Asset type: 'peg' (for HeldAsset) or 'hole' (for FixedAsset)")
    parser.add_argument("--mass", type=float, default=0.05,
                        help="Mass in kg (default: 0.05)")
    parser.add_argument("--friction", type=float, default=0.75,
                        help="Friction coefficient (default: 0.75)")
    parser.add_argument("--origin", type=str, default="bottom_center",
                        choices=["bottom_center", "center", "bottom_corner"],
                        help="Where to place the mesh origin (default: bottom_center)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for mesh dimensions (e.g., 0.001 to convert mm to meters)")

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Convert
    asset_info = convert_from_template(
        input_path=args.input,
        output_path=args.output,
        prim_name=args.prim_name,
        asset_type=args.type,
        mass=args.mass,
        friction=args.friction,
        origin=args.origin,
        scale=args.scale
    )

    simulation_app.close()
    return asset_info


if __name__ == "__main__":
    main()
