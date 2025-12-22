"""
Bulk OBJ to Factory USD Converter

Batch converts .obj files to .usd files using IsaacLab factory templates,
and generates manifest entries for peg-hole pairs.

Usage:
    python scripts/bulk_obj_to_factory_usd.py /path/to/input_folder --scale 0.001

    Input folder must contain:
        holes/    - .obj files for hole assets (can have subfolders)
        pegs/     - .obj files for peg assets (can have subfolders)

    Output:
        assets/peg_hole/holes/{relative_path}.usd
        assets/peg_hole/pegs/{relative_path}.usd
        assets/peg_hole/manifest.json (updated)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Need to launch Isaac Sim app first to use USD properly
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Import the shared conversion function (must be after Isaac Sim launch)
from obj_convert_utils import convert_from_template

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "peg_hole"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"


def find_obj_files(folder: Path) -> list:
    """Recursively find all .obj files in a folder."""
    obj_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith('.obj'):
                obj_files.append(Path(root) / f)
    return sorted(obj_files)


def sanitize_prim_name(name: str) -> str:
    """Sanitize name for USD prim compatibility (replace invalid characters)."""
    # USD prim names can't contain '+' and some other special characters
    return name.replace('+', '_')


def get_relative_path(obj_path: Path, base_folder: Path) -> Path:
    """Get the relative path from base folder, replacing .obj with .usd and sanitizing."""
    rel = obj_path.relative_to(base_folder)
    # Sanitize the filename (stem) while preserving directory structure
    sanitized_stem = sanitize_prim_name(rel.stem)
    return rel.with_name(sanitized_stem + '.usd')


def derive_prim_name(obj_path: Path) -> str:
    """Derive prim name from .obj filename (without extension), sanitized for USD."""
    return sanitize_prim_name(obj_path.stem)


def process_holes(
    input_folder: Path,
    mass: float,
    friction: float,
    origin: str,
    scale: float,
    base_height: float,
) -> dict:
    """
    Process all hole .obj files.

    Returns dict mapping hole_name -> hole_info
    """
    holes_folder = input_folder / "holes"
    if not holes_folder.exists():
        raise RuntimeError(f"Holes folder not found: {holes_folder}")

    obj_files = find_obj_files(holes_folder)
    print(f"\nFound {len(obj_files)} hole .obj files")

    holes_info = {}

    for obj_path in obj_files:
        prim_name = derive_prim_name(obj_path)
        rel_path = get_relative_path(obj_path, holes_folder)
        output_path = ASSETS_DIR / "holes" / rel_path
        usd_rel_path = f"holes/{rel_path}"

        print(f"  Converting hole: {obj_path.name} -> {output_path}")

        asset_info = convert_from_template(
            input_path=str(obj_path),
            output_path=str(output_path),
            prim_name=prim_name,
            asset_type="hole",
            mass=mass,
            friction=friction,
            origin=origin,
            scale=scale,
            verbose=False,
        )

        # Calculate diameter from width dimensions (holes don't have 'diameter' key)
        diameter = max(asset_info['width_x'], asset_info['width_y'])

        holes_info[prim_name] = {
            'usd_path': usd_rel_path,
            'prim_name': prim_name,
            'diameter': diameter,
            'height': asset_info['height'],
            'base_height': base_height,
            'mass': mass,
            'friction': friction,
        }

    return holes_info


def find_matching_hole(peg_name: str, holes_info: dict) -> str:
    """
    Find the hole that matches this peg by comparing size suffixes.

    Peg size is extracted from after the last "_" (e.g., arch_short_lrg -> lrg)
    Hole size is extracted from before the first "_" (e.g., lrg_holes -> lrg)

    Raises error if 0 or >1 matches found.
    """
    # Extract peg size (after last underscore)
    if '_' not in peg_name:
        raise RuntimeError(
            f"Peg name '{peg_name}' must contain underscore to extract size suffix"
        )
    peg_size = peg_name.rsplit('_', 1)[1]

    # Find holes with matching size prefix
    matches = []
    for hole_name in holes_info.keys():
        if '_' in hole_name:
            hole_size = hole_name.split('_', 1)[0]
        else:
            hole_size = hole_name
        if hole_size == peg_size:
            matches.append(hole_name)

    if len(matches) == 0:
        raise RuntimeError(
            f"No matching hole found for peg '{peg_name}' (size: '{peg_size}'). "
            f"Available holes: {list(holes_info.keys())}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple holes match peg '{peg_name}' (size: '{peg_size}'): {matches}."
        )

    return matches[0]


def process_pegs(
    input_folder: Path,
    holes_info: dict,
    mass: float,
    friction: float,
    origin: str,
    scale: float,
) -> dict:
    """
    Process all peg .obj files and create manifest entries.

    Returns dict of manifest variants.
    """
    pegs_folder = input_folder / "pegs"
    if not pegs_folder.exists():
        raise RuntimeError(f"Pegs folder not found: {pegs_folder}")

    obj_files = find_obj_files(pegs_folder)
    print(f"\nFound {len(obj_files)} peg .obj files")

    variants = {}

    for obj_path in obj_files:
        prim_name = derive_prim_name(obj_path)
        rel_path = get_relative_path(obj_path, pegs_folder)
        output_path = ASSETS_DIR / "pegs" / rel_path
        usd_rel_path = f"pegs/{rel_path}"

        print(f"  Converting peg: {obj_path.name} -> {output_path}")

        # Find matching hole
        hole_name = find_matching_hole(prim_name, holes_info)
        print(f"    Matched to hole: {hole_name}")

        asset_info = convert_from_template(
            input_path=str(obj_path),
            output_path=str(output_path),
            prim_name=prim_name,
            asset_type="peg",
            mass=mass,
            friction=friction,
            origin=origin,
            scale=scale,
            verbose=False,
        )

        # Create manifest entry
        variants[prim_name] = {
            'peg': {
                'usd_path': usd_rel_path,
                'prim_name': prim_name,
                'diameter': asset_info['diameter'],
                'height': asset_info['height'],
                'mass': mass,
                'friction': friction,
            },
            'hole': holes_info[hole_name],
            'goal_offset': list(asset_info.get('goal_offset', (0.0, 0.0))),
            'clearance': 0.003,  # 3mm default clearance
        }

    return variants


def load_manifest() -> dict:
    """Load existing manifest or return empty structure."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return {
        'version': '1.0',
        'description': 'Peg-Hole asset variant manifest.',
        'variants': {},
    }


def save_manifest(manifest: dict):
    """Save manifest to file."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk convert OBJ files to factory-compatible USD files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all pegs and holes from a folder (mm to meters):
  python scripts/bulk_obj_to_factory_usd.py /path/to/meshes --scale 0.001

  # With custom mass and friction:
  python scripts/bulk_obj_to_factory_usd.py /path/to/meshes --scale 0.001 --mass 0.03 --friction 0.8

  # Replace manifest entirely:
  python scripts/bulk_obj_to_factory_usd.py /path/to/meshes --scale 0.001 --replace-manifest
        """
    )

    parser.add_argument("input_folder", type=str,
                        help="Path to folder containing holes/ and pegs/ subfolders")
    parser.add_argument("--mass", type=float, default=0.05,
                        help="Mass in kg (default: 0.05)")
    parser.add_argument("--friction", type=float, default=0.75,
                        help="Friction coefficient (default: 0.75)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for mesh dimensions (e.g., 0.001 for mm->m)")
    parser.add_argument("--origin", type=str, default="bottom_center",
                        choices=["bottom_center", "center", "bottom_corner"],
                        help="Where to place the mesh origin (default: bottom_center)")
    parser.add_argument("--base-height", type=float, default=0.0,
                        help="Base height for holes (default: 0.0)")
    parser.add_argument("--replace-manifest", action="store_true",
                        help="Replace manifest entirely instead of merging")

    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Bulk OBJ to Factory USD Converter")
    print(f"{'='*60}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {ASSETS_DIR}")
    print(f"Scale: {args.scale}")
    print(f"Mass: {args.mass} kg")
    print(f"Friction: {args.friction}")
    print(f"Origin: {args.origin}")
    print(f"Base height: {args.base_height}")
    print(f"Replace manifest: {args.replace_manifest}")

    # Process holes first
    print(f"\n{'='*60}")
    print("Processing Holes")
    print(f"{'='*60}")
    holes_info = process_holes(
        input_folder=input_folder,
        mass=args.mass,
        friction=args.friction,
        origin=args.origin,
        scale=args.scale,
        base_height=args.base_height,
    )
    print(f"\nProcessed {len(holes_info)} holes")

    # Process pegs
    print(f"\n{'='*60}")
    print("Processing Pegs")
    print(f"{'='*60}")
    variants = process_pegs(
        input_folder=input_folder,
        holes_info=holes_info,
        mass=args.mass,
        friction=args.friction,
        origin=args.origin,
        scale=args.scale,
    )
    print(f"\nProcessed {len(variants)} pegs")

    # Update manifest
    print(f"\n{'='*60}")
    print("Updating Manifest")
    print(f"{'='*60}")

    if args.replace_manifest:
        manifest = {
            'version': '1.0',
            'description': 'Peg-Hole asset variant manifest. Auto-generated by bulk_obj_to_factory_usd.py',
            'variants': variants,
        }
        print(f"Replacing manifest with {len(variants)} new entries")
    else:
        manifest = load_manifest()
        existing_count = len(manifest.get('variants', {}))
        manifest['variants'].update(variants)
        new_count = len(manifest['variants'])
        print(f"Merged {len(variants)} entries into manifest")
        print(f"  Previous entries: {existing_count}")
        print(f"  Total entries: {new_count}")

    save_manifest(manifest)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"  Holes converted: {len(holes_info)}")
    print(f"  Pegs converted: {len(variants)}")
    print(f"  Manifest entries: {len(manifest['variants'])}")

    simulation_app.close()


if __name__ == "__main__":
    main()
