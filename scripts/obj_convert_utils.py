"""
OBJ to Factory USD Conversion Utilities

Shared functions for converting OBJ mesh files to USD files using IsaacLab factory templates.
This module assumes Isaac Sim has already been launched by the calling script.

NOTE: Isaac Sim must be launched BEFORE importing this module.
"""

import numpy as np
import os

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Vt
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Factory asset paths
FACTORY_PEG_USD = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_peg_8mm.usd"
FACTORY_HOLE_USD = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_hole_8mm.usd"

# Goal offset coordinates for multi-hole plate
_X1, _X2, _X3 = -0.07412, -0.01589, 0.068915
_Y1, _Y2, _Y3 = -0.071335, -0.01133, 0.065475

# Mapping from peg name prefix to goal offset [x, y]
GOAL_OFFSET_BY_PREFIX = {
    'arch': [_X3, _Y1],
    'circle3': [_X2, _Y1],
    'circle': [_X2, _Y2],
    'hex': [_X1, _Y1],
    'oval': [_X2, _Y3],
    'rect': [_X3, _Y3],
    'sqr+circle': [_X1, _Y3],
    'square2': [_X3, _Y2],
    'star': [_X1, _Y2],
}


def get_goal_offset_for_peg(prim_name: str) -> tuple:
    """
    Get goal offset based on peg name prefix.

    The prefix is extracted as the substring from the beginning to the first "_".
    If no matching prefix is found, returns (0.0, 0.0).

    Args:
        prim_name: The prim name of the peg (e.g., 'arch_short_small')

    Returns:
        tuple: (x_offset, y_offset) in meters
    """
    # Extract prefix (everything before first underscore)
    if '_' in prim_name:
        prefix = prim_name.split('_')[0]
    else:
        prefix = prim_name

    # Look up offset
    if prefix in GOAL_OFFSET_BY_PREFIX:
        offset = GOAL_OFFSET_BY_PREFIX[prefix]
        return (offset[0], offset[1])

    return (0.0, 0.0)


def load_obj(filepath: str) -> tuple:
    """
    Load an OBJ file and return vertices and face indices.
    Handles negative indices (OBJ allows relative indexing from end of list).

    Returns:
        vertices: numpy array of shape (N, 3)
        faces: list of face vertex indices
    """
    vertices = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face_verts = []
                for p in parts[1:]:
                    indices = p.split('/')
                    v_idx = int(indices[0])
                    # Handle negative indices (relative to end of list)
                    if v_idx < 0:
                        v_idx = len(vertices) + v_idx
                    else:
                        v_idx = v_idx - 1  # OBJ is 1-indexed
                    face_verts.append(v_idx)
                faces.append(face_verts)

    vertices = np.array(vertices, dtype=np.float64)
    return vertices, faces


def scale_mesh(vertices: np.ndarray, scale: float) -> np.ndarray:
    """Scale mesh vertices by a uniform factor."""
    return vertices * scale


def recenter_mesh(vertices: np.ndarray, origin: str = "bottom_center") -> tuple:
    """
    Recenter mesh vertices so the origin is at the specified location.

    Returns:
        recentered_vertices: numpy array of shape (N, 3)
        bbox_info: dict with bounding box information
    """
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    size = max_bounds - min_bounds
    center = (min_bounds + max_bounds) / 2

    bbox_info = {
        'original_min': min_bounds.copy(),
        'original_max': max_bounds.copy(),
        'size': size.copy(),
        'original_center': center.copy()
    }

    if origin == "bottom_center":
        translation = np.array([center[0], center[1], min_bounds[2]])
    elif origin == "center":
        translation = center
    elif origin == "bottom_corner":
        translation = min_bounds
    else:
        raise ValueError(f"Unknown origin type: {origin}")

    recentered = vertices - translation

    bbox_info['translation'] = translation
    bbox_info['new_min'] = recentered.min(axis=0)
    bbox_info['new_max'] = recentered.max(axis=0)

    return recentered, bbox_info


def align_widest_to_y(vertices: np.ndarray) -> tuple:
    """
    Rotate mesh so the widest of X/Y dimensions aligns with Y-axis.
    Must be called after recentering so rotation is around the part's Z-axis.

    Returns:
        vertices: possibly rotated vertices
        rotated: bool indicating if rotation was applied
    """
    x_extent = vertices[:, 0].max() - vertices[:, 0].min()
    y_extent = vertices[:, 1].max() - vertices[:, 1].min()

    if x_extent > y_extent:
        # Rotate 90° CCW around Z: (x, y, z) -> (-y, x, z)
        rotated = np.empty_like(vertices)
        rotated[:, 0] = -vertices[:, 1]
        rotated[:, 1] = vertices[:, 0]
        rotated[:, 2] = vertices[:, 2]
        return rotated, True

    return vertices, False


def flatten_faces(faces: list) -> tuple:
    """
    Flatten face data for USD mesh format.
    USD meshes support arbitrary polygons, so no triangulation needed.
    """
    face_vertex_counts = []
    face_vertex_indices = []

    for face in faces:
        if len(face) < 3:
            continue
        face_vertex_counts.append(len(face))
        face_vertex_indices.extend(face)

    return face_vertex_counts, face_vertex_indices


def find_mesh_prim(stage: Usd.Stage) -> Usd.Prim:
    """Find the mesh prim in the stage."""
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return prim
    raise RuntimeError("No mesh prim found in stage")


def find_physics_prim(stage: Usd.Stage) -> Usd.Prim:
    """Find the prim with RigidBodyAPI (where physics properties are)."""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    raise RuntimeError("No prim with RigidBodyAPI found in stage")


def rename_prim_hierarchy(stage: Usd.Stage, old_root_name: str, new_name: str):
    """
    Rename all Xform prims in the USD hierarchy to new_name.

    This ensures every Xform in the hierarchy has the same name.
    """
    root_layer = stage.GetRootLayer()

    # Find the actual root prim
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("No default prim found in stage")

    # Find all Xform prims to rename (collect paths first, then rename)
    prims_to_rename = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Xform":
            prims_to_rename.append(prim.GetPath())

    # Rename from deepest to shallowest to avoid path invalidation
    prims_to_rename.sort(key=lambda p: len(str(p)), reverse=True)

    for prim_path in prims_to_rename:
        prim_spec = root_layer.GetPrimAtPath(prim_path)
        if prim_spec:
            prim_spec.name = new_name

    # Update the default prim
    root_layer.defaultPrim = new_name


def update_joint_body_references(stage: Usd.Stage, old_name: str, new_name: str, verbose: bool = True):
    """
    Update joint body references after prim renaming.

    Joints have physics:body0 and physics:body1 relationships that point to
    specific prims. When we rename prims, these references become stale and
    cause PhysX errors like "CreateJoint - no bodies defined".

    Instead of string manipulation on the old paths, we find the prim with
    RigidBodyAPI and set body1 to point directly to it. This handles the
    complex renaming where ALL Xforms (including /Root) get renamed.

    Args:
        stage: The USD stage to modify
        old_name: The old prim name (unused, kept for API compatibility)
        new_name: The new prim name (unused, kept for API compatibility)
        verbose: Whether to print debug information
    """
    # Find the prim with RigidBodyAPI - this is what body1 should point to
    rigidbody_prim_path = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigidbody_prim_path = prim.GetPath()
            if verbose:
                print(f"    Found RigidBody prim: {rigidbody_prim_path}")
            break

    if not rigidbody_prim_path:
        if verbose:
            print("    WARNING: No RigidBodyAPI prim found!")
        return

    # Update joints to point to the RigidBody prim
    for prim in stage.Traverse():
        # Check if this is any kind of joint (FixedJoint, RevoluteJoint, etc.)
        if 'Joint' in prim.GetTypeName():
            if verbose:
                print(f"    Found joint: {prim.GetPath()} ({prim.GetTypeName()})")

            body1_rel = prim.GetRelationship('physics:body1')
            if body1_rel:
                old_targets = body1_rel.GetTargets()
                if verbose:
                    print(f"      physics:body1 old: {old_targets}")

                # Set body1 to point to the RigidBody prim
                body1_rel.SetTargets([rigidbody_prim_path])
                if verbose:
                    print(f"      physics:body1 new: [{rigidbody_prim_path}]")


def convert_from_template(
    input_path: str,
    output_path: str,
    prim_name: str,
    asset_type: str,
    mass: float,
    friction: float,
    origin: str = "bottom_center",
    scale: float = 1.0,
    verbose: bool = True
) -> dict:
    """
    Convert OBJ file to factory-compatible USD using factory template.

    This loads the actual IsaacLab factory USD as a template, ensuring all
    physics settings match exactly. We only replace the mesh geometry and
    rename the prims.

    Args:
        input_path: Path to input OBJ file
        output_path: Path to output USD file
        prim_name: Name for the USD prim
        asset_type: 'peg' or 'hole'
        mass: Mass in kg
        friction: Friction coefficient
        origin: Where to place mesh origin ('bottom_center', 'center', 'bottom_corner')
        scale: Scale factor for mesh dimensions
        verbose: Whether to print progress messages

    Returns:
        dict with asset info (prim_name, type, dimensions, mass, friction, diameter)
    """
    def log(msg):
        if verbose:
            print(msg)

    log(f"\n{'='*60}")
    log(f"OBJ to Factory USD Converter (Template Mode)")
    log(f"{'='*60}")
    log(f"Input OBJ: {input_path}")
    log(f"Output USD: {output_path}")
    log(f"Prim name: {prim_name}")
    log(f"Asset type: {asset_type}")
    log(f"Mass: {mass} kg")
    log(f"Friction: {friction}")
    log(f"Origin: {origin}")
    log(f"Scale: {scale}")

    # Select template based on asset type
    if asset_type == "peg":
        template_path = FACTORY_PEG_USD
        old_prim_name = "factory_peg_8mm"
    elif asset_type == "hole":
        template_path = FACTORY_HOLE_USD
        old_prim_name = "factory_hole_8mm"
    else:
        raise ValueError(f"Unknown asset type: {asset_type}. Use 'peg' or 'hole'.")

    log(f"\nTemplate: {template_path}")

    # Load OBJ
    log(f"\nLoading OBJ file...")
    vertices, faces = load_obj(input_path)
    log(f"  Loaded {len(vertices)} vertices, {len(faces)} faces")

    # Scale mesh if needed
    if scale != 1.0:
        log(f"\nScaling mesh by {scale}...")
        original_bounds = vertices.max(axis=0) - vertices.min(axis=0)
        log(f"  Original size: {original_bounds}")
        vertices = scale_mesh(vertices, scale)
        new_bounds = vertices.max(axis=0) - vertices.min(axis=0)
        log(f"  Scaled size: {new_bounds}")

    # Recenter mesh
    log(f"\nRecentering mesh (origin: {origin})...")
    vertices, bbox_info = recenter_mesh(vertices, origin)
    log(f"  Size: {bbox_info['size']}")
    log(f"  New bounds: {bbox_info['new_min']} to {bbox_info['new_max']}")

    # Align widest dimension to Y-axis
    vertices, was_rotated = align_widest_to_y(vertices)
    if was_rotated:
        log(f"\n  Rotated 90° to align widest dimension with Y-axis")
        # Update bbox_info after rotation
        bbox_info['size'] = np.array([bbox_info['size'][1], bbox_info['size'][0], bbox_info['size'][2]])
        bbox_info['new_min'] = vertices.min(axis=0)
        bbox_info['new_max'] = vertices.max(axis=0)
        log(f"  New size after rotation: {bbox_info['size']}")

    # Load the factory template
    log(f"\nLoading factory template...")
    template_stage = Usd.Stage.Open(template_path)
    if not template_stage:
        raise RuntimeError(f"Failed to open template: {template_path}")

    # Export to output path (creates a copy)
    log(f"  Exporting copy to: {output_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log(f"  Created output directory: {output_dir}")

    template_stage.GetRootLayer().Export(output_path)

    # Open the copy for editing
    log(f"\nModifying USD...")
    stage = Usd.Stage.Open(output_path)

    # Find and update the mesh
    log(f"  Replacing mesh geometry...")
    mesh_prim = find_mesh_prim(stage)
    mesh = UsdGeom.Mesh(mesh_prim)

    # Prepare mesh data
    face_vertex_counts, face_vertex_indices = flatten_faces(faces)
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]

    # Replace mesh geometry
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))

    # Clear normals to let renderer compute them (preserves sharp edges)
    normals_attr = mesh.GetNormalsAttr()
    if normals_attr:
        normals_attr.Clear()

    # Clear material binding to avoid stale references from template
    # The template mesh has material bindings pointing to the old template paths
    # which become invalid after we rename prims
    log(f"  Clearing material bindings...")
    material_binding = UsdShade.MaterialBindingAPI(mesh_prim)
    if material_binding:
        # Remove direct binding
        material_binding.UnbindDirectBinding()
        # Also clear any collection-based bindings
        for binding in material_binding.GetDirectBindingRel().GetTargets():
            material_binding.GetDirectBindingRel().ClearTargets(True)
            break

    # Update mass if specified
    log(f"  Updating mass to {mass} kg...")
    physics_prim = find_physics_prim(stage)
    mass_api = UsdPhysics.MassAPI(physics_prim)
    if mass_api:
        mass_api.GetMassAttr().Set(mass)

    # Update friction on collision mesh
    log(f"  Updating friction to {friction}...")
    material_api = UsdPhysics.MaterialAPI(mesh_prim)
    if material_api:
        static_friction_attr = material_api.GetStaticFrictionAttr()
        dynamic_friction_attr = material_api.GetDynamicFrictionAttr()
        if static_friction_attr:
            static_friction_attr.Set(friction)
        if dynamic_friction_attr:
            dynamic_friction_attr.Set(friction)

    # Save before renaming (renaming requires working with Sdf layer)
    stage.Save()

    # Rename prims to new name
    if prim_name != old_prim_name:
        log(f"  Renaming prims: {old_prim_name} -> {prim_name}...")
        rename_prim_hierarchy(stage, old_prim_name, prim_name)
        log(f"  Updating joint body references...")
        update_joint_body_references(stage, old_prim_name, prim_name)
        stage.Save()

    log(f"\n  Saved to: {output_path}")

    # Compute asset info
    size = bbox_info['size']
    asset_info = {
        'prim_name': prim_name,
        'type': asset_type,
        'width_x': float(size[0]),
        'width_y': float(size[1]),
        'height': float(size[2]),
        'mass': mass,
        'friction': friction,
    }

    if asset_type == "peg":
        asset_info['diameter'] = float(max(size[0], size[1]))
        # Get goal offset based on peg name prefix
        goal_offset = get_goal_offset_for_peg(prim_name)
        asset_info['goal_offset'] = goal_offset

    log(f"\n{'='*60}")
    log(f"Asset Info (for manifest.json):")
    log(f"{'='*60}")
    log(f"  prim_name: \"{prim_name}\"")
    if asset_type == "peg":
        log(f"  diameter: {asset_info['diameter']:.6f}  # meters ({asset_info['diameter']*1000:.3f}mm)")
        if asset_info.get('goal_offset', (0.0, 0.0)) != (0.0, 0.0):
            gx, gy = asset_info['goal_offset']
            log(f"  goal_offset: [{gx:.6f}, {gy:.6f}]  # meters ({gx*1000:.2f}mm, {gy*1000:.2f}mm)")
    log(f"  height: {asset_info['height']:.6f}  # meters ({asset_info['height']*1000:.3f}mm)")
    log(f"  mass: {mass}")
    log(f"  friction: {friction}")
    log(f"{'='*60}\n")

    return asset_info
