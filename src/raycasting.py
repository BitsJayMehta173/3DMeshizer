import numpy as np
import trimesh

FACES = ['front', 'back', 'left', 'right', 'top', 'bottom']

def normalize_mesh(mesh):
    """
    Centers the mesh at the origin and scales it to fit within a [-1, 1]^3 bounding box.
    Returns the normalized mesh and the original scale and translation for decoding.
    """
    bounds = mesh.bounds
    center = np.mean(bounds, axis=0)
    mesh.apply_translation(-center)
    
    scale = np.max(mesh.extents) / 2.0
    mesh.apply_scale(1.0 / scale)
    
    return mesh, center, scale

def get_camera_rays(face, resolution=512):
    """
    Generates ray origins and directions for a specific cube face.
    The cube spans from -1 to 1 on all axes.
    """
    # Create grid in [-1, 1]
    # To match image coordinates (origin top-left):
    # For a standard front view (XY plane), u maps to X, v maps to -Y.
    grid_1d = np.linspace(-1, 1, resolution)
    u, v = np.meshgrid(grid_1d, grid_1d)
    
    # We want top-left to be (-1, 1) and bottom-right to be (1, -1) for XY.
    # So Y should be reversed.
    v = -v 
    
    N = resolution * resolution
    u = u.flatten()
    v = v.flatten()
    
    origins = np.zeros((N, 3))
    directions = np.zeros((N, 3))
    
    # Place cameras slightly outside the bounding box to avoid precision issues
    dist = 1.01 
    
    if face == 'front': # Look at -Z
        origins[:, 0] = u
        origins[:, 1] = v
        origins[:, 2] = dist
        directions[:, 2] = -1
    elif face == 'back': # Look at +Z
        # If we look from back, to keep left-right correct (so right side of image is object's left)
        # u maps to -X.
        origins[:, 0] = -u
        origins[:, 1] = v
        origins[:, 2] = -dist
        directions[:, 2] = 1
    elif face == 'right': # Look at -X
        # Looking from +X, looking -X.
        # u maps to Z, v maps to Y
        origins[:, 0] = dist
        origins[:, 1] = v
        origins[:, 2] = -u  # right side of image is +Z
        directions[:, 0] = -1
    elif face == 'left': # Look at +X
        origins[:, 0] = -dist
        origins[:, 1] = v
        origins[:, 2] = u
        directions[:, 0] = 1
    elif face == 'top': # Look at -Y
        # Looking down from +Y.
        # u maps to X, v maps to Z (since we look from top, up is -Z)
        origins[:, 0] = u
        origins[:, 1] = dist
        origins[:, 2] = v 
        directions[:, 1] = -1
    elif face == 'bottom': # Look at +Y
        origins[:, 0] = u
        origins[:, 1] = -dist
        origins[:, 2] = -v # up is +Z
        directions[:, 1] = 1
    else:
        raise ValueError(f"Unknown face {face}")
        
    return origins, directions

def cast_rays(mesh, origins, directions):
    """
    Casts rays and returns the distances to the first intersection.
    If no intersection, distance is set to np.inf.
    """
    # trimesh.ray.intersects_location returns (locations, index_ray, index_tri)
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )
    
    distances = np.full(len(origins), np.inf)
    if len(locations) > 0:
        hit_origins = origins[index_ray]
        hit_distances = np.linalg.norm(locations - hit_origins, axis=1)
        distances[index_ray] = hit_distances
        
    return distances
