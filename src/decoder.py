import numpy as np
import open3d as o3d
import time
from src.raycasting import FACES, get_camera_rays

def decode_point_cloud(height_maps, resolution=512, invert_normals=False):
    """
    Back-projects the height maps into a single dense 3D point cloud.
    If invert_normals is True, surface normals will point along the ray direction (used for back-faces).
    """
    all_points = []
    all_normals = []
    
    for face in FACES:
        hmap = height_maps[face]
        origins, directions = get_camera_rays(face, resolution)
        
        distances = hmap.flatten()
        valid = ~np.isinf(distances) & (distances > 0)
        
        valid_distances = distances[valid]
        valid_origins = origins[valid]
        valid_directions = directions[valid]
        
        points = valid_origins + valid_directions * valid_distances[:, np.newaxis]
        # For external cameras, the surface normal points back at the camera.
        # For internal (back-faces), the surface normal points along the ray direction.
        if invert_normals:
            normals = valid_directions
        else:
            normals = -valid_directions
        
        all_points.append(points)
        all_normals.append(normals)
        
    final_points = np.vstack(all_points)
    final_normals = np.vstack(all_normals)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.normals = o3d.utility.Vector3dVector(final_normals)
    
    return pcd

def reconstruct_mesh_poisson(pcd, depth=9):
    """
    Reconstructs a mesh from the point cloud using Poisson Surface Reconstruction.
    """
    print(f"Running Poisson Surface Reconstruction (depth={depth})...")
    start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    # Clean up artifacts by removing low density vertices
    densities = np.asarray(densities)
    if len(densities) > 0:
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"Poisson reconstruction took {time.time() - start:.2f}s")
    return mesh

def reconstruct_mesh_ball_pivoting(pcd, radii=[0.01, 0.02, 0.04, 0.08]):
    """
    Reconstructs a mesh using Ball Pivoting Algorithm.
    """
    print("Running Ball Pivoting Algorithm...")
    start = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    print(f"BPA took {time.time() - start:.2f}s")
    return mesh
