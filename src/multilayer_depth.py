import numpy as np
import trimesh
import open3d as o3d
import time
from src.raycasting import FACES, get_camera_rays
from src.decoder import decode_point_cloud

def cast_rays_multilayer(mesh, origins, directions, num_layers=4):
    """
    Casts rays and finds multiple intersections for depth peeling.
    """
    N = len(origins)
    all_distances = []
    
    current_origins = np.copy(origins)
    current_directions = np.copy(directions)
    
    # Track the cumulative distance from the initial camera plane
    cumulative_distances = np.zeros(N)
    
    for layer in range(num_layers):
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=current_origins,
            ray_directions=current_directions,
            multiple_hits=False
        )
        
        layer_distances = np.full(N, np.inf)
        
        if len(locations) > 0:
            hit_origins = current_origins[index_ray]
            hit_distances = np.linalg.norm(locations - hit_origins, axis=1)
            
            layer_distances[index_ray] = cumulative_distances[index_ray] + hit_distances
            
            # Update origins for the next cast to be slightly past the hit point
            # 1e-4 is a tiny epsilon to prevent the ray from instantly hitting the exact same triangle again
            current_origins[index_ray] = locations + current_directions[index_ray] * 1e-4
            cumulative_distances[index_ray] += hit_distances + 1e-4
            
        all_distances.append(layer_distances)
        
    return all_distances

def encode_multilayer(mesh, resolution=256, num_layers=4):
    """
    Encodes the mesh into multiple layers of depth maps.
    Generates num_layers * 6 total height maps.
    """
    print(f"Encoding Multi-Layer (Layers: {num_layers}, Res: {resolution})")
    start = time.time()
    
    encoded_data = {'maps': {}}
    
    for face in FACES:
        o, d = get_camera_rays(face, resolution)
        distances_layers = cast_rays_multilayer(mesh, o, d, num_layers)
        
        for layer, dist in enumerate(distances_layers):
            encoded_data['maps'][f"{face}_L{layer}"] = dist.reshape((resolution, resolution))
            
    print(f"Multi-Layer Encoding took {time.time() - start:.2f}s")
    return encoded_data

def decode_multilayer(encoded_data, resolution=256):
    """
    Decodes all depth peeling layers and perfectly fuses them into a single 
    point cloud with zero seams, because all points share the same camera grid.
    """
    print("Decoding Multi-Layer Data...")
    
    points = []
    normals = []
    
    # Extract by layers
    keys = list(encoded_data['maps'].keys())
    num_layers = max([int(k.split('_L')[1]) for k in keys]) + 1
    
    for layer in range(num_layers):
        layer_maps = {face: encoded_data['maps'][f"{face}_L{layer}"] for face in FACES}
        
        # Odd layers (1, 3, etc) are hitting the back-faces of the mesh cavities, 
        # so their normals should point away from the camera (invert_normals=True).
        is_backface = (layer % 2 == 1)
        local_pcd = decode_point_cloud(layer_maps, resolution=resolution, invert_normals=is_backface)
        
        if len(local_pcd.points) > 0:
            points.append(np.asarray(local_pcd.points))
            normals.append(np.asarray(local_pcd.normals))
            
    final_points = np.vstack(points)
    final_normals = np.vstack(normals)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.normals = o3d.utility.Vector3dVector(final_normals)
    
    print(f"Total fused points from all layers: {len(pcd.points)}")
    return pcd
