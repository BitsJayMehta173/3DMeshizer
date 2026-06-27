import numpy as np
import trimesh
import open3d as o3d
import time
from src.raycasting import FACES, get_camera_rays
from src.decoder import decode_point_cloud

def encode_multi_cube(mesh, base_res=256, local_res=64, min_faces=50):
    """
    Implements the "Spare Parts" algorithm.
    Extracts parts of the mesh not hit by the global bounding cube cameras
    and encodes them locally.
    """
    print(f"Encoding Multi-Cube (Global Res: {base_res}, Local Res: {local_res})")
    start = time.time()
    
    hit_faces = set()
    global_height_maps = {}
    
    # 1. Global Pass
    for face in FACES:
        origins, directions = get_camera_rays(face, base_res)
        
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
            
            # Record hit faces
            hit_faces.update(index_tri)
            
        global_height_maps[face] = distances.reshape((base_res, base_res))
        
    encoded_data = {
        'global_maps': global_height_maps,
        'spare_parts': []
    }
    
    # 2. Extract unhit faces
    all_faces = set(range(len(mesh.faces)))
    unhit_faces = list(all_faces - hit_faces)
    
    print(f"Global pass hit {len(hit_faces)} faces. {len(unhit_faces)} faces missed.")
    
    if len(unhit_faces) == 0:
        print("No spare parts needed!")
        return encoded_data
        
    # Create submesh for unhit faces
    unhit_mesh = mesh.submesh([unhit_faces], append=True)
    
    # Trimesh submesh can sometimes return a list or Trimesh object
    if isinstance(unhit_mesh, list): 
        if len(unhit_mesh) > 0:
            unhit_mesh = unhit_mesh[0]
        else:
            return encoded_data
            
    if unhit_mesh is None or unhit_mesh.is_empty:
        return encoded_data
        
    # Avoid memory-intensive graph splitting (networkx) by treating all unhit faces as one or a few parts
    # For this prototype, we'll just treat the entire unhit submesh as a single "spare part" bounding volume.
    components = [unhit_mesh]
    print(f"Treating unhit mesh as a single spare part to conserve memory.")
    
    # 3. Local Pass for significant components
    for i, comp in enumerate(components):
        if len(comp.faces) < min_faces:
            continue
            
        # Normalize this local component
        local_mesh = comp.copy()
        bounds = local_mesh.bounds
        center = np.mean(bounds, axis=0)
        local_mesh.apply_translation(-center)
        
        scale = np.max(local_mesh.extents) / 2.0
        if scale == 0: scale = 1e-6
        local_mesh.apply_scale(1.0 / scale)
        
        local_maps = {}
        for face in FACES:
            o, d = get_camera_rays(face, local_res)
            loc, ir, it = local_mesh.ray.intersects_location(
                ray_origins=o, ray_directions=d, multiple_hits=False
            )
            dist = np.full(len(o), np.inf)
            if len(loc) > 0:
                dist[ir] = np.linalg.norm(loc - o[ir], axis=1)
            local_maps[face] = dist.reshape((local_res, local_res))
            
        encoded_data['spare_parts'].append({
            'center': center,
            'scale': scale,
            'maps': local_maps
        })
        
    print(f"Encoded {len(encoded_data['spare_parts'])} significant spare parts.")
    print(f"Multi-Cube Encoding took {time.time() - start:.2f}s")
    return encoded_data

def decode_multi_cube(encoded_data, base_res=256, local_res=64):
    """
    Decodes global and local height maps and fuses them into one point cloud.
    """
    print("Decoding Multi-Cube...")
    
    # Decode global
    global_pcd = decode_point_cloud(encoded_data['global_maps'], resolution=base_res)
    points = [np.asarray(global_pcd.points)]
    normals = [np.asarray(global_pcd.normals)]
    
    # Decode spare parts
    for part in encoded_data['spare_parts']:
        local_pcd = decode_point_cloud(part['maps'], resolution=local_res)
        
        if len(local_pcd.points) == 0:
            continue
            
        local_points = np.asarray(local_pcd.points)
        local_normals = np.asarray(local_pcd.normals)
        
        # Transform back to global space
        scale = part['scale']
        center = part['center']
        
        transformed_points = (local_points * scale) + center
        
        points.append(transformed_points)
        normals.append(local_normals)
        
    final_points = np.vstack(points)
    final_normals = np.vstack(normals)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.normals = o3d.utility.Vector3dVector(final_normals)
    
    print(f"Total fused points: {len(pcd.points)}")
    return pcd
