import numpy as np
import trimesh
import open3d as o3d
import time
from src.raycasting import FACES, get_camera_rays
from src.decoder import decode_point_cloud

def encode_bisection(mesh, resolution=256):
    """
    Slices the mesh vertically down the middle (X=0) and encodes both halves 
    independently into their own bounding cubes.
    """
    print(f"Encoding Bisection (Res: {resolution})")
    start = time.time()
    
    # slice_mesh_plane keeps the geometry on the positive side of the normal (or negative depending on internal definition)
    # trimesh docs: keeps the side of the mesh where dot(normal, (v - origin)) < 0
    # So normal [1,0,0] keeps X < 0 (Left side). normal [-1,0,0] keeps X > 0 (Right side).
    print("Slicing mesh into two halves along X=0 plane...")
    left_mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[1, 0, 0], plane_origin=[0, 0, 0])
    right_mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[-1, 0, 0], plane_origin=[0, 0, 0])
    
    encoded_data = {'halves': []}
    
    for half_name, half_mesh in [('Left', left_mesh), ('Right', right_mesh)]:
        if half_mesh is None or half_mesh.is_empty:
            print(f"Warning: {half_name} half is empty!")
            continue
            
        print(f"Encoding {half_name} Half (Vertices: {len(half_mesh.vertices)})...")
        bounds = half_mesh.bounds
        center = np.mean(bounds, axis=0)
        half_mesh.apply_translation(-center)
        
        scale = np.max(half_mesh.extents) / 2.0
        if scale == 0: scale = 1e-6
        half_mesh.apply_scale(1.0 / scale)
        
        height_maps = {}
        for face in FACES:
            o, d = get_camera_rays(face, resolution)
            loc, ir, it = half_mesh.ray.intersects_location(
                ray_origins=o, ray_directions=d, multiple_hits=False
            )
            dist = np.full(len(o), np.inf)
            if len(loc) > 0:
                dist[ir] = np.linalg.norm(loc - o[ir], axis=1)
            height_maps[face] = dist.reshape((resolution, resolution))
            
        encoded_data['halves'].append({
            'name': half_name,
            'center': center,
            'scale': scale,
            'maps': height_maps
        })
        
    print(f"Bisection Encoding took {time.time() - start:.2f}s")
    return encoded_data

def decode_bisection(encoded_data, resolution=256):
    """
    Decodes both halves and fuses them back together.
    """
    print("Decoding Bisection Data...")
    
    points = []
    normals = []
    
    for half in encoded_data['halves']:
        local_pcd = decode_point_cloud(half['maps'], resolution=resolution)
        if len(local_pcd.points) == 0:
            continue
            
        local_points = np.asarray(local_pcd.points)
        local_normals = np.asarray(local_pcd.normals)
        
        scale = half['scale']
        center = half['center']
        
        transformed_points = (local_points * scale) + center
        
        points.append(transformed_points)
        normals.append(local_normals)
        
    final_points = np.vstack(points)
    final_normals = np.vstack(normals)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.normals = o3d.utility.Vector3dVector(final_normals)
    
    print(f"Total fused points from both halves: {len(pcd.points)}")
    return pcd
