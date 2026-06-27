import numpy as np
import trimesh
import open3d as o3d
import time
from src.raycasting import FACES, get_camera_rays
from src.decoder import decode_point_cloud

def count_leaf_nodes(node):
    if node is None: return 0
    if node['is_leaf']: return 1
    return count_leaf_nodes(node['left']) + count_leaf_nodes(node['right'])

def encode_recursive(mesh, depth=0, max_depth=2, min_missed_faces=1000, resolution=128):
    """
    Intelligent recursive KD-tree partitioning based on occluded faces.
    """
    if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
        return None
        
    bounds = mesh.bounds
    center = np.mean(bounds, axis=0)
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    
    scale = np.max(mesh.extents) / 2.0
    if scale == 0: scale = 1e-6
    mesh.apply_scale(1.0 / scale)
    
    hit_faces = set()
    height_maps = {}
    
    for face in FACES:
        o, d = get_camera_rays(face, resolution)
        loc, ir, it = mesh.ray.intersects_location(
            ray_origins=o, ray_directions=d, multiple_hits=False
        )
        dist = np.full(len(o), np.inf)
        if len(loc) > 0:
            dist[ir] = np.linalg.norm(loc - o[ir], axis=1)
            hit_faces.update(it)
        height_maps[face] = dist.reshape((resolution, resolution))
        
    missed_count = len(mesh.faces) - len(hit_faces)
    indent = "  " * depth
    print(f"{indent}Depth {depth}: Missed {missed_count} faces out of {len(mesh.faces)}")
    
    if missed_count > min_missed_faces and depth < max_depth:
        # Intelligent split along longest axis of the local mesh
        extents = mesh.extents
        axis = np.argmax(extents)
        normal_pos = [0, 0, 0]
        normal_neg = [0, 0, 0]
        normal_pos[axis] = 1
        normal_neg[axis] = -1
        
        print(f"{indent}Splitting along axis {axis}...")
        
        left_mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=normal_pos, plane_origin=[0,0,0])
        right_mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=normal_neg, plane_origin=[0,0,0])
        
        left_node = encode_recursive(left_mesh, depth+1, max_depth, min_missed_faces, resolution)
        right_node = encode_recursive(right_mesh, depth+1, max_depth, min_missed_faces, resolution)
        
        return {
            'is_leaf': False,
            'center': center,
            'scale': scale,
            'left': left_node,
            'right': right_node
        }
    else:
        # Leaf node
        return {
            'is_leaf': True,
            'center': center,
            'scale': scale,
            'maps': height_maps
        }

def decode_recursive(node, resolution=128):
    """
    Recursively decodes leaf nodes and transforms them up the spatial hierarchy.
    """
    if node is None:
        return None
        
    if node['is_leaf']:
        local_pcd = decode_point_cloud(node['maps'], resolution=resolution)
        if len(local_pcd.points) == 0:
            return None
        pts = np.asarray(local_pcd.points)
        norms = np.asarray(local_pcd.normals)
    else:
        left_res = decode_recursive(node['left'], resolution)
        right_res = decode_recursive(node['right'], resolution)
        
        pts_list = []
        norms_list = []
        if left_res is not None:
            pts_list.append(left_res[0])
            norms_list.append(left_res[1])
        if right_res is not None:
            pts_list.append(right_res[0])
            norms_list.append(right_res[1])
            
        if len(pts_list) == 0:
            return None
            
        pts = np.vstack(pts_list)
        norms = np.vstack(norms_list)
        
    # Apply hierarchical transform to bring into parent's coordinate space
    transformed_pts = (pts * node['scale']) + node['center']
    
    return transformed_pts, norms

def decode_and_fuse(root_node, resolution=128):
    print("Decoding Recursive Tree...")
    res = decode_recursive(root_node, resolution)
    if res is None:
        raise ValueError("Failed to decode any points.")
        
    pts, norms = res
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(norms)
    
    print(f"Total fused points from all recursive leaves: {len(pcd.points)}")
    return pcd
