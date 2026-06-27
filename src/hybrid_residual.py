import numpy as np
import trimesh
import open3d as o3d
import time
from scipy.spatial import KDTree
from src.raycasting import FACES, get_camera_rays
from src.decoder import decode_point_cloud

def encode_hybrid(mesh, resolution=256, threshold=0.005):
    """
    Hybrid Residual Compression:
    1. Encode the mesh into 6 height maps (fast global pass).
    2. Immediately simulate decoding to get the reconstructed point cloud.
    3. Find all original vertices that are "missed" (too far from any decoded point).
    4. Store the height maps + residual missing vertices.
    """
    print(f"Encoding Hybrid Residual (Res: {resolution}, Threshold: {threshold})")
    start = time.time()

    # --- Step 1: Global Height Map Encoding ---
    height_maps = {}
    for face in FACES:
        o, d = get_camera_rays(face, resolution)
        loc, ir, it = mesh.ray.intersects_location(
            ray_origins=o, ray_directions=d, multiple_hits=False
        )
        dist = np.full(len(o), np.inf)
        if len(loc) > 0:
            dist[ir] = np.linalg.norm(loc - o[ir], axis=1)
        height_maps[face] = dist.reshape((resolution, resolution))

    # --- Step 2: Internal Decode Simulation ---
    print("  Internally simulating decode to find missed vertices...")
    decoded_pcd = decode_point_cloud(height_maps, resolution=resolution)
    decoded_pts = np.asarray(decoded_pcd.points)

    # --- Step 3: KD-Tree Nearest Neighbor Search ---
    original_vertices = np.asarray(mesh.vertices)
    print(f"  Running KD-Tree search on {len(original_vertices)} original vertices...")

    kdtree = KDTree(decoded_pts)
    distances, _ = kdtree.query(original_vertices, k=1, workers=-1)

    # --- Step 4: Extract Residual (Missing) Vertices ---
    missing_mask = distances > threshold
    missing_vertices = original_vertices[missing_mask]
    missing_count = len(missing_vertices)
    total_vertices = len(original_vertices)

    print(f"  Found {missing_count} / {total_vertices} missed vertices ({100.0*missing_count/total_vertices:.2f}%).")

    encoded_data = {
        'height_maps': height_maps,
        'residual_vertices': missing_vertices,
        'threshold': threshold,
        'resolution': resolution,
        'original_vertex_count': total_vertices,
        'missing_vertex_count': missing_count,
    }

    print(f"Hybrid Encoding took {time.time() - start:.2f}s")
    return encoded_data


def decode_hybrid(encoded_data):
    """
    Decodes height maps and merges with residual vertices for a flawless point cloud,
    re-estimating normals consistently.
    """
    print("Decoding Hybrid Residual Data...")
    resolution = encoded_data['resolution']

    # Decode the base height maps
    base_pcd = decode_point_cloud(encoded_data['height_maps'], resolution=resolution)
    base_pts = np.asarray(base_pcd.points)

    # Build the residual contribution
    residual_pts = encoded_data['residual_vertices']
    print(f"  Adding {len(residual_pts)} residual vertices to the point cloud...")

    all_pts = np.vstack([base_pts, residual_pts]) if len(residual_pts) > 0 else base_pts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)

    # Re-estimate normals consistently
    print("  Re-estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    print(f"  Total fused points (base + residual): {len(pcd.points)}")
    return pcd


def calculate_hybrid_storage(encoded_data, png_dir="data/compressed/png"):
    """
    Calculates the total storage size by summing PNG sizes and a compressed residual array.
    """
    import os, tempfile
    from src.compression import compress_png

    png_bytes, _ = compress_png(encoded_data['height_maps'], png_dir)

    # Compress the residual vertices as a NumPy array
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        tmp_path = f.name
    np.savez_compressed(tmp_path, vertices=encoded_data['residual_vertices'])
    residual_bytes = os.path.getsize(tmp_path + '.npz') if os.path.exists(tmp_path + '.npz') else os.path.getsize(tmp_path)
    os.unlink(tmp_path) if os.path.exists(tmp_path) else None

    total_bytes = png_bytes + residual_bytes
    return png_bytes, residual_bytes, total_bytes
