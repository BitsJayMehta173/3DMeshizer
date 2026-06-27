"""
Ultimate Hybrid: Multi-Layer Depth Peeling (4 layers) + Residual Safety Net.
Step 1: Encode 4 depth layers (24 maps) for maximum base coverage.
Step 2: Internally decode them and KD-Tree diff against original vertices.
Step 3: Append only the still-missing vertices as a small residual NPZ.
Step 4: Re-estimate normals on the entire fused cloud and run Poisson.
"""
import numpy as np
import trimesh
import open3d as o3d
import time
import os
import tempfile
from scipy.spatial import KDTree

from src.raycasting import FACES, get_camera_rays
from src.multilayer_depth import encode_multilayer, decode_multilayer
from src.decoder import reconstruct_mesh_poisson
from src.compression import compress_png


def encode_ultimate(mesh, resolution=256, num_layers=4, threshold=0.005):
    print("=" * 60)
    print(f"  ULTIMATE HYBRID ENCODING")
    print(f"  Layers: {num_layers} | Resolution: {resolution} | Threshold: {threshold}")
    print("=" * 60)
    start = time.time()

    # --- Step 1: Multi-Layer Depth Peeling ---
    print("\n[1/4] Running multi-layer depth peeling...")
    encoded_ml = encode_multilayer(mesh, resolution=resolution, num_layers=num_layers)

    # --- Step 2: Internally decode to see what we captured ---
    print("[2/4] Internally decoding to evaluate coverage...")
    decoded_pcd = decode_multilayer(encoded_ml, resolution=resolution)
    decoded_pts = np.asarray(decoded_pcd.points)

    # --- Step 3: KD-Tree diff against original vertices ---
    original_vertices = np.asarray(mesh.vertices)
    total_verts = len(original_vertices)
    print(f"[3/4] KD-Tree search on {total_verts:,} vertices vs {len(decoded_pts):,} decoded points...")

    kdtree = KDTree(decoded_pts)
    distances, _ = kdtree.query(original_vertices, k=1, workers=-1)

    missing_mask = distances > threshold
    residual_vertices = original_vertices[missing_mask]
    missing_count = len(residual_vertices)
    print(f"      After {num_layers} depth layers: {missing_count:,} / {total_verts:,} vertices still missing "
          f"({100.0 * missing_count / total_verts:.2f}%)")

    # --- Step 4: Pack everything ---
    encoded_data = {
        'multilayer_maps': encoded_ml['maps'],
        'residual_vertices': residual_vertices,
        'resolution': resolution,
        'num_layers': num_layers,
        'threshold': threshold,
        'original_vertex_count': total_verts,
        'missing_vertex_count': missing_count,
    }

    print(f"[4/4] Ultimate encoding done in {time.time() - start:.2f}s")
    return encoded_data


def decode_ultimate(encoded_data):
    print("\nDecoding Ultimate Hybrid...")
    resolution = encoded_data['resolution']

    # Decode multi-layer base
    ml_pcd = decode_multilayer({'maps': encoded_data['multilayer_maps']}, resolution=resolution)
    base_pts = np.asarray(ml_pcd.points)

    # Append residuals
    residual_pts = encoded_data['residual_vertices']
    print(f"  Base (multi-layer): {len(base_pts):,} points")
    print(f"  Residual safety net: {len(residual_pts):,} points")

    if len(residual_pts) > 0:
        all_pts = np.vstack([base_pts, residual_pts])
    else:
        all_pts = base_pts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)

    # Re-estimate normals consistently on the full fused cloud
    print(f"  Re-estimating normals on {len(all_pts):,} total points...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    print(f"  Total fused points: {len(pcd.points):,}")
    return pcd


def calculate_ultimate_storage(encoded_data, out_dir="data/compressed/ultimate_png"):
    # PNG compression for all 24 depth maps
    png_bytes, _ = compress_png(encoded_data['multilayer_maps'], out_dir)

    # Compressed residual vertices
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        tmp_path = f.name
    np.savez_compressed(tmp_path, vertices=encoded_data['residual_vertices'])
    npz_path = tmp_path + '.npz' if os.path.exists(tmp_path + '.npz') else tmp_path
    residual_bytes = os.path.getsize(npz_path)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return png_bytes, residual_bytes, png_bytes + residual_bytes
