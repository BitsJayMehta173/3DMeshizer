"""
Analysis script: compares the original mesh vertices to the hybrid reconstructed mesh
and reports how many original vertices are still not captured.
"""
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import KDTree
from src.utils import download_sample_mesh
from src.raycasting import normalize_mesh
import os

def analyze_missing(threshold=0.005):
    print("=" * 60)
    print("  VERTEX COVERAGE ANALYSIS")
    print("=" * 60)

    mesh_path = download_sample_mesh()
    orig_mesh = trimesh.load(mesh_path)
    orig_size_bytes = os.path.getsize(mesh_path)

    # Normalize (same transform used during encoding)
    orig_mesh_norm, center, scale = normalize_mesh(orig_mesh)
    original_vertices = np.asarray(orig_mesh_norm.vertices)
    total_verts = len(original_vertices)

    print(f"\nOriginal Mesh:")
    print(f"  Vertices  : {total_verts:,}")
    print(f"  Faces     : {len(orig_mesh_norm.faces):,}")
    print(f"  File size : {orig_size_bytes / 1024:.2f} KB")

    # Load reconstructed meshes and check vertex coverage
    methods = {
        "Baseline (6 maps)":         "data/decoded",  # no file for baseline, skip
        "Multi-Cube Spare Parts":    "data/decoded/multicube_reconstructed.ply",
        "Bi-Section Split":          "data/decoded/bisection_reconstructed.ply",
        "Multi-Layer Depth Peeling": "data/decoded/multilayer_reconstructed.ply",
        "Hybrid Residual (Best)":    "data/decoded/hybrid_reconstructed.ply",
    }

    compressed_sizes = {
        "Multi-Cube Spare Parts":    188.9 + 5.0,     # 6 PNGs + spare part NPZ (KB)
        "Bi-Section Split":          350.0,
        "Multi-Layer Depth Peeling": 501.44,
        "Hybrid Residual (Best)":    423.01,
    }

    print(f"\n{'Method':<30} {'Recon Verts':>12} {'Missing Verts':>14} {'Coverage %':>11} {'Compressed KB':>14} {'Saved %':>9}")
    print("-" * 97)

    for name, path in methods.items():
        if not os.path.exists(path):
            continue

        recon = o3d.io.read_triangle_mesh(path)
        recon_pts = np.asarray(recon.vertices)

        if len(recon_pts) == 0:
            continue

        # Normalize reconstructed mesh to same space for fair comparison
        c = recon.get_center()
        recon.translate(-c)
        s = max(np.max(recon.get_max_bound() - recon.get_min_bound()) / 2.0, 1e-6)
        recon.scale(1.0 / s, center=(0, 0, 0))
        recon_pts = np.asarray(recon.vertices)

        # KD-Tree search
        kdtree = KDTree(recon_pts)
        distances, _ = kdtree.query(original_vertices, k=1, workers=-1)

        missing_mask = distances > threshold
        missing_count = missing_mask.sum()
        coverage_pct = 100.0 * (1 - missing_count / total_verts)

        comp_kb = compressed_sizes.get(name, 0)
        saved_pct = 100.0 * (1 - (comp_kb * 1024) / orig_size_bytes) if comp_kb > 0 else 0

        print(f"{name:<30} {len(recon_pts):>12,} {missing_count:>14,} {coverage_pct:>10.2f}% {comp_kb:>13.1f} {saved_pct:>8.2f}%")

    print("\n" + "=" * 60)
    print("  HYBRID RESIDUAL DEEP ANALYSIS")
    print("=" * 60)

    if os.path.exists("data/decoded/hybrid_reconstructed.ply"):
        recon = o3d.io.read_triangle_mesh("data/decoded/hybrid_reconstructed.ply")
        c = recon.get_center()
        recon.translate(-c)
        s = max(np.max(recon.get_max_bound() - recon.get_min_bound()) / 2.0, 1e-6)
        recon.scale(1.0 / s, center=(0, 0, 0))
        recon_pts = np.asarray(recon.vertices)

        kdtree = KDTree(recon_pts)
        distances, _ = kdtree.query(original_vertices, k=1, workers=-1)

        for eps in [0.001, 0.005, 0.010, 0.020, 0.050]:
            missing = (distances > eps).sum()
            coverage = 100.0 * (1 - missing / total_verts)
            print(f"  Threshold e={eps:.3f} -> {missing:,} missing vertices ({coverage:.2f}% coverage)")

if __name__ == "__main__":
    analyze_missing()
