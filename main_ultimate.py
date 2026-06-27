import os
import numpy as np
import open3d as o3d
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from src.utils import download_sample_mesh
from src.raycasting import normalize_mesh
from src.decoder import reconstruct_mesh_poisson
from src.evaluation import evaluate_reconstruction
from src.ultimate_hybrid import encode_ultimate, decode_ultimate, calculate_ultimate_storage

# Import the software renderer from our new render_comparison module
sys.path.insert(0, os.path.dirname(__file__))
from render_comparison import make_comparison


def run_ultimate(resolution=256, num_layers=4, threshold=0.005):
    mesh_path = download_sample_mesh()
    orig_size_bytes = os.path.getsize(mesh_path)

    mesh = trimesh.load(mesh_path)
    mesh, center, scale = normalize_mesh(mesh)

    # Encode
    encoded_data = encode_ultimate(mesh, resolution=resolution,
                                   num_layers=num_layers, threshold=threshold)

    # Storage breakdown
    print("\nCalculating compressed sizes...")
    png_bytes, residual_bytes, total_bytes = calculate_ultimate_storage(encoded_data)
    total_kb  = total_bytes / 1024.0
    png_kb    = png_bytes / 1024.0
    res_kb    = residual_bytes / 1024.0
    orig_kb   = orig_size_bytes / 1024.0
    ratio     = orig_size_bytes / total_bytes
    saved_pct = 100.0 * (1 - total_bytes / orig_size_bytes)

    print(f"\n{'='*60}")
    print(f"  ULTIMATE HYBRID — COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Original mesh size        : {orig_kb:>10.2f} KB")
    print(f"  Multi-layer PNGs (24 maps): {png_kb:>10.2f} KB")
    print(f"  Residual safety net (NPZ) : {res_kb:>10.2f} KB")
    print(f"  Total compressed          : {total_kb:>10.2f} KB")
    print(f"  Compression ratio         : {ratio:>10.1f}x")
    print(f"  Space saved               : {saved_pct:>10.2f}%")

    # Decode
    pcd = decode_ultimate(encoded_data)

    # Reconstruct
    recon_mesh = reconstruct_mesh_poisson(pcd, depth=9)

    # Evaluate
    metrics = evaluate_reconstruction(mesh, recon_mesh)
    print(f"\n{'='*60}")
    print(f"  ULTIMATE HYBRID — QUALITY METRICS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.6f}")

    # Save
    os.makedirs("data/decoded", exist_ok=True)
    out_ply = "data/decoded/ultimate_reconstructed.ply"
    o3d.io.write_triangle_mesh(out_ply, recon_mesh)
    print(f"\nSaved: {out_ply}")

    # Render 4-angle comparison with the software renderer
    os.makedirs("results/plots", exist_ok=True)
    out_img = "results/plots/ultimate_comparison.png"
    make_comparison(mesh_path, out_ply, out_img, method_name="Ultimate Hybrid")
    print(f"Saved: {out_img}")

    print(f"\n{'='*60}")
    print(f"  Space saved: {saved_pct:.2f}% | Normal consistency: {metrics['normal_consistency']*100:.1f}%")
    print(f"  Missing verts (threshold {threshold}): {encoded_data['missing_vertex_count']:,}")
    print(f"{'='*60}\n")

    return recon_mesh, metrics, saved_pct


if __name__ == "__main__":
    run_ultimate()
