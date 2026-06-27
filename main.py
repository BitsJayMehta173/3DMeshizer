import os
import sys
import numpy as np
import open3d as o3d
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import download_sample_mesh
from src.raycasting import normalize_mesh
from src.decoder import reconstruct_mesh_poisson
from src.evaluation import evaluate_reconstruction
from src.ultimate_hybrid import encode_ultimate, decode_ultimate, calculate_ultimate_storage
from render_comparison import make_comparison

def main(resolution=256, num_layers=4, threshold=0.005):
    """
    Main orchestrator for the Ultimate Hybrid 3D Mesh Compression pipeline.
    """
    print("\nStarting Ultimate Hybrid 3D Mesh Compression Pipeline...\n")
    
    # 1. Download/load the test mesh (Armadillo)
    mesh_path = download_sample_mesh()
    orig_size_bytes = os.path.getsize(mesh_path)
    
    mesh = trimesh.load(mesh_path)
    mesh, center, scale = normalize_mesh(mesh)

    # 2. Encode the mesh using Ultimate Hybrid Architecture
    # (Multi-Layer Depth Peeling + KD-Tree Residual Safety Net)
    encoded_data = encode_ultimate(mesh, resolution=resolution, 
                                   num_layers=num_layers, threshold=threshold)

    # 3. Calculate compressed sizes
    print("\nCalculating compressed file sizes...")
    png_bytes, residual_bytes, total_bytes = calculate_ultimate_storage(encoded_data)
    total_kb  = total_bytes / 1024.0
    orig_kb   = orig_size_bytes / 1024.0
    ratio     = orig_size_bytes / total_bytes
    saved_pct = 100.0 * (1 - total_bytes / orig_size_bytes)

    print(f"\n{'='*60}")
    print(f"  ULTIMATE HYBRID — COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Original mesh size        : {orig_kb:>10.2f} KB")
    print(f"  Total compressed          : {total_kb:>10.2f} KB")
    print(f"  Compression ratio         : {ratio:>10.1f}x")
    print(f"  Space saved               : {saved_pct:>10.2f}%")

    # 4. Decode the data back into a fused point cloud
    pcd = decode_ultimate(encoded_data)

    # 5. Reconstruct the surface via Poisson
    recon_mesh = reconstruct_mesh_poisson(pcd, depth=9)

    # 6. Evaluate metrics (Chamfer distance, normal consistency)
    metrics = evaluate_reconstruction(mesh, recon_mesh)
    print(f"\n{'='*60}")
    print(f"  ULTIMATE HYBRID — QUALITY METRICS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.6f}")

    # 7. Save outputs
    os.makedirs("data/decoded", exist_ok=True)
    out_ply = "data/decoded/ultimate_reconstructed.ply"
    o3d.io.write_triangle_mesh(out_ply, recon_mesh)
    
    os.makedirs("results/plots", exist_ok=True)
    out_img = "results/plots/ultimate_comparison.png"
    make_comparison(mesh_path, out_ply, out_img, method_name="Ultimate Hybrid")
    
    print(f"\nAll tasks complete! View the mesh interactively with: python visualize_interactive.py")

if __name__ == "__main__":
    main()
