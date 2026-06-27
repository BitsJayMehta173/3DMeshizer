import os
import json
import numpy as np
from src.utils import download_sample_mesh, load_mesh
from src.raycasting import normalize_mesh
from src.encoder import encode_mesh, save_height_maps_npy
from src.compression import evaluate_compressions
from src.decoder import decode_point_cloud, reconstruct_mesh_poisson
from src.evaluation import evaluate_reconstruction
from src.visualization import plot_height_maps, render_mesh_o3d

def run_baseline(resolution=256):
    print(f"--- Running Baseline Cube Height Map Compression (Res: {resolution}) ---")
    
    # 1. Download & Load
    mesh_path = download_sample_mesh()
    mesh = load_mesh(mesh_path)
    print(f"Original Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")
    
    # 2. Normalize
    mesh, center, scale = normalize_mesh(mesh)
    
    # 3. Encode
    height_maps = encode_mesh(mesh, resolution=resolution)
    
    # Save raw for reference
    save_height_maps_npy(height_maps, "data/heightmaps/raw")
    
    # Ensure plots dir exists
    os.makedirs("results/plots", exist_ok=True)
    plot_height_maps(height_maps, save_path="results/plots/height_maps.png")
    
    # 4. Compress
    print("\nEvaluating Compressions...")
    comp_results = evaluate_compressions(height_maps, "data/compressed")
    print(json.dumps(comp_results, indent=2))
    
    # 5. Decode
    print("\nDecoding...")
    pcd = decode_point_cloud(height_maps, resolution=resolution)
    print(f"Decoded Point Cloud has {len(pcd.points)} points.")
    
    # 6. Reconstruct
    recon_mesh = reconstruct_mesh_poisson(pcd, depth=9)
    print(f"Reconstructed Mesh: {len(recon_mesh.vertices)} vertices, {len(recon_mesh.triangles)} triangles.")
    
    # 7. Evaluate
    metrics = evaluate_reconstruction(mesh, recon_mesh)
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
        
    # Render
    render_mesh_o3d(recon_mesh, "results/plots/reconstructed.png")
    
    return metrics, comp_results

if __name__ == "__main__":
    run_baseline(resolution=256)
