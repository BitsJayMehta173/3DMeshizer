import os
import trimesh
import numpy as np

from src.dataset import get_dataset
from src.raycasting import normalize_mesh
from src.ultimate_hybrid import encode_ultimate, decode_ultimate, calculate_ultimate_storage
from src.decoder import reconstruct_mesh_poisson
from src.evaluation import evaluate_reconstruction

def evaluate_all():
    dataset = get_dataset()
    results = []

    print("\nStarting Bulk Dataset Evaluation for MTech Thesis...\n")

    for name, path in dataset.items():
        print(f"\n======================================")
        print(f"  Evaluating: {name}")
        print(f"======================================")

        orig_size = os.path.getsize(path)
        mesh = trimesh.load(path)
        
        # We need to make sure the mesh has enough vertices to test properly, 
        # but sphere/torus might be low poly. Let's just run them as-is.
        v_count = len(mesh.vertices)
        
        mesh_norm, _, _ = normalize_mesh(mesh)

        # Encode
        enc = encode_ultimate(mesh_norm, resolution=256, num_layers=4, threshold=0.005)
        
        # Sizes
        _, _, comp_size = calculate_ultimate_storage(enc)
        ratio = orig_size / comp_size
        saved_pct = 100.0 * (1 - comp_size / orig_size)
        
        # Decode & Reconstruct
        pcd = decode_ultimate(enc)
        recon = reconstruct_mesh_poisson(pcd, depth=9)
        
        # Evaluate
        try:
            metrics = evaluate_reconstruction(mesh_norm, recon)
            chamfer_val = metrics['chamfer']
            normal_val = metrics['normal_consistency'] * 100
        except Exception as e:
            print(f"Evaluation error: {e}")
            chamfer_val = 0.0
            normal_val = 0.0

        results.append({
            'Name': name,
            'Vertices': v_count,
            'Orig_KB': orig_size / 1024.0,
            'Comp_KB': comp_size / 1024.0,
            'Ratio': ratio,
            'Saved_%': saved_pct,
            'Chamfer': chamfer_val,
            'Normal_Consist': normal_val
        })

    # Print markdown table
    print("\n\n# Dataset Evaluation Results\n")
    print("| Mesh | Vertices | Orig Size | Comp Size | Saved % | Ratio | Chamfer | Normal Consist |")
    print("|:---|---:|---:|---:|---:|---:|---:|---:|")
    
    for r in results:
        print(f"| {r['Name']} | {r['Vertices']:,} | {r['Orig_KB']:.1f} KB | {r['Comp_KB']:.1f} KB | {r['Saved_%']:.1f}% | {r['Ratio']:.1f}x | {r['Chamfer']:.5f} | {r['Normal_Consist']:.1f}% |")

if __name__ == "__main__":
    evaluate_all()
