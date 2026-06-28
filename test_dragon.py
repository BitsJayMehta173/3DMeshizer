import os
import urllib.request
import trimesh
import numpy as np
import open3d as o3d

from src.raycasting import normalize_mesh
from src.ultimate_hybrid import encode_ultimate, decode_ultimate, calculate_ultimate_storage
from src.decoder import reconstruct_mesh_poisson
from render_comparison import make_comparison
from src.evaluation import evaluate_reconstruction

def get_dragon():
    os.makedirs("data/meshes", exist_ok=True)
    out_path = "data/meshes/dragon.obj"
    if not os.path.exists(out_path):
        print("Downloading Stanford Dragon...")
        url = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/xyzrgb_dragon.obj"
        urllib.request.urlretrieve(url, out_path)
    return out_path

def test_on_dragon():
    mesh_path = get_dragon()
    orig_size = os.path.getsize(mesh_path)
    
    mesh = trimesh.load(mesh_path)
    # Ensure it's a single trimesh
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            print("Failed to load geometry from dragon.obj")
            return
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in mesh.geometry.values())
        )
    
    print(f"Loaded Dragon: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    mesh_norm, _, _ = normalize_mesh(mesh)
    
    print("Encoding Dragon with Ultimate Hybrid...")
    encoded = encode_ultimate(mesh_norm, resolution=256, num_layers=4, threshold=0.005)
    
    _, _, comp_size = calculate_ultimate_storage(encoded, out_dir="data/compressed/dragon_png")
    
    ratio = orig_size / comp_size
    saved_pct = 100.0 * (1 - comp_size / orig_size)
    
    print(f"Original size: {orig_size / 1024:.1f} KB")
    print(f"Compressed size: {comp_size / 1024:.1f} KB")
    print(f"Space Saved: {saved_pct:.1f}% ({ratio:.1f}x)")
    
    print("Decoding and Reconstructing...")
    pcd = decode_ultimate(encoded)
    recon = reconstruct_mesh_poisson(pcd, depth=9)
    
    metrics = evaluate_reconstruction(mesh_norm, recon)
    print("Evaluation Metrics:")
    print(f"Chamfer: {metrics['chamfer']:.5f}")
    print(f"Normal Consist: {metrics['normal_consistency']*100:.1f}%")
    
    out_ply = "data/decoded/dragon_recon.ply"
    o3d.io.write_triangle_mesh(out_ply, recon)
    
    out_png = "results/plots/dragon_comparison.png"
    make_comparison(mesh_path, out_ply, out_png, method_name="Ultimate Hybrid (Dragon)")
    print(f"Saved comparison image to: {out_png}")

if __name__ == "__main__":
    test_on_dragon()
