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

def get_teapot():
    os.makedirs("data/meshes", exist_ok=True)
    out_path = "data/meshes/teapot.obj"
    if not os.path.exists(out_path):
        print("Downloading Utah Teapot...")
        url = "https://raw.githubusercontent.com/jaz303/utah-teapot/master/teapot.obj"
        urllib.request.urlretrieve(url, out_path)
    return out_path

def test_on_teapot():
    mesh_path = get_teapot()
    orig_size = os.path.getsize(mesh_path)
    
    mesh = trimesh.load(mesh_path)
    # The Utah Teapot OBJ might have multiple sub-meshes or a weird format,
    # Let's ensure it's a single trimesh
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            print("Failed to load geometry from teapot.obj")
            return
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in mesh.geometry.values())
        )
    
    print(f"Loaded Teapot: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    mesh_norm, _, _ = normalize_mesh(mesh)
    
    print("Encoding Teapot with Ultimate Hybrid...")
    encoded = encode_ultimate(mesh_norm, resolution=256, num_layers=4, threshold=0.005)
    
    _, _, comp_size = calculate_ultimate_storage(encoded, out_dir="data/compressed/teapot_png")
    
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
    
    out_ply = "data/decoded/teapot_recon.ply"
    o3d.io.write_triangle_mesh(out_ply, recon)
    
    out_png = "results/plots/teapot_comparison.png"
    make_comparison(mesh_path, out_ply, out_png, method_name="Ultimate Hybrid (Teapot)")
    print(f"Saved comparison image to: {out_png}")

if __name__ == "__main__":
    test_on_teapot()
