import os
import open3d as o3d
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import time

from src.utils import download_sample_mesh
from src.decoder import reconstruct_mesh_poisson
from src.evaluation import evaluate_reconstruction
from src.recursive_bisection import encode_recursive, decode_and_fuse, count_leaf_nodes
from src.compression import compress_png

def extract_all_maps(node, prefix="", map_dict=None):
    """
    Recursively extracts all height maps from the tree into a flat dictionary
    so we can pass them to the compression function and measure their total size.
    """
    if map_dict is None:
        map_dict = {}
    if node is None:
        return map_dict
    if node['is_leaf']:
        for face, m in node['maps'].items():
            map_dict[f"{prefix}_{face}"] = m
    else:
        extract_all_maps(node['left'], prefix + "L", map_dict)
        extract_all_maps(node['right'], prefix + "R", map_dict)
    return map_dict

def render_mesh_comparison(orig_mesh_path, recon_mesh, save_path):
    def get_img(m):
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=800)
        
        if not m.has_vertex_normals():
            m.compute_vertex_normals()
            
        m_copy = o3d.geometry.TriangleMesh(m)
        normals = np.asarray(m_copy.vertex_normals)
        colors = (normals + 1.0) / 2.0
        m_copy.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        vis.add_geometry(m_copy)
        
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.8)
        
        for _ in range(5):
            vis.poll_events()
            vis.update_renderer()
            
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        return np.asarray(img)

    orig_m = o3d.io.read_triangle_mesh(orig_mesh_path)
    c = orig_m.get_center()
    orig_m.translate(-c)
    s = np.max(orig_m.get_max_bound() - orig_m.get_min_bound()) / 2.0
    orig_m.scale(1.0 / s, center=(0,0,0))
    
    img_orig = get_img(orig_m)
    img_recon = get_img(recon_mesh)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_orig)
    axes[0].set_title("Original Mesh")
    axes[0].axis('off')
    
    axes[1].imshow(img_recon)
    axes[1].set_title("Reconstructed (Recursive Bisection)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

def run_recursive(max_depth=3, resolution=128):
    print(f"--- Running Intelligent Recursive Bisection Compression (Max Depth: {max_depth}) ---")
    
    mesh_path = download_sample_mesh()
    mesh = trimesh.load(mesh_path)
    
    encoded_tree = encode_recursive(mesh, max_depth=max_depth, resolution=resolution)
    
    leaves = count_leaf_nodes(encoded_tree)
    total_maps = leaves * 6
    print(f"\nTree built with {leaves} leaf nodes (Total {total_maps} height maps).")
    
    # Measure compression size
    all_maps = extract_all_maps(encoded_tree, prefix="root")
    print("Compressing all maps to calculate total file size...")
    total_bytes, _ = compress_png(all_maps, "data/compressed/recursive_png")
    kb_size = total_bytes / 1024.0
    print(f"Total Compressed Size (PNG): {kb_size:.2f} KB for {total_maps} maps.")
    
    # Decode
    pcd = decode_and_fuse(encoded_tree, resolution=resolution)
    
    # Reconstruct
    recon_mesh = reconstruct_mesh_poisson(pcd, depth=9)
    
    # Evaluate
    metrics = evaluate_reconstruction(mesh, recon_mesh)
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
        
    os.makedirs("data/decoded", exist_ok=True)
    o3d.io.write_triangle_mesh("data/decoded/recursive_reconstructed.ply", recon_mesh)
    print("Saved reconstructed mesh to data/decoded/recursive_reconstructed.ply")
    
    os.makedirs("results/plots", exist_ok=True)
    render_mesh_comparison(mesh_path, recon_mesh, "results/plots/recursive_comparison.png")
    
    return recon_mesh, metrics, kb_size, total_maps

if __name__ == "__main__":
    # We use 128 resolution for recursive to balance memory, since max_depth=3 means up to 8x6=48 maps.
    run_recursive(max_depth=3, resolution=128)
