import os
import open3d as o3d
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from src.utils import download_sample_mesh
from src.raycasting import normalize_mesh
from src.decoder import reconstruct_mesh_poisson
from src.evaluation import evaluate_reconstruction
from src.multi_cube_partitioning import encode_multi_cube, decode_multi_cube
import time

def render_mesh_comparison(orig_mesh_path, recon_mesh, save_path):
    def get_img(m, color):
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=800)
        
        if not m.has_vertex_normals():
            m.compute_vertex_normals()
            
        m_copy = o3d.geometry.TriangleMesh(m)
        m_copy.paint_uniform_color(color)
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
    
    img_orig = get_img(orig_m, [0.3, 0.7, 0.3])
    img_recon = get_img(recon_mesh, [0.2, 0.5, 0.8])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_orig)
    axes[0].set_title("Original Mesh")
    axes[0].axis('off')
    
    axes[1].imshow(img_recon)
    axes[1].set_title("Reconstructed (Multi-Cube Spare Parts)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")


def run_multicube(base_res=256, local_res=64):
    print(f"--- Running Multi-Cube Compression ---")
    
    mesh_path = download_sample_mesh()
    mesh = trimesh.load(mesh_path)
    
    # Normalize globally
    mesh, center, scale = normalize_mesh(mesh)
    
    # Encode
    encoded_data = encode_multi_cube(mesh, base_res=base_res, local_res=local_res, min_faces=100)
    
    # Decode
    pcd = decode_multi_cube(encoded_data, base_res=base_res, local_res=local_res)
    
    # Reconstruct
    recon_mesh = reconstruct_mesh_poisson(pcd, depth=9)
    
    # Evaluate
    metrics = evaluate_reconstruction(mesh, recon_mesh)
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
        
    os.makedirs("results/plots", exist_ok=True)
    render_mesh_comparison(mesh_path, recon_mesh, "results/plots/multicube_comparison.png")
    
    # Save the mesh for interactive viewing
    os.makedirs("data/decoded", exist_ok=True)
    o3d.io.write_triangle_mesh("data/decoded/multicube_reconstructed.ply", recon_mesh)
    print("Saved reconstructed mesh to data/decoded/multicube_reconstructed.ply")
    
    return recon_mesh, metrics

if __name__ == "__main__":
    run_multicube()
