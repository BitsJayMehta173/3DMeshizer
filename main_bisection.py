import os
import open3d as o3d
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from src.utils import download_sample_mesh
from src.raycasting import normalize_mesh
from src.decoder import reconstruct_mesh_poisson
from src.evaluation import evaluate_reconstruction
from src.bisection_partitioning import encode_bisection, decode_bisection

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
    axes[1].set_title("Reconstructed (Bi-Section Split)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

def run_bisection(resolution=256):
    print(f"--- Running Bi-Section Mesh Splitting Compression ---")
    
    mesh_path = download_sample_mesh()
    mesh = trimesh.load(mesh_path)
    
    # Normalize globally first
    mesh, center, scale = normalize_mesh(mesh)
    
    # Encode
    encoded_data = encode_bisection(mesh, resolution=resolution)
    
    # Decode
    pcd = decode_bisection(encoded_data, resolution=resolution)
    
    # Reconstruct
    recon_mesh = reconstruct_mesh_poisson(pcd, depth=9)
    
    # Evaluate
    metrics = evaluate_reconstruction(mesh, recon_mesh)
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
        
    os.makedirs("results/plots", exist_ok=True)
    render_mesh_comparison(mesh_path, recon_mesh, "results/plots/bisection_comparison.png")
    
    # Save the mesh for interactive viewing
    os.makedirs("data/decoded", exist_ok=True)
    o3d.io.write_triangle_mesh("data/decoded/bisection_reconstructed.ply", recon_mesh)
    print("Saved reconstructed mesh to data/decoded/bisection_reconstructed.ply")
    
    return recon_mesh, metrics

if __name__ == "__main__":
    run_bisection()
