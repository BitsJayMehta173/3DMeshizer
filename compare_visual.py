import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from src.utils import download_sample_mesh
from src.raycasting import normalize_mesh
from src.encoder import encode_mesh
from src.decoder import decode_point_cloud, reconstruct_mesh_poisson
import time

def render_mesh(mesh, color):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)
    
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
        
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.paint_uniform_color(color)
    
    vis.add_geometry(mesh_copy)
    
    # Simple view setup
    ctr = vis.get_view_control()
    # Armadillo looks good from the front roughly
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    vis.poll_events()
    vis.update_renderer()
    
    # We need to give it a moment to render properly sometimes
    for _ in range(5):
        vis.poll_events()
        vis.update_renderer()
        
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return np.asarray(image)

if __name__ == "__main__":
    print("Loading Original...")
    mesh_path = download_sample_mesh()
    
    orig_mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    # Center and scale for O3D
    center = orig_mesh_o3d.get_center()
    orig_mesh_o3d.translate(-center)
    scale = np.max(orig_mesh_o3d.get_max_bound() - orig_mesh_o3d.get_min_bound()) / 2.0
    orig_mesh_o3d.scale(1.0 / scale, center=(0,0,0))
    
    print("Encoding...")
    mesh_t = trimesh.load(mesh_path)
    mesh_t, _, _ = normalize_mesh(mesh_t)
    height_maps = encode_mesh(mesh_t, resolution=256)
    
    print("Decoding...")
    pcd = decode_point_cloud(height_maps, resolution=256)
    recon_mesh = reconstruct_mesh_poisson(pcd, depth=9)
    
    print("Rendering Images...")
    img_orig = render_mesh(orig_mesh_o3d, [0.3, 0.7, 0.3])
    img_recon = render_mesh(recon_mesh, [0.8, 0.5, 0.2])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_orig)
    axes[0].set_title("Original Mesh")
    axes[0].axis('off')
    
    axes[1].imshow(img_recon)
    axes[1].set_title("Reconstructed from 6 Height Maps")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/plots/comparison.png")
    print("Saved comparison.png")
