import open3d as o3d
import trimesh
import numpy as np
import os
from src.utils import download_sample_mesh
from src.raycasting import normalize_mesh
from src.ultimate_hybrid import encode_ultimate, decode_ultimate

def visualize_raw_points():
    print("Encoding and decoding mesh to get the RAW Point Cloud...")
    
    # 1. Load Original
    mesh_path = download_sample_mesh()
    mesh = trimesh.load(mesh_path)
    mesh, _, _ = normalize_mesh(mesh)
    
    # 2. Encode & Decode to get the Point Cloud (Before Poisson)
    # We do a fast encode here just to get the cloud
    encoded = encode_ultimate(mesh, resolution=256, num_layers=4, threshold=0.005)
    pcd = decode_ultimate(encoded)
    
    print(f"\nOriginal Mesh Vertices: {len(mesh.vertices):,}")
    print(f"Captured Point Cloud:   {len(np.asarray(pcd.points)):,}")
    print("\nVisualizing RAW Point Cloud...")
    print("You will see that NO parts of the geometry are missing here.")

    # 3. Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Raw Point Cloud (Before Poisson)", width=1024, height=768)
    
    # Color the point cloud so it's easier to see
    pcd.paint_uniform_color([0.2, 0.7, 0.2]) # Green
    
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_raw_points()
