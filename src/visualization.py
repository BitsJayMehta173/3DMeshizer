import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d
from src.raycasting import FACES

def plot_height_maps(height_maps, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, face in enumerate(FACES):
        hmap = height_maps[face]
        # Mask out infinity for visualization
        vis_map = np.copy(hmap)
        valid_mask = ~np.isinf(vis_map)
        
        if np.any(valid_mask):
            vmin = np.min(vis_map[valid_mask])
            vmax = np.max(vis_map[valid_mask])
            vis_map[~valid_mask] = np.nan # Use nan so cmap can ignore it
        else:
            vmin, vmax = 0, 1
            
        im = axes[i].imshow(vis_map, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(face.capitalize())
        axes[i].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved height maps plot to {save_path}")
        plt.close()
    else:
        plt.show()

def render_mesh_o3d(mesh, save_path):
    """
    Renders an Open3D mesh to an image file.
    Requires an offscreen renderer or simply a quick capture if a display is available.
    We will use matplotlib to just save a basic representation or skip full offscreen 
    rendering if headless, but let's try a simple approach with Open3D's Visualizer.
    """
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        
        # Add mesh, add some color for better lighting
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture image
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        img = (np.asarray(image) * 255).astype(np.uint8)
        plt.imsave(save_path, img)
        print(f"Saved mesh render to {save_path}")
    except Exception as e:
        print(f"Could not render mesh (likely headless environment): {e}")

