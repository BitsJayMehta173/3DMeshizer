import open3d as o3d
import os
import numpy as np

def run_interactive_viewer(mesh_path):
    if not os.path.exists(mesh_path):
        print(f"Error: Could not find {mesh_path}")
        print("Please run `python main_multicube.py` first to generate the mesh.")
        return
        
    print(f"Loading {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
        
    # Color the mesh using its normals to make all curves, crevices, and depths instantly pop!
    # Normals range from -1 to 1. We shift them to 0 to 1 for vibrant RGB coloring.
    normals = np.asarray(mesh.vertex_normals)
    colors = (normals + 1.0) / 2.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    print("\n--- INTERACTIVE VIEWER CONTROLS ---")
    print("Left Click + Drag  : Rotate")
    print("Right Click + Drag : Pan")
    print("Scroll Wheel       : Zoom")
    print("\n--- HOTKEYS INSIDE THE WINDOW ---")
    print("Press 'W' to toggle Wireframe Mode (shows the polygons)")
    print("Press 'S' to toggle Solid Surface Mode")
    print("Press 'Q' or 'Esc' to close the window.")
    
    # Draw geometries opens a native interactive window
    o3d.visualization.draw_geometries([mesh], 
                                      window_name="Reconstructed Mesh Visualizer",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_wireframe=True,
                                      mesh_show_back_face=True)

if __name__ == "__main__":
    # Loads the BEST reconstruction: Ultimate Hybrid (Multi-Layer + Residual Safety Net)
    run_interactive_viewer("data/decoded/ultimate_reconstructed.ply")
