import os
import open3d as o3d
import trimesh
import numpy as np

def download_sample_mesh(save_dir="data/meshes", filename="armadillo.ply"):
    """
    Downloads a sample character mesh (Armadillo) if it doesn't exist.
    Returns the absolute path to the mesh.
    """
    os.makedirs(save_dir, exist_ok=True)
    mesh_path = os.path.join(save_dir, filename)
    
    if not os.path.exists(mesh_path):
        print("Downloading sample character mesh (Armadillo)...")
        armadillo = o3d.data.ArmadilloMesh()
        mesh = o3d.io.read_triangle_mesh(armadillo.path)
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Saved to {mesh_path}")
    else:
        print(f"Mesh already exists at {mesh_path}")
        
    return mesh_path

def load_mesh(filepath):
    """
    Loads a mesh using trimesh.
    """
    return trimesh.load(filepath)
