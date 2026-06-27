import open3d as o3d
import trimesh
import os
from src.utils import download_sample_mesh

def get_dataset():
    """
    Returns a dictionary of mesh names to their file paths,
    generating or downloading them as needed.
    """
    os.makedirs("data/meshes", exist_ok=True)
    dataset = {}

    # 1. Sphere (Convex, simplest)
    sphere_path = "data/meshes/sphere.ply"
    if not os.path.exists(sphere_path):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
        o3d.io.write_triangle_mesh(sphere_path, mesh)
    dataset["Sphere"] = sphere_path

    # 2. Torus (Genus 1, hole)
    torus_path = "data/meshes/torus.ply"
    if not os.path.exists(torus_path):
        mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=1.0, tube_radius=0.4, radial_resolution=50, tubular_resolution=50)
        o3d.io.write_triangle_mesh(torus_path, mesh)
    dataset["Torus"] = torus_path

    # 3. Armadillo (Highly complex, concave)
    dataset["Armadillo"] = download_sample_mesh()

    return dataset

if __name__ == "__main__":
    ds = get_dataset()
    for name, path in ds.items():
        print(f"{name}: {path} ({os.path.getsize(path)/1024:.1f} KB)")
