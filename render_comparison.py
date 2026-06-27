"""
Standalone comparison renderer using pure matplotlib (no OpenGL/GPU required).
Renders original vs reconstructed meshes side-by-side from multiple angles.
"""
import numpy as np
import trimesh
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os

def mesh_to_trimesh(o3d_mesh):
    """Convert open3d mesh to trimesh."""
    verts = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def render_trimesh_matplotlib(ax, mesh, azim=30, elev=20, title="", color='steelblue'):
    """Render a trimesh onto a matplotlib 3D axis using Poly3DCollection."""
    verts = mesh.vertices
    faces = mesh.faces

    # Sample a subset of faces for performance
    if len(faces) > 15000:
        idx = np.random.choice(len(faces), 15000, replace=False)
        faces = faces[idx]

    triangles = verts[faces]

    # Compute face normals for shading
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len = np.where(norm_len == 0, 1, norm_len)
    normals = normals / norm_len

    # Diffuse lighting from a fixed direction
    light = np.array([0.5, 0.5, 1.0])
    light /= np.linalg.norm(light)
    intensity = np.clip(np.dot(normals, light), 0.1, 1.0)

    base_color = np.array(matplotlib.colors.to_rgb(color))
    face_colors = base_color[np.newaxis, :] * intensity[:, np.newaxis]
    face_colors = np.clip(face_colors, 0, 1)

    poly = Poly3DCollection(triangles, linewidths=0, antialiased=False)
    poly.set_facecolor(face_colors)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    # Set axis limits
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)

def make_comparison(orig_mesh_path, recon_ply_path, save_path, method_name="Reconstructed"):
    # Load meshes
    orig_o3d = o3d.io.read_triangle_mesh(orig_mesh_path)
    recon_o3d = o3d.io.read_triangle_mesh(recon_ply_path)

    # Normalize original to [-1,1] bounding box
    c = orig_o3d.get_center()
    orig_o3d.translate(-c)
    s = np.max(orig_o3d.get_max_bound() - orig_o3d.get_min_bound()) / 2.0
    orig_o3d.scale(1.0 / s, center=(0, 0, 0))

    # Normalize reconstructed
    c2 = recon_o3d.get_center()
    recon_o3d.translate(-c2)
    s2 = np.max(recon_o3d.get_max_bound() - recon_o3d.get_min_bound()) / 2.0
    if s2 > 0:
        recon_o3d.scale(1.0 / s2, center=(0, 0, 0))

    orig_tm = mesh_to_trimesh(orig_o3d)
    recon_tm = mesh_to_trimesh(recon_o3d)

    angles = [
        (20,  30,  "Front"),
        (20,  120, "Side"),
        (70,  30,  "Top"),
        (20,  210, "Back"),
    ]

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')

    for i, (elev, azim, label) in enumerate(angles):
        # Original
        ax1 = fig.add_subplot(2, 4, i + 1, projection='3d')
        ax1.set_facecolor('#16213e')
        render_trimesh_matplotlib(ax1, orig_tm, azim=azim, elev=elev,
                                  title=f"Original — {label}", color='#4fc3f7')

        # Reconstructed
        ax2 = fig.add_subplot(2, 4, i + 5, projection='3d')
        ax2.set_facecolor('#16213e')
        render_trimesh_matplotlib(ax2, recon_tm, azim=azim, elev=elev,
                                  title=f"{method_name} — {label}", color='#81c784')

    plt.suptitle(f"Mesh Comparison: Original vs {method_name}",
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    mesh_path = "data/meshes/armadillo.ply"
    method = sys.argv[1] if len(sys.argv) > 1 else "hybrid"

    configs = {
        "hybrid":     ("data/decoded/hybrid_reconstructed.ply",
                       "results/plots/hybrid_comparison.png",       "Hybrid Residual"),
        "multilayer": ("data/decoded/multilayer_reconstructed.ply",
                       "results/plots/multilayer_comparison.png",   "Multi-Layer Depth Peeling"),
        "bisection":  ("data/decoded/bisection_reconstructed.ply",
                       "results/plots/bisection_comparison.png",    "Bi-Section Split"),
        "multicube":  ("data/decoded/multicube_reconstructed.ply",
                       "results/plots/multicube_comparison.png",    "Multi-Cube Spare Parts"),
    }

    if method not in configs:
        print(f"Unknown method '{method}'. Choose from: {list(configs.keys())}")
        sys.exit(1)

    recon_path, out_path, label = configs[method]
    os.makedirs("results/plots", exist_ok=True)
    make_comparison(mesh_path, recon_path, out_path, method_name=label)
