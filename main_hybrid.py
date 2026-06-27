import os
import open3d as o3d
import trimesh
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import download_sample_mesh
from src.raycasting import normalize_mesh
from src.decoder import reconstruct_mesh_poisson
from src.evaluation import evaluate_reconstruction
from src.hybrid_residual import encode_hybrid, decode_hybrid, calculate_hybrid_storage


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
    orig_m.scale(1.0 / s, center=(0, 0, 0))

    img_orig = get_img(orig_m)
    img_recon = get_img(recon_mesh)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_orig)
    axes[0].set_title("Original Mesh")
    axes[0].axis('off')
    axes[1].imshow(img_recon)
    axes[1].set_title("Reconstructed (Hybrid Residual)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")


def run_hybrid(resolution=256, threshold=0.005):
    print(f"\n{'='*60}")
    print(f"  HYBRID RESIDUAL COMPRESSION PIPELINE")
    print(f"  Resolution: {resolution} | Threshold: {threshold}")
    print(f"{'='*60}\n")

    mesh_path = download_sample_mesh()
    mesh = trimesh.load(mesh_path)
    original_size_bytes = os.path.getsize(mesh_path)
    original_size_kb = original_size_bytes / 1024.0

    print(f"Original mesh: {len(mesh.vertices):,} vertices | {original_size_kb:.2f} KB\n")

    # Normalize mesh
    mesh, center, scale = normalize_mesh(mesh)

    # Encode
    encoded_data = encode_hybrid(mesh, resolution=resolution, threshold=threshold)

    # Compute storage breakdown
    print("\nCalculating compressed file sizes...")
    png_bytes, residual_bytes, total_bytes = calculate_hybrid_storage(
        encoded_data, png_dir="data/compressed/hybrid_png"
    )
    total_kb = total_bytes / 1024.0
    png_kb = png_bytes / 1024.0
    residual_kb = residual_bytes / 1024.0

    compression_ratio = original_size_bytes / total_bytes
    compression_pct = 100.0 * (1 - total_bytes / original_size_bytes)

    print(f"\n--- Compression Summary ---")
    print(f"  Original mesh size        : {original_size_kb:>10.2f} KB")
    print(f"  Height maps (6 PNGs)      : {png_kb:>10.2f} KB")
    print(f"  Residual vertices (NPZ)   : {residual_kb:>10.2f} KB")
    print(f"  Total compressed size     : {total_kb:>10.2f} KB")
    print(f"  Compression ratio         : {compression_ratio:>10.1f}x")
    print(f"  Space saved               : {compression_pct:>10.2f}%")

    # Decode
    pcd = decode_hybrid(encoded_data)

    # Reconstruct
    recon_mesh = reconstruct_mesh_poisson(pcd, depth=9)

    # Evaluate
    metrics = evaluate_reconstruction(mesh, recon_mesh)
    print("\n--- Reconstruction Quality ---")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.6f}")

    # Save decoded mesh
    os.makedirs("data/decoded", exist_ok=True)
    o3d.io.write_triangle_mesh("data/decoded/hybrid_reconstructed.ply", recon_mesh)
    print("\nSaved: data/decoded/hybrid_reconstructed.ply")

    # Render comparison
    os.makedirs("results/plots", exist_ok=True)
    render_mesh_comparison(mesh_path, recon_mesh, "results/plots/hybrid_comparison.png")

    print(f"\n{'='*60}")
    print(f"  RESULT: Hybrid Residual achieved {compression_pct:.2f}% compression")
    print(f"          with {metrics['normal_consistency']*100:.1f}% normal consistency!")
    print(f"{'='*60}\n")

    return recon_mesh, metrics, compression_pct


if __name__ == "__main__":
    run_hybrid(resolution=256, threshold=0.005)
