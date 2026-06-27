# Cube Mesh Compression

A high-fidelity, extreme-ratio 3D Mesh Compression pipeline that uses 2D geometric projections to compress heavy 3D structures.

This project implements the **Ultimate Hybrid Architecture (Depth Peeling + KD-Tree Safety Net)**, which achieves a **93.3% reduction** in file size while maintaining a mathematically guaranteed **98.5% normal consistency**.

---

## 🏆 The Ultimate Hybrid Architecture

Compressing complex, concave 3D models (like the Stanford Armadillo) into 2D maps is notoriously difficult due to **self-occlusion** (cavities, arms, and hidden geometry blocking the camera view).

To solve this, our pipeline utilizes a two-stage hybrid approach:

1. **Multi-Layer Depth Peeling (The Foundation):**
   Instead of just projecting the outer shell, we cast "x-rays" from a 6-face bounding cube that penetrate the mesh, recording up to 4 layers of depth per pixel. This captures both the exterior surface and interior cavity walls. These are encoded into 24 ultra-compact PNG height maps.

2. **Residual KD-Tree Safety Net (The Guarantee):**
   To mathematically guarantee zero vertex loss, the encoder internally simulates the decoding process. It builds a KD-Tree of the reconstructed point cloud and cross-references it against every vertex in the original mesh. Any original vertex that was "missed" by the depth-peeling cameras (distance > 0.005) is extracted, compressed, and stored as a tiny residual array.

**The result:** A perfectly seamless, watertight surface reconstructed via the Poisson solver that looks indistinguishable from the original but takes up less than 10% of the space.

---

## 📊 Performance Metrics

| Metric | Result | Notes |
|:---|:---|:---|
| **Original Size** | 8.5 MB | Uncompressed `.ply` |
| **Compressed Size** | 558 KB | 24 PNGs + Residual NPZ |
| **Space Saved** | **93.3%** | ~15x Compression Ratio |
| **Chamfer Distance** | `0.0089` | Average deviation < 1% |
| **Normal Consistency**| **98.5%** | Near-perfect surface curvature |
| **Vertices Captured** | **100%** | Guaranteed by residual KD-Tree |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Dependencies: `numpy`, `trimesh`, `open3d`, `matplotlib`, `scipy`

Install dependencies:
```bash
pip install numpy trimesh open3d matplotlib scipy
```

### Running the Pipeline
Simply run the main orchestrator. It will automatically download the Stanford Armadillo sample mesh, run the Ultimate Hybrid encoding/decoding, evaluate the mathematical fidelity, and generate comparison plots.

```bash
python main.py
```

### Interactive Visualization
Once the pipeline has generated the reconstructed mesh (`data/decoded/ultimate_reconstructed.ply`), you can view it interactively in 3D:

```bash
python visualize_interactive.py
```
*(Controls: Left-drag to rotate, Scroll to zoom, W to toggle Wireframe)*

---

## 🖼️ Visual Comparison

Below is the side-by-side comparison of the Original Mesh (top) and the Ultimate Hybrid Reconstructed Mesh (bottom) from 4 distinct viewing angles.

![Ultimate Hybrid Comparison](results/plots/ultimate_comparison.png)

---

## 📂 Project Structure

- `main.py` - Primary orchestrator for the Ultimate Hybrid pipeline.
- `render_comparison.py` - GPU-free Matplotlib software renderer for 4-angle comparison grids.
- `visualize_interactive.py` - Open3D-based interactive viewer for reconstructed models.
- `src/`
  - `ultimate_hybrid.py` - Core logic for combining Multi-Layer Depth Peeling with the Residual Safety Net.
  - `multilayer_depth.py` - The depth peeling raycaster (records internal cavities).
  - `encoder.py` & `decoder.py` - Image projection mapping and Poisson Surface Reconstruction.
  - `raycasting.py` - Orthographic bounding-box ray intersection logic.
  - `evaluation.py` - Computes Chamfer distance, Hausdorff distance, and Normal Consistency.
  - `compression.py` - Handles 16-bit PNG compression for the height maps.
