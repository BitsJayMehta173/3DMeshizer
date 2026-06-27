import numpy as np
import open3d as o3d
import trimesh
import time

def compute_normal_consistency(pcd1, pcd2):
    """
    Computes the mean absolute dot product of normals for nearest neighbors.
    """
    if not pcd1.has_normals() or not pcd2.has_normals():
        return 0.0
        
    tree = o3d.geometry.KDTreeFlann(pcd2)
    normals1 = np.asarray(pcd1.normals)
    normals2 = np.asarray(pcd2.normals)
    points1 = np.asarray(pcd1.points)
    
    consistency = []
    for i in range(len(points1)):
        _, idx, _ = tree.search_knn_vector_3d(points1[i], 1)
        n1 = normals1[i]
        n2 = normals2[idx[0]]
        # absolute dot product to handle flipped normals
        dot_prod = np.abs(np.dot(n1, n2)) 
        consistency.append(dot_prod)
        
    return np.mean(consistency)

def evaluate_reconstruction(orig_mesh_t, recon_mesh_o, num_samples=100000):
    """
    orig_mesh_t: trimesh.Trimesh
    recon_mesh_o: open3d.geometry.TriangleMesh
    """
    print(f"Evaluating reconstruction (Chamfer, Hausdorff, etc) with {num_samples} samples...")
    start = time.time()
    
    # Sample from original trimesh
    points1, face_indices = trimesh.sample.sample_surface(orig_mesh_t, num_samples)
    normals1 = orig_mesh_t.face_normals[face_indices]
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.normals = o3d.utility.Vector3dVector(normals1)
    
    # Prepare reconstructed mesh
    recon_mesh_o.compute_vertex_normals()
    # Sample from reconstructed open3d mesh
    # open3d sample_points_uniformly doesn't reliably sample normals in all versions, 
    # but Poisson reconstruction point clouds do have normals. Let's sample uniformly and use vertex normals.
    pcd2 = recon_mesh_o.sample_points_uniformly(number_of_points=num_samples)
    
    # If the sampled pcd2 doesn't have normals, we'll assign them using nearest neighbor from mesh vertices
    if not pcd2.has_normals():
        recon_mesh_o.compute_vertex_normals()
        vcd = o3d.geometry.PointCloud()
        vcd.points = recon_mesh_o.vertices
        vcd.normals = recon_mesh_o.vertex_normals
        
        tree = o3d.geometry.KDTreeFlann(vcd)
        normals2 = []
        p2_points = np.asarray(pcd2.points)
        for i in range(len(p2_points)):
            _, idx, _ = tree.search_knn_vector_3d(p2_points[i], 1)
            normals2.append(np.asarray(vcd.normals)[idx[0]])
        pcd2.normals = o3d.utility.Vector3dVector(np.vstack(normals2))

    # Compute distances
    dist1_to_2 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    dist2_to_1 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    
    chamfer = np.mean(dist1_to_2) + np.mean(dist2_to_1)
    hausdorff = max(np.max(dist1_to_2), np.max(dist2_to_1))
    rmse = np.sqrt((np.mean(dist1_to_2**2) + np.mean(dist2_to_1**2)) / 2.0)
    
    # Normal consistency
    nc_1_to_2 = compute_normal_consistency(pcd1, pcd2)
    nc_2_to_1 = compute_normal_consistency(pcd2, pcd1)
    normal_consistency = (nc_1_to_2 + nc_2_to_1) / 2.0
    
    print(f"Evaluation took {time.time() - start:.2f}s")
    
    return {
        'chamfer': chamfer,
        'hausdorff': hausdorff,
        'rmse': rmse,
        'normal_consistency': normal_consistency
    }
