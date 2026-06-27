import numpy as np
import trimesh
from src.raycasting import FACES, get_camera_rays

def cast_rays_multilayer(mesh, origins, directions, num_layers=4):
    """
    Casts rays and finds multiple intersections for depth peeling.
    """
    N = len(origins)
    all_distances = []
    
    current_origins = np.copy(origins)
    current_directions = np.copy(directions)
    
    # Track the cumulative distance from the initial camera plane
    cumulative_distances = np.zeros(N)
    
    for layer in range(num_layers):
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=current_origins,
            ray_directions=current_directions,
            multiple_hits=False
        )
        
        layer_distances = np.full(N, np.inf)
        
        if len(locations) > 0:
            hit_origins = current_origins[index_ray]
            hit_distances = np.linalg.norm(locations - hit_origins, axis=1)
            
            layer_distances[index_ray] = cumulative_distances[index_ray] + hit_distances
            
            # Update origins for the next cast to be slightly past the hit point
            current_origins[index_ray] = locations + current_directions[index_ray] * 1e-4
            cumulative_distances[index_ray] += hit_distances + 1e-4
            
        all_distances.append(layer_distances)
        
    return all_distances
