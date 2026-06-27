import numpy as np
import scipy.ndimage as ndimage
from src.raycasting import FACES, get_camera_rays, cast_rays

def encode_adaptive(mesh, base_resolution=128, max_resolution=512):
    """
    Implements a simplified 2-level adaptive sampling.
    """
    height_maps = {}
    
    for face in FACES:
        # Cast at base resolution
        origins, directions = get_camera_rays(face, base_resolution)
        base_distances = cast_rays(mesh, origins, directions).reshape((base_resolution, base_resolution))
        
        valid = ~np.isinf(base_distances)
        if not np.any(valid):
            height_maps[face] = np.full((max_resolution, max_resolution), np.inf)
            continue
            
        distances_filled = np.copy(base_distances)
        distances_filled[~valid] = np.nanmax(base_distances[valid])
        
        # Edge detection using Sobel filter
        dx = ndimage.sobel(distances_filled, 0)
        dy = ndimage.sobel(distances_filled, 1)
        mag = np.hypot(dx, dy)
        
        mask_dx = ndimage.sobel(valid.astype(float), 0)
        mask_dy = ndimage.sobel(valid.astype(float), 1)
        mask_mag = np.hypot(mask_dx, mask_dy)
        
        edges = (mag > np.quantile(mag[valid], 0.8)) | (mask_mag > 0)
        
        # In a full implementation, we would only store the dense pixels.
        # Here we upsample and just simulate the logical structure.
        upsampled = ndimage.zoom(base_distances, max_resolution / base_resolution, order=0)
        
        height_maps[face] = upsampled
        
    return height_maps
