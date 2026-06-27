import numpy as np
import time
import os
from src.raycasting import FACES, get_camera_rays, cast_rays

def encode_mesh(mesh, resolution=512):
    """
    Generates 6 height maps for the given mesh.
    """
    height_maps = {}
    print(f"Encoding mesh at resolution {resolution}x{resolution}...")
    start_time = time.time()
    
    for face in FACES:
        origins, directions = get_camera_rays(face, resolution)
        distances = cast_rays(mesh, origins, directions)
        height_map = distances.reshape((resolution, resolution))
        height_maps[face] = height_map
        print(f"  Processed {face} face")
        
    end_time = time.time()
    print(f"Encoding took {end_time - start_time:.2f} seconds")
    return height_maps

def save_height_maps_npy(height_maps, save_dir):
    """
    Saves the float height maps directly as .npy files.
    """
    os.makedirs(save_dir, exist_ok=True)
    for face, hmap in height_maps.items():
        np.save(os.path.join(save_dir, f"{face}.npy"), hmap)
    print(f"Saved raw height maps to {save_dir}")
