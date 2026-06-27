import numpy as np
import pywt
import gzip
import os
import time
from PIL import Image
import io

SCALE_FACTOR = 20000.0
MAX_DEPTH = 3.0

def _to_uint16(float_map):
    m = np.copy(float_map)
    is_bg = np.isinf(m)
    m[is_bg] = 0.0
    m = np.clip(m, 0, MAX_DEPTH)
    return (m * SCALE_FACTOR).astype(np.uint16)

def _to_float(uint_map):
    m = uint_map.astype(np.float32) / SCALE_FACTOR
    m[uint_map == 0] = np.inf
    return m

def compress_png(height_maps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sizes = {}
    for face, hmap in height_maps.items():
        uint_map = _to_uint16(hmap)
        img = Image.fromarray(uint_map, mode='I;16')
        path = os.path.join(save_dir, f"{face}.png")
        img.save(path, format='PNG')
        sizes[face] = os.path.getsize(path)
    return sum(sizes.values()), sizes

def decompress_png(load_dir, resolution):
    faces = ['front', 'back', 'left', 'right', 'top', 'bottom']
    height_maps = {}
    for face in faces:
        path = os.path.join(load_dir, f"{face}.png")
        img = Image.open(path)
        uint_map = np.array(img, dtype=np.uint16)
        height_maps[face] = _to_float(uint_map)
    return height_maps

def compress_npz(height_maps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "maps.npz")
    # Compress directly as float
    np.savez_compressed(path, **height_maps)
    size = os.path.getsize(path)
    return size, {'all': size}

def compress_gzip(height_maps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sizes = {}
    for face, hmap in height_maps.items():
        path = os.path.join(save_dir, f"{face}.gz")
        uint_map = _to_uint16(hmap)
        with gzip.open(path, 'wb') as f:
            f.write(uint_map.tobytes())
        sizes[face] = os.path.getsize(path)
    return sum(sizes.values()), sizes

def evaluate_compressions(height_maps, save_dir):
    """
    Evaluates multiple compression methods.
    Returns a dictionary of results.
    """
    results = {}
    
    # 1. NPZ (NumPy Compressed - Float baseline)
    start = time.time()
    npz_size, _ = compress_npz(height_maps, os.path.join(save_dir, "npz"))
    t_npz = time.time() - start
    results['NPZ'] = {'size_bytes': npz_size, 'time_enc': t_npz}
    
    # 2. PNG (16-bit Lossless Image Compression)
    start = time.time()
    png_size, _ = compress_png(height_maps, os.path.join(save_dir, "png"))
    t_png = time.time() - start
    results['PNG'] = {'size_bytes': png_size, 'time_enc': t_png}
    
    # 3. GZIP (Raw 16-bit array gzip)
    start = time.time()
    gz_size, _ = compress_gzip(height_maps, os.path.join(save_dir, "gzip"))
    t_gz = time.time() - start
    results['GZIP'] = {'size_bytes': gz_size, 'time_enc': t_gz}
    
    # Can add Wavelet, JPEG-XL wrappers later if needed.
    
    return results
