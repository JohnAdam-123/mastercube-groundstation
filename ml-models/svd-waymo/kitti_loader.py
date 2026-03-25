import numpy as np

def load_kitti_bin(file_path):
    """
    Load KITTI LiDAR data from a .bin file
    Each point is repesented as (X, Y, Z, Intensity)
    """
    data = np.fromfile(file_path, dtype=np.float32)
    points = data.reshape(-1, 4) # Reshape to (N, 4)
    return points