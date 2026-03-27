import os
import numpy as np

# ----------------------------
# WAYMO DATA HELPERS
# ----------------------------
def load_waymo_npy_frames(folder_path, max_frames=None):
    """Load all .npy Waymo frames from a folder."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    frames = []
    for i, file in enumerate(files):
        if max_frames is not None and i >= max_frames:
            break
        frame = np.load(os.path.join(folder_path, file))
        frames.append(frame)
    return frames

def batch_waymo_frames(frames, batch_size=4):
    """Split Waymo frames into batches."""
    batches = []
    for i in range(0, len(frames), batch_size):
        batches.append(frames[i:i + batch_size])
    return batches

# ----------------------------
# NUSCENES DATA HELPERS
# ----------------------------
def load_nuscenes_bin(folder_path, max_frames=None):
    """Load Nuscenes frames stored as .bin LiDAR files."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".bin")])
    frames = []
    for i, file in enumerate(files):
        if max_frames is not None and i >= max_frames:
            break
        frame = np.fromfile(os.path.join(folder_path, file), dtype=np.float32).reshape(-1, 4)
        frames.append(frame)
    return frames

def batch_nuscenes_frames(frames, batch_size=4):
    """Split Nuscenes frames into batches."""
    batches = []
    for i in range(0, len(frames), batch_size):
        batches.append(frames[i:i + batch_size])
    return batches

# ----------------------------
# KITTI HELPER (OPTIONAL)
# ----------------------------
def load_kitti_bin(file_path):
    """Load a single KITTI LiDAR frame from .bin file."""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points