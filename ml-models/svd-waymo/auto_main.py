import os
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from data_loaders import (
    load_kitti_bin,
    load_waymo_npy_frames,
    batch_waymo_frames,
    load_nuscenes_bin,
    batch_nuscenes_frames,
)

# ----------------------------
# CONFIGURATION
# ----------------------------
dataset_choice = "waymo"  # options: "kitti", "waymo", "nuscenes"
waymo_folder = "./datasets/waymo_npy"
nuscenes_folder = "./datasets/nuscenes_bin"
kitti_file = "./datasets/kitti/0000000000.bin"
batch_size = 4
max_frames = 20  # None for all frames
eps = 0.5
min_samples = 5
save_clusters = True
visualize_clusters = True
output_folder = "./clustered_frames"

os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
if dataset_choice.lower() == "kitti":
    frames = [load_kitti_bin(kitti_file)]
    print(f"Loaded KITTI shape: {frames[0].shape}")

elif dataset_choice.lower() == "waymo":
    frames = load_waymo_npy_frames(waymo_folder, max_frames=max_frames)
    frames = batch_waymo_frames(frames, batch_size=batch_size)
    print(f"Loaded Waymo {len(frames)} batches, batch size {batch_size}")

elif dataset_choice.lower() == "nuscenes":
    frames = load_nuscenes_bin(nuscenes_folder, max_frames=max_frames)
    frames = batch_nuscenes_frames(frames, batch_size=batch_size)
    print(f"Loaded Nuscenes {len(frames)} batches, batch size {batch_size}")

else:
    raise ValueError("Invalid dataset choice. Choose 'kitti', 'waymo', or 'nuscenes'.")

# ----------------------------
# PROCESS BATCHES (DBSCAN + Save + Visualize)
# ----------------------------
for batch_idx, batch in enumerate(frames):
    print(f"\nProcessing batch {batch_idx+1}/{len(frames)}")

    # If single-frame dataset (KITTI), wrap it in a list
    if dataset_choice.lower() == "kitti":
        batch = [batch]

    for frame_idx, frame in enumerate(batch):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(frame[:, :3])
        labels = clustering.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f" Frame {frame_idx+1}: {frame.shape[0]} points, {num_clusters} clusters found")

        # Save clustered frame
        if save_clusters:
            out_file = os.path.join(output_folder, f"{dataset_choice}_batch{batch_idx}_frame{frame_idx}.npy")
            np.save(out_file, np.hstack((frame, labels.reshape(-1, 1))))
        
        # Optional 3D visualization
        if visualize_clusters:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            unique_labels = set(labels)
            colors = plt.cm.get_cmap('tab20', len(unique_labels))
            for k in unique_labels:
                class_member_mask = (labels == k)
                xyz = frame[class_member_mask, :3]
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, color=colors(k), label=f"Cluster {k}")
            ax.set_title(f"Batch {batch_idx} Frame {frame_idx} DBSCAN Clusters")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.legend()
            plt.show()