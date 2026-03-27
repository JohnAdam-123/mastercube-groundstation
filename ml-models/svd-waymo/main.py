import numpy as np
import matplotlib.pyplot as plt
import os

# Import existing PCA/SVD functions
from svd_pca import compute_pca_svd, project_data, reconstruct_data

from mpl_toolkits.mplot3d import Axes3D

# Choose dataset loader (waymo_loader or kitti_loader)
from kitti_loader import load_kitti_bin 
from waymo_loader import load_waymo_frame
from data_loaders import(
    load_kitti_bin,
    load_waymo_npy_frames,
    batch_waymo_frames,
    load_nuscenes_bin,
    batch_nuscenes_frames
)

from sklearn.cluster import DBSCAN

# ----------------------------
# CONFIGURATION
# ----------------------------
dataset_choice = "kitti"  # options: "kitti", "waymo", or "nuscenes"

# Paths for each dataset
kitti_file = "./datasets/kitti/0000000000.bin"
waymo_folder = "./datasets/waymo_npy"  # folder with Waymo .npy frames
nuscenes_folder = "./datasets/nuscenes_bin"  # folder with Nuscenes .bin frames

# Optional: how many frames to load (None = all)
batch_size = 4  # number of frames per batch
max_frames = 100

# ----------------------------
# LOAD DATA
# ----------------------------
if dataset_choice.lower() == "kitti":
    # KITTI is usually one frame per file, so we wrap it in a batch manually
    points = load_kitti_bin(kitti_file)
    batches = [points]  # single "batch" with 1 frame
    print("Loaded KITTI shape:", points.shape)

elif dataset_choice.lower() == "waymo":
    frames = load_waymo_npy_frames(waymo_folder, max_frames=max_frames)
    batches = batch_waymo_frames(frames, batch_size=batch_size)
    print(f"Loaded Waymo frames: {len(frames)}, batches: {len(batches)}")

elif dataset_choice.lower() == "nuscenes":
    frames = load_nuscenes_bin(nuscenes_folder, max_frames=max_frames)
    batches = batch_nuscenes_frames(frames, batch_size=batch_size)
    print(f"Loaded Nuscenes frames: {len(frames)}, batches: {len(batches)}")

else:
    raise ValueError("Invalid dataset choice. Please choose 'kitti', 'waymo', or 'nuscenes'.")

# ----------------------------
# EXAMPLE PROCESSING LOOP
# ----------------------------
for batch_idx, batch in enumerate(batches):
    # batch is a list/array of frames
    print(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} frames")
    # Example: run DBSCAN or your model here on the batch
    # dbscan_result = run_dbscan(batch)


# Separating spatial + intensity data, keep only XYZ for PCA/SVD intensity coloring later
spatial_points = points[:, :3] # (X, Y, Z)
intensity = points[:, 3] # Intensity values

print("Loaded Original shape:", points.shape)
print("Loaded Spatial points shape:", spatial_points.shape)

# Apply PCA via SVD
components, S, mean = compute_pca_svd(spatial_points)

# Reduced to 2D
projected = project_data(spatial_points, components, mean, k = 2)

print("Original shape:", points.shape)
print("Reduced shape:", projected.shape)
print("Top components:\n", components[:2])
print("Singular values:", S)

# Reconstruct using top 2 principal components
reconstructed = reconstruct_data(projected, components, mean, k = 2)

print("Reconstructed shape:", reconstructed.shape)

intensity = points[:, 3] # Intensity values for coloring

# Normalize intensity for better visualization
intensity = np.log1p(intensity) # Log scale for better contrast
intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

# Residual (difference between original and reconstructed)
residual = spatial_points - reconstructed

# Magnitude of residuals per point
residual_norm = np.linalg.norm(residual, axis = 1)

print("Residual shape:", residual.shape)
print("Residual norm stats:",
      np.min(residual_norm),
      np.max(residual_norm),
      np.mean(residual_norm))

# Threshold for anomaly detection (separating anomalies from normal points)
# Threshold ( This can be tuned based on the distribution of residual norms)
threshold = np.mean(residual_norm) + 2 * np.std(residual_norm)

# Create mask for anomalies
anomalies = residual_norm > threshold

print("Threshold:", threshold)
print("Number of anomalies points detected:", np.sum(anomalies))

# Extracting anomaly points for clustering
anomaly_points = spatial_points[anomalies]

# Applying Clustering (DBSCAN) to anomaly points
clustering = DBSCAN(eps = 1.5, min_samples = 10).fit(anomaly_points)

labels = clustering.labels_

print("Number of clusters detected:", len(set(labels)) - (1 if -1 in labels else 0))
print("Cluster labels:", set(labels))

# ----------------------------
# 3D Visualization
# ----------------------------

# Visualizing the original 3D data (first 2 dimensions for simplicity)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(spatial_points[:,0], spatial_points[:,1], spatial_points[:,2], s=0.5, c='blue')
ax.set_title(f"{dataset_choice.upper()} LiDAR 3D Points")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig(f"{dataset_choice.upper()}_3d.png")
plt.show()
plt.close()

# ----------------------------
# 2D PCA Projection Visualization
# ----------------------------
plt.figure(figsize=(8,6))
plt.scatter(projected[:,0], projected[:,1], s=0.5, c='red')
plt.title(f"{dataset_choice.upper()} PCA 2D Projection")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis('equal')
plt.savefig(f"{dataset_choice.upper()}_2d_pca.png")
plt.show()
plt.close()



# 3D visualization of reconstructed point cloud data
plt.figure(figsize=(8,6))
ax = plt.subplot(111, projection='3d')
ax.scatter(reconstructed[:,0], reconstructed[:,1], reconstructed[:,2], s=0.5, c='green')
ax.set_title(f"{dataset_choice.upper()} Reconstructed 3D Points (Top 2 PCs)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig(f"{dataset_choice.upper()}_reconstructed_3d.png")
plt.show()
plt.close()


# Noise Visualization
noise = spatial_points - reconstructed

plt.figure(figsize=(8,6))
plt.scatter(
    noise[:, 0],
    noise[:, 1],
    s=0.5,
    c='red'
)
plt.title(f"Noise / Residual Visualization from {dataset_choice.upper()} PCA Reconstruction")
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(f"Noise_in_{dataset_choice.upper()}.png")
plt.show()
plt.close()


# Residual norm visualization (Anomalies Visualization)
plt.figure()
plt.scatter(
    spatial_points[:, 0],
    spatial_points[:, 1],
    c=residual_norm,
    s = 1,
    cmap = 'inferno'
)

plt.colorbar(label = "Residual Magnitude")
plt.title(f"Anomaly Map Visualization for {dataset_choice.upper()}")
plt.savefig(f"anomaly_map_in_{dataset_choice.upper()}.png")
plt.show()
plt.close()

# Visualize anomalies in 3D ONLY
plt.figure(figsize=(8,6))
ax = plt.subplot(111, projection='3d')
ax.scatter(
    spatial_points[anomalies, 0],
    spatial_points[anomalies, 1],
    spatial_points[anomalies, 2],
    c='red',
    s=2,
    alpha=0.8,
    label = "Anomalies"
)
ax.scatter(
    spatial_points[anomalies, 0],
    spatial_points[anomalies, 1],
    spatial_points[anomalies, 2],
    c='blue',
    s=1,
    alpha=0.3,
    label = "Normal points"
)
plt.legend()
ax.set_title(f"{dataset_choice.upper()} Detected Anomalies (3D)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig(f"{dataset_choice.upper()}_anomalies_3d.png")
plt.show()
plt.close()


# Visualizing clusters of anomalies in 3D
plt.figure()

plt.scatter(
    anomaly_points[:, 0],
    anomaly_points[:, 1],
    c = labels,
    cmap = 'tab10',
    s = 2
)

plt.title(f"Clustered Anomalies in {dataset_choice.upper()} (Top view Objects)")
plt.colorbar(label = "Cluster Label")
plt.savefig(f"clustered_objects_anomalies_in_{dataset_choice.upper()}.png")
plt.show()
plt.close()


# OBJECT REPRESENTATION
# For each cluster, we can compute a simple bounding box or centroid to represent the detected object
plt.figure()

# Plot all anomaly points in light color
plt.scatter(
    anomaly_points[:, 0],
    anomaly_points[:, 1],
    c = 'lightgray',
    s = 1,
    alpha = 0.3,
    label = "Anomaly points"
)
# Draw bounding boxes for each cluster
unique_labels = set(labels)

for label in unique_labels:
    if label == -1:
        continue # skip noise points

    cluster = anomaly_points[labels == label]

    x_min, y_min = np.min(cluster[:, 0]), np.min(cluster[:, 1])
    x_max, y_max = np.max(cluster[:, 0]), np.max(cluster[:, 1])

    # Draw rectangle (bounding box)
    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        edgecolor='red',
        linewidth=2
    )
    plt.gca().add_patch(rect)

plt.legend()
plt.title("Detected objects with Bounding Boxes in {dataset_choice.upper()}")
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(f"detected_objects_{dataset_choice.upper()}.png")
plt.show()
plt.close()