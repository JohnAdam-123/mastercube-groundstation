import numpy as np
import matplotlib.pyplot as plt
import os

from svd_pca import compute_pca_svd, project_data, reconstruct_data
from mpl_toolkits.mplot3d import Axes3D
from kitti_loader import load_kitti_bin
from sklearn.cluster import DBSCAN


# After loading the KITTI dataset, I had to replace my random data with actual point cloud data from the dataset.
""" The following code assumes you have the KITTI dataset downloaded and the `kitti_loader` module,
    properly set up to read the binary point cloud files.

    That was this here below:-

    # Simulated 3D point cloud
    np.random.seed(42)
    points = np.random.rand(100, 3)

"""
file_path = os.path.expanduser("~/datasets/kitti/0000000000.bin")

points = load_kitti_bin(file_path)

# Separating spatial + intensity data
spatial_points = points[:, :3] # (X, Y)
intensity = points[:, :3] 

print("Loaded KITTI shape:", spatial_points.shape)

# Apply PCA via SVD
components, S, mean = compute_pca_svd(spatial_points)

# Reduced to 2D
projected = project_data(spatial_points, components, mean, k = 2)

print("Original shape:", points.shape)
print("Reduced shape:", projected.shape)
print("Top components:\n", components[:2])
print("Singular values:", S)

# Reconstruct using top 2 principal components
reconstructed = reconstruct_data(spatial_points, components, mean, k = 2)

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


# VISUALIZATIONS

# Visualizing the original 3D data (first 2 dimensions for simplicity)
plt.figure(figsize=(8, 6))
plt.scatter(
    spatial_points[:, 0],
    spatial_points[:, 1],
    c = intensity,
    cmap = 'viridis',
    s = 1
)

plt.title("KITTI LiDAR Top View")
plt.colorbar(label = "Intensity")

plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("kitti_topview.png")
plt.show()
plt.close()

# 3D visualization of original point cloud data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(
    spatial_points[:, 0],
    spatial_points[:, 1],
    spatial_points[:, 2],
    c = intensity,
    cmap = 'viridis',
    s = 1
)

ax.set_title("KITTI LiDAR 3D View")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.savefig("kitti_3d.png")
plt.show()

# 2D visualization of original point cloud data
plt.figure(figsize=(10, 7))

plt.scatter(
    spatial_points[:, 0],
    spatial_points[:, 1],
    c=intensity,
    cmap='viridis',
    s=1 
)

plt.title("Original LiDAR")
plt.savefig("kitti_original.png")
plt.show()

# 3D visualization of reconstructed point cloud data
plt.figure()

plt.scatter(
    reconstructed[:, 0],
    reconstructed[:, 1],
    c=intensity,
    cmap='viridis',
    s=1
)

plt.title("Reconstructed LiDAR")
plt.savefig("Reconstructed.png")
plt.show()

# Noise Visualization
noise = spatial_points - reconstructed

plt.figure()
plt.scatter(
    noise[:, 0],
    noise[:, 1],
    c=intensity,
    cmap='viridis',
    s=1
)

plt.title("Noise / Residual")
plt.savefig("Noise.png")
plt.show()

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
plt.title("Anomaly Map Visualization (Top View)")
plt.savefig("anomaly_map.png")
plt.show()

# Visualize anomalies in 3D ONLY
plt.figure()

# Normal points (Light)
plt.scatter(
    spatial_points[~anomalies, 0],
    spatial_points[~anomalies, 1],
    s = 1,
    alpha = 0.3,
    label = "Normal points",
    c = 'blue'
)

# Anomalies (Hightlighted)
plt.scatter(
    spatial_points[anomalies, 0],
    spatial_points[anomalies, 1],
    s = 2,
    alpha = 0.8,
    label = "Anomalies",
    c = 'red'
)
plt.legend()
plt.title("Detected Anomalies in KITTI LiDAR (Top View)")
plt.savefig("detected_anomalies.png")
plt.show()

# Visualizing clusters of anomalies in 3D
plt.figure()

plt.scatter(
    anomaly_points[:, 0],
    anomaly_points[:, 1],
    c = labels,
    cmap = 'tab10',
    s = 2
)

plt.title("Clustered Anomalies in KITTI LiDAR (Top view Objects)")
plt.colorbar(label = "Cluster Label")
plt.savefig("clustered_objects_anomalies.png")
plt.show()


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

plt.title("Detected objects with Bounding Boxes")
plt.savefig("detected_objects.png")
plt.show()