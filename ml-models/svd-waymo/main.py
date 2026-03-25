import numpy as np
import matplotlib.pyplot as plt
from svd_pca import compute_pca_svd, project_data
from mpl_toolkits.mplot3d import Axes3D
from kitti_loader import load_kitti_bin
import os

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

intensity = points[:, 3] # Intensity values for coloring

# Normalize intensity for better visualization
intensity = np.log1p(intensity) # Log scale for better contrast
intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

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