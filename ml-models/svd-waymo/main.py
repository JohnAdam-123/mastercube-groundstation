import numpy as np
import matplotlib.pyplot as plt
from svd_pca import compute_pca_svd, project_data
from mpl_toolkits.mplot3d import Axes3D

# Simulated 3D point cloud
np.random.seed(42)
points = np.random.rand(100, 3)

# Apply PCA via SVD
components, S, mean = compute_pca_svd(points)

# Reduced to 2D
projected = project_data(points, components, mean, k = 2)

print("Original shape:", points.shape)
print("Reduced shape:", projected.shape)
print("Top components:\n", components[:2])
print("Singular values:", S)

# Visualizing the original 3D data (first 2 dimensions for simplicity)
plt.figure()
plt.scatter(points[:, 0], points[:, 1])
plt.title("Original Data (2D view)")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("original.png")
plt.show()
plt.close()

# Visualizing the projected 2D data
plt.figure()
plt.scatter(projected[:, 0], projected[:, 1])
plt.title("PCA Reduced Data (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("reduced.png")
plt.show()

# 3D visualization of original point cloud data
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2],c=points[:, 2])
ax.set_title("3D Point Cloud (Original)")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.savefig("Original_3D.png")
plt.show()