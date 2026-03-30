import numpy as np

def extract_features(point_cloud):
    # Apply SVD / PCA
    u, s, vh = np.linalg.svd(point_cloud, full_matrices=False)
    
    return {
        "singular_values": s.tolist(),
        "rank": len(s)
    }