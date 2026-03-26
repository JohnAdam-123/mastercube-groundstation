import numpy as np

def compute_pca_svd(X):
    """
    Perform PCA using SVD
    X: (N, D) data matrix
    """

    # Center the data
    mean = np.mean(X, axis = 0)
    X_centered = X - mean

    # SVD Decomposition
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Principal components
    components = Vt

    return components, S, mean

def project_data(X, components, mean, k = 2):
    """
    Project data onto top-k principal components
    """
    X_centered = X - mean
    return np.dot(X_centered, components[:k].T)

def reconstruct_data(X, components, mean, k):
    """
    Reconstruct data from top-k principal components
    """
    # Center data
    X_centered = X - mean

    # Project to lower dimension
    projected = X_centered @ components[:k].T

    # Reconstruct back to original space
    reconstructed = projected @ components[:k] + mean

    return reconstructed
