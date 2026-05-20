import numpy as np


def compute_connectivity_matrix(timeseries: np.ndarray) -> np.ndarray:
    """
    Compute an ROI-to-ROI functional connectivity matrix using Pearson correlation.

    Parameters
    ----------
    timeseries : np.ndarray
        Array of shape (n_timepoints, n_regions), where each column represents
        the fMRI-like signal from one brain region.

    Returns
    -------
    np.ndarray
        Correlation matrix of shape (n_regions, n_regions).
    """

    if not isinstance(timeseries, np.ndarray):
        timeseries = np.asarray(timeseries)

    if timeseries.ndim != 2:
        raise ValueError("timeseries must be a 2D array of shape (n_timepoints, n_regions).")

    if timeseries.shape[0] < 2:
        raise ValueError("timeseries must contain at least two time points.")

    return np.corrcoef(timeseries.T)


def flatten_upper_triangle(matrix: np.ndarray, include_diagonal: bool = False) -> np.ndarray:
    """
    Flatten the upper triangle of a square connectivity matrix into a feature vector.

    Parameters
    ----------
    matrix : np.ndarray
        Square connectivity matrix.
    include_diagonal : bool
        Whether to include diagonal values. For connectivity features, this is
        usually False because diagonal values are self-correlations.

    Returns
    -------
    np.ndarray
        One-dimensional vector containing upper-triangle connectivity values.
    """

    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2D array.")

    k = 0 if include_diagonal else 1
    upper_indices = np.triu_indices_from(matrix, k=k)

    return matrix[upper_indices]


def create_connectivity_feature_names(region_names):
    """
    Create readable feature names for upper-triangle connectivity values.

    Example:
        temporal__occipital_connectivity
    """

    feature_names = []

    for i in range(len(region_names)):
        for j in range(i + 1, len(region_names)):
            feature_names.append(f"{region_names[i]}__{region_names[j]}_connectivity")

    return feature_names