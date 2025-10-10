"""
Spatial statistics for point patterns

Implements Ripley's K function and related statistics
"""

import numpy as np
from scipy.spatial import distance_matrix
from typing import Tuple


def ripley_k(
    points: np.ndarray,
    radii: np.ndarray,
    volume: float,
) -> np.ndarray:
    """
    Compute Ripley's K function
    
    K(r) = (1/λn) Σᵢ Σⱼ≠ᵢ I(dᵢⱼ < r)
    
    Args:
        points: Point coordinates (N, 3)
        radii: Array of radii to evaluate
        volume: Volume of observation window
    
    Returns:
        K values for each radius
    """
    n = len(points)
    intensity = n / volume
    
    # Compute pairwise distances
    dists = distance_matrix(points, points)
    np.fill_diagonal(dists, np.inf)
    
    K_values = []
    for r in radii:
        # Count pairs within radius r
        count = np.sum(dists < r)
        K = count / (intensity * n)
        K_values.append(K)
    
    return np.array(K_values)


def ripley_l(K_values: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Compute L function (variance-stabilized K)
    
    L(r) = sqrt(K(r)/π) - r
    
    For Poisson process, L(r) ≈ 0
    L(r) > 0 indicates clustering
    L(r) < 0 indicates regularity
    """
    L_values = np.sqrt(K_values / np.pi) - radii
    return L_values


class RipleyK:
    """Ripley's K function analyzer"""
    
    def __init__(self, radius: float = 100.0):
        self.radius = radius
        self.volume = (4/3) * np.pi * radius**3
    
    def compute(
        self,
        points: np.ndarray,
        max_r: float = 50.0,
        num_radii: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute K and L functions
        
        Returns:
            radii, K_values, L_values
        """
        radii = np.linspace(0.1, max_r, num_radii)
        K_values = ripley_k(points, radii, self.volume)
        L_values = ripley_l(K_values, radii)
        
        return radii, K_values, L_values


def compute_spatial_statistics(points: np.ndarray, radius: float = 100.0) -> dict:
    """
    Compute various spatial statistics
    
    Args:
        points: Point coordinates
        radius: Domain radius
    
    Returns:
        Dictionary of statistics
    """
    n = len(points)
    volume = (4/3) * np.pi * radius**3
    intensity = n / volume
    
    # Nearest neighbor distances
    dists = distance_matrix(points, points)
    np.fill_diagonal(dists, np.inf)
    nn_dists = np.min(dists, axis=1)
    
    stats = {
        'num_points': n,
        'intensity': intensity,
        'mean_nn_dist': np.mean(nn_dists),
        'std_nn_dist': np.std(nn_dists),
        'min_nn_dist': np.min(nn_dists),
    }
    
    return stats

