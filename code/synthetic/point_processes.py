"""
Spatial point processes for synthetic organoid generation

Implements various point processes:
- Poisson (homogeneous and inhomogeneous)
- Matérn cluster process
- Strauss hard-core process

References:
- Illian et al. (2008), Statistical Analysis and Modelling of Spatial Point Patterns
- Diggle (2013), Statistical Analysis of Spatial and Spatio-Temporal Point Patterns
- Baddeley et al. (2015), Spatial Point Patterns with R
"""

import numpy as np
from scipy.spatial import distance_matrix
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class PoissonProcess:
    """
    Homogeneous Poisson point process in 3D sphere
    
    Complete spatial randomness (CSR) - baseline for comparison
    """
    
    def __init__(self, intensity: float = 100.0, radius: float = 100.0):
        """
        Initialize Poisson process
        
        Args:
            intensity: Expected number of points per unit volume
            radius: Radius of spherical domain
        """
        self.intensity = intensity
        self.radius = radius
    
    def generate(self, num_points: Optional[int] = None) -> np.ndarray:
        """
        Generate points from homogeneous Poisson process in sphere
        
        Args:
            num_points: Exact number of points (if None, use Poisson count)
        
        Returns:
            points: Array of shape (n, 3) with (x, y, z) coordinates
        """
        # Determine number of points
        if num_points is None:
            volume = (4/3) * np.pi * self.radius**3
            expected = self.intensity * volume
            num_points = np.random.poisson(expected)
        
        # Generate uniform points in sphere using rejection sampling
        points = []
        while len(points) < num_points:
            # Generate in cube
            candidates = np.random.uniform(-self.radius, self.radius, (num_points * 2, 3))
            # Keep only those inside sphere
            distances = np.linalg.norm(candidates, axis=1)
            inside = candidates[distances <= self.radius]
            points.extend(inside)
        
        points = np.array(points[:num_points])
        
        logger.info(f"Generated {len(points)} points from Poisson process")
        return points


class InhomogeneousPoissonProcess:
    """
    Inhomogeneous Poisson process with spatial gradient
    
    Models spatial heterogeneity (e.g., cell density gradients)
    """
    
    def __init__(
        self,
        intensity_func: Callable,
        radius: float = 100.0,
        max_intensity: float = 200.0,
    ):
        """
        Initialize inhomogeneous Poisson process
        
        Args:
            intensity_func: Function λ(x, y, z) -> intensity at that point
            radius: Radius of spherical domain
            max_intensity: Maximum intensity (for thinning)
        """
        self.intensity_func = intensity_func
        self.radius = radius
        self.max_intensity = max_intensity
    
    def generate(self, num_points: int) -> np.ndarray:
        """
        Generate points using thinning algorithm
        
        Args:
            num_points: Approximate number of points
        
        Returns:
            points: Generated points
        """
        # Generate from homogeneous Poisson with max intensity
        homog = PoissonProcess(self.max_intensity, self.radius)
        candidates = homog.generate(int(num_points * 1.5))  # Oversample
        
        # Thin according to intensity function
        probs = np.array([
            self.intensity_func(p[0], p[1], p[2]) / self.max_intensity
            for p in candidates
        ])
        
        keep = np.random.uniform(size=len(candidates)) < probs
        points = candidates[keep]
        
        logger.info(f"Generated {len(points)} points from inhomogeneous Poisson")
        return points


class MaternClusterProcess:
    """
    Matérn cluster process
    
    Models clustered (aggregated) patterns common in biological systems
    
    Reference: Matérn (1960)
    """
    
    def __init__(
        self,
        parent_intensity: float = 10.0,
        cluster_radius: float = 20.0,
        offspring_per_parent: float = 10.0,
        radius: float = 100.0,
    ):
        """
        Initialize Matérn cluster process
        
        Args:
            parent_intensity: Intensity of parent (cluster center) process
            cluster_radius: Radius of clusters
            offspring_per_parent: Mean number of offspring per parent
            radius: Radius of spherical domain
        """
        self.parent_intensity = parent_intensity
        self.cluster_radius = cluster_radius
        self.offspring_per_parent = offspring_per_parent
        self.radius = radius
    
    def generate(self, num_points: Optional[int] = None) -> np.ndarray:
        """
        Generate Matérn cluster process
        
        Args:
            num_points: Target number of points (adjusts parent intensity)
        
        Returns:
            points: Clustered points
        """
        # Generate parent points
        if num_points is not None:
            # Adjust parent intensity to get approximately num_points total
            volume = (4/3) * np.pi * self.radius**3
            adjusted_intensity = num_points / (self.offspring_per_parent * volume)
            parent_process = PoissonProcess(adjusted_intensity, self.radius)
        else:
            parent_process = PoissonProcess(self.parent_intensity, self.radius)
        
        parents = parent_process.generate()
        
        # Generate offspring around each parent
        all_offspring = []
        for parent in parents:
            # Number of offspring (Poisson distributed)
            n_offspring = np.random.poisson(self.offspring_per_parent)
            
            # Generate offspring uniformly in sphere around parent
            offspring = []
            while len(offspring) < n_offspring:
                # Uniform in sphere of cluster_radius
                candidates = np.random.uniform(
                    -self.cluster_radius,
                    self.cluster_radius,
                    (n_offspring * 2, 3)
                )
                candidates = candidates + parent
                
                # Keep only those inside cluster and inside domain
                dist_from_parent = np.linalg.norm(candidates - parent, axis=1)
                dist_from_origin = np.linalg.norm(candidates, axis=1)
                
                valid = (dist_from_parent <= self.cluster_radius) & (dist_from_origin <= self.radius)
                offspring.extend(candidates[valid])
            
            all_offspring.extend(offspring[:n_offspring])
        
        points = np.array(all_offspring)
        
        logger.info(f"Generated {len(points)} points in {len(parents)} clusters")
        return points


class StraussProcess:
    """
    Strauss hard-core inhibition process
    
    Models regular (inhibited) patterns due to minimum distance constraints
    
    Reference: Strauss (1975)
    """
    
    def __init__(
        self,
        intensity: float = 100.0,
        interaction_radius: float = 10.0,
        beta: float = 0.1,
        radius: float = 100.0,
    ):
        """
        Initialize Strauss process
        
        Args:
            intensity: Base intensity
            interaction_radius: Radius of inhibition
            beta: Interaction parameter (0 = hard-core, 1 = Poisson)
            radius: Radius of spherical domain
        """
        self.intensity = intensity
        self.interaction_radius = interaction_radius
        self.beta = beta
        self.radius = radius
    
    def generate(self, num_points: int, max_iter: int = 10000) -> np.ndarray:
        """
        Generate Strauss process using Metropolis-Hastings MCMC
        
        Args:
            num_points: Target number of points
            max_iter: Maximum MCMC iterations
        
        Returns:
            points: Regularly spaced points
        """
        # Initialize with Poisson process
        poisson = PoissonProcess(self.intensity, self.radius)
        points = poisson.generate(num_points)
        
        # Metropolis-Hastings sampling
        for iteration in range(max_iter):
            # Propose: add, delete, or move a point
            action = np.random.choice(['add', 'delete', 'move'], p=[0.3, 0.3, 0.4])
            
            if action == 'add' and len(points) < num_points * 1.5:
                # Add new point
                new_point = self._generate_uniform_point()
                proposed = np.vstack([points, new_point])
            
            elif action == 'delete' and len(points) > num_points * 0.5:
                # Delete random point
                idx = np.random.randint(len(points))
                proposed = np.delete(points, idx, axis=0)
            
            elif action == 'move' and len(points) > 0:
                # Move random point
                idx = np.random.randint(len(points))
                proposed = points.copy()
                proposed[idx] = self._generate_uniform_point()
            
            else:
                continue
            
            # Accept/reject based on Strauss energy
            if self._accept_proposal(points, proposed):
                points = proposed
        
        logger.info(f"Generated {len(points)} points with Strauss process")
        return points
    
    def _generate_uniform_point(self) -> np.ndarray:
        """Generate single uniform point in sphere"""
        while True:
            point = np.random.uniform(-self.radius, self.radius, 3)
            if np.linalg.norm(point) <= self.radius:
                return point
    
    def _compute_energy(self, points: np.ndarray) -> float:
        """Compute Strauss process energy"""
        if len(points) == 0:
            return 0.0
        
        # Count pairs within interaction radius
        dists = distance_matrix(points, points)
        np.fill_diagonal(dists, np.inf)
        num_close_pairs = np.sum(dists < self.interaction_radius) / 2
        
        # Strauss energy
        energy = -len(points) * np.log(self.intensity) - num_close_pairs * np.log(self.beta)
        
        return energy
    
    def _accept_proposal(self, current: np.ndarray, proposed: np.ndarray) -> bool:
        """Metropolis-Hastings acceptance"""
        energy_current = self._compute_energy(current)
        energy_proposed = self._compute_energy(proposed)
        
        # Accept if energy decreases, or with probability exp(-ΔE)
        if energy_proposed < energy_current:
            return True
        else:
            prob = np.exp(-(energy_proposed - energy_current))
            return np.random.uniform() < prob


def create_gradient_intensity(
    gradient_type: str = 'radial',
    center: np.ndarray = np.array([0, 0, 0]),
    max_intensity: float = 200.0,
    min_intensity: float = 50.0,
) -> Callable:
    """
    Create intensity function with gradient
    
    Args:
        gradient_type: 'radial' (center-to-periphery) or 'linear'
        center: Center point for radial gradient
        max_intensity: Maximum intensity
        min_intensity: Minimum intensity
    
    Returns:
        Intensity function λ(x, y, z)
    """
    if gradient_type == 'radial':
        def intensity_func(x, y, z):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            # Linear decrease from center
            return max(min_intensity, max_intensity - dist / 100.0 * (max_intensity - min_intensity))
        return intensity_func
    
    elif gradient_type == 'linear':
        def intensity_func(x, y, z):
            # Gradient along z-axis
            normalized_z = (z + 100) / 200  # Assuming radius=100
            return min_intensity + normalized_z * (max_intensity - min_intensity)
        return intensity_func
    
    else:
        raise ValueError(f"Unknown gradient type: {gradient_type}")


if __name__ == "__main__":
    # Test point processes
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    logging.basicConfig(level=logging.INFO)
    
    processes = {
        'Poisson': PoissonProcess(intensity=50),
        'Matérn': MaternClusterProcess(parent_intensity=5, cluster_radius=15),
        'Strauss': StraussProcess(intensity=50, interaction_radius=15, beta=0.1),
    }
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, (name, process) in enumerate(processes.items(), 1):
        points = process.generate(num_points=200)
        
        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.6)
        ax.set_title(f"{name} Process (n={len(points)})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('point_processes.png', dpi=150)
    print("Saved visualization to point_processes.png")

