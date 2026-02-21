"""
Classical MD Trajectory Feature Extraction
==========================================

Proven methods for analyzing molecular dynamics trajectories.
These features capture coordinated residue motion that correlates
with cryptic pocket formation.

Features:
    - DCC: Dynamic Cross-Correlation matrix
    - MI: Mutual Information (nonlinear correlations)
    - RMSF: Root Mean Square Fluctuation per residue
    - Contact Frequency: Transient contact detection
    - PCA: Collective motion contributions

Reference methods used in MD analysis for decades.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class TrajectoryFeatures:
    """Container for extracted trajectory features."""
    
    # Node features [N, num_node_features]
    node_features: np.ndarray
    
    # Edge features [N, N, num_edge_features]
    edge_features: np.ndarray
    
    # Individual components (for interpretability)
    dcc: np.ndarray           # [N, N] Dynamic Cross-Correlation
    mi: np.ndarray            # [N, N] Mutual Information
    contact_freq: np.ndarray  # [N, N] Contact frequency
    rmsf: np.ndarray          # [N] Root Mean Square Fluctuation
    pca_contrib: np.ndarray   # [N] PCA contribution per residue
    
    def to_torch(self, device: torch.device = None) -> 'TrajectoryFeaturesTorch':
        """Convert to PyTorch tensors."""
        return TrajectoryFeaturesTorch(
            node_features=torch.from_numpy(self.node_features).float().to(device),
            edge_features=torch.from_numpy(self.edge_features).float().to(device),
            dcc=torch.from_numpy(self.dcc).float().to(device),
            mi=torch.from_numpy(self.mi).float().to(device),
            contact_freq=torch.from_numpy(self.contact_freq).float().to(device),
            rmsf=torch.from_numpy(self.rmsf).float().to(device),
            pca_contrib=torch.from_numpy(self.pca_contrib).float().to(device)
        )


@dataclass  
class TrajectoryFeaturesTorch:
    """PyTorch tensor version of trajectory features."""
    node_features: Tensor
    edge_features: Tensor
    dcc: Tensor
    mi: Tensor
    contact_freq: Tensor
    rmsf: Tensor
    pca_contrib: Tensor


class ClassicalFeatureExtractor:
    """
    Extract classical MD analysis features from trajectory data.
    
    These are proven methods used in structural biology for decades.
    The key insight is that cryptic pocket formation involves coordinated
    motion of multiple residues, which these metrics directly measure.
    
    Args:
        contact_threshold: Distance threshold for contact definition (Å)
        mi_bins: Number of bins for mutual information estimation
        pca_components: Number of PCA components to consider
    """
    
    def __init__(
        self,
        contact_threshold: float = 8.0,
        mi_bins: int = 20,
        pca_components: int = 10
    ):
        self.contact_threshold = contact_threshold
        self.mi_bins = mi_bins
        self.pca_components = pca_components
    
    def compute_dcc(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Dynamic Cross-Correlation matrix.
        
        DCC measures the correlation of fluctuations between residue pairs.
        High positive DCC = residues move together
        High negative DCC = residues move oppositely (anticorrelated)
        
        For cryptic pockets:
        - Residues forming the pocket often have high positive DCC
        - They move together to open/close the pocket
        
        Reference: Ichiye & Karplus (1991) Proteins
        
        Args:
            trajectory: (n_frames, n_residues, 3) Cα coordinates
            
        Returns:
            dcc: (n_residues, n_residues) correlation matrix in [-1, 1]
        """
        n_frames, n_residues, _ = trajectory.shape
        
        # Compute fluctuations from mean structure
        mean_coords = trajectory.mean(axis=0)  # (n_residues, 3)
        fluctuations = trajectory - mean_coords  # (n_frames, n_residues, 3)
        
        # Compute DCC matrix
        dcc = np.zeros((n_residues, n_residues))
        
        # Precompute norms for efficiency
        norms = np.sqrt(np.sum(fluctuations ** 2, axis=(0, 2)))  # (n_residues,)
        norms = np.where(norms > 0, norms, 1e-8)  # Avoid division by zero
        
        for i in range(n_residues):
            for j in range(i, n_residues):
                # Inner product of fluctuation vectors summed over frames
                corr = np.sum(fluctuations[:, i, :] * fluctuations[:, j, :])
                
                # Normalize by individual fluctuation magnitudes
                dcc[i, j] = corr / (norms[i] * norms[j])
                dcc[j, i] = dcc[i, j]
        
        return dcc
    
    def compute_dcc_fast(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Vectorized DCC computation for large trajectories.
        
        Same result as compute_dcc but ~10x faster for large systems.
        """
        n_frames, n_residues, _ = trajectory.shape
        
        # Fluctuations from mean
        mean_coords = trajectory.mean(axis=0)
        fluctuations = trajectory - mean_coords  # (F, N, 3)
        
        # Reshape for matrix multiplication: (N, F*3)
        flat = fluctuations.transpose(1, 0, 2).reshape(n_residues, -1)
        
        # Correlation matrix via dot product
        corr_matrix = flat @ flat.T  # (N, N)
        
        # Normalize
        norms = np.sqrt(np.diag(corr_matrix))
        norms = np.where(norms > 0, norms, 1e-8)
        dcc = corr_matrix / np.outer(norms, norms)
        
        return dcc
    
    def compute_mutual_information(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Mutual Information between residue motions.
        
        MI captures ALL statistical dependencies, including nonlinear ones
        that DCC misses. This is important because:
        - Cryptic pocket opening may involve nonlinear coupling
        - MI >= 0, with higher values indicating more dependence
        
        Reference: Lange & Grubmüller (2006) Proteins
        
        Args:
            trajectory: (n_frames, n_residues, 3) coordinates
            
        Returns:
            mi: (n_residues, n_residues) mutual information matrix
        """
        n_frames, n_residues, _ = trajectory.shape
        
        # Compute displacement magnitudes from mean
        mean_coords = trajectory.mean(axis=0)
        displacements = np.linalg.norm(trajectory - mean_coords, axis=2)  # (F, N)
        
        mi = np.zeros((n_residues, n_residues))
        
        for i in range(n_residues):
            for j in range(i, n_residues):
                mi_val = self._compute_mi_pair(
                    displacements[:, i], 
                    displacements[:, j]
                )
                mi[i, j] = mi[j, i] = mi_val
        
        return mi
    
    def _compute_mi_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI between two 1D arrays using histogram method."""
        # Joint histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=self.mi_bins)
        
        # Convert to probability
        pxy = hist_2d / hist_2d.sum()
        pxy = pxy + 1e-10  # Avoid log(0)
        
        # Marginals
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        
        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        with np.errstate(divide='ignore', invalid='ignore'):
            mi = np.sum(pxy * np.log(pxy / (px[:, None] * py[None, :])))
            mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        
        return max(0.0, mi)  # MI is non-negative
    
    def compute_contact_frequency(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Fraction of frames where residue pairs are in contact.
        
        Transient contacts are key for cryptic pockets:
        - Low contact frequency in apo state
        - High contact frequency when pocket is formed
        - Changes in contact frequency indicate pocket dynamics
        
        Args:
            trajectory: (n_frames, n_residues, 3) coordinates
            
        Returns:
            contact_freq: (n_residues, n_residues) in [0, 1]
        """
        n_frames, n_residues, _ = trajectory.shape
        contact_count = np.zeros((n_residues, n_residues))
        
        for frame in trajectory:
            # Pairwise distances
            diff = frame[:, None, :] - frame[None, :, :]  # (N, N, 3)
            distances = np.linalg.norm(diff, axis=2)  # (N, N)
            
            # Count contacts
            contacts = distances < self.contact_threshold
            contact_count += contacts.astype(float)
        
        contact_freq = contact_count / n_frames
        return contact_freq
    
    def compute_contact_frequency_fast(self, trajectory: np.ndarray) -> np.ndarray:
        """Vectorized contact frequency computation."""
        n_frames, n_residues, _ = trajectory.shape
        
        # All pairwise distances at once: (F, N, N)
        diff = trajectory[:, :, None, :] - trajectory[:, None, :, :]
        distances = np.linalg.norm(diff, axis=-1)  # (F, N, N)
        
        # Count frames in contact
        contacts = (distances < self.contact_threshold).astype(float)
        contact_freq = contacts.mean(axis=0)  # (N, N)
        
        return contact_freq
    
    def compute_rmsf(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Root Mean Square Fluctuation per residue.
        
        RMSF measures flexibility:
        - High RMSF = flexible region
        - Cryptic pockets often in flexible regions (they need to move)
        
        Reference: Standard MD analysis metric
        
        Args:
            trajectory: (n_frames, n_residues, 3) coordinates
            
        Returns:
            rmsf: (n_residues,) fluctuation per residue
        """
        mean_coords = trajectory.mean(axis=0)  # (N, 3)
        fluctuations = trajectory - mean_coords  # (F, N, 3)
        
        # RMS of fluctuations
        msf = np.mean(np.sum(fluctuations ** 2, axis=2), axis=0)  # (N,)
        rmsf = np.sqrt(msf)
        
        return rmsf
    
    def compute_pca_contribution(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Per-residue contribution to principal components.
        
        PCA identifies collective motions:
        - Top PCs capture large-scale conformational changes
        - Residues with high PC contribution are involved in these motions
        - Pocket-forming residues often contribute to specific PCs
        
        Reference: Amadei et al. (1993) Proteins (Essential Dynamics)
        
        Args:
            trajectory: (n_frames, n_residues, 3) coordinates
            
        Returns:
            pca_contrib: (n_residues,) weighted contribution to top PCs
        """
        n_frames, n_residues, _ = trajectory.shape
        
        # Flatten: (F, N*3)
        flat = trajectory.reshape(n_frames, -1)
        flat_centered = flat - flat.mean(axis=0)
        
        # SVD
        try:
            U, S, Vt = np.linalg.svd(flat_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback if SVD fails
            return np.ones(n_residues) / n_residues
        
        # Number of components to consider
        n_components = min(self.pca_components, len(S))
        
        # Reshape principal components: (n_components, n_residues, 3)
        components = Vt[:n_components].reshape(n_components, n_residues, 3)
        
        # Per-residue contribution = sum of squared coefficients
        residue_contrib = np.sum(components ** 2, axis=2)  # (n_components, n_residues)
        
        # Weight by eigenvalues (variance explained)
        eigenvalues = S[:n_components] ** 2
        weights = eigenvalues / eigenvalues.sum()
        
        # Weighted sum of contributions
        pca_contrib = (residue_contrib * weights[:, None]).sum(axis=0)  # (n_residues,)
        
        return pca_contrib
    
    def extract_all(
        self,
        trajectory: np.ndarray,
        fast: bool = True,
        verbose: bool = False
    ) -> TrajectoryFeatures:
        """
        Extract all features from a trajectory.
        
        Args:
            trajectory: (n_frames, n_residues, 3) Cα coordinates
            fast: Use vectorized implementations (faster but more memory)
            verbose: Print progress
            
        Returns:
            TrajectoryFeatures containing all extracted features
        """
        n_frames, n_residues, _ = trajectory.shape
        
        if verbose:
            print(f"Extracting features from trajectory: {n_frames} frames, {n_residues} residues")
        
        # DCC
        if verbose:
            print("  Computing DCC...")
        dcc = self.compute_dcc_fast(trajectory) if fast else self.compute_dcc(trajectory)
        
        # Mutual Information (always slow for now)
        if verbose:
            print("  Computing Mutual Information...")
        mi = self.compute_mutual_information(trajectory)
        
        # Contact frequency
        if verbose:
            print("  Computing Contact Frequency...")
        contact_freq = (
            self.compute_contact_frequency_fast(trajectory) if fast 
            else self.compute_contact_frequency(trajectory)
        )
        
        # RMSF
        if verbose:
            print("  Computing RMSF...")
        rmsf = self.compute_rmsf(trajectory)
        
        # PCA contribution
        if verbose:
            print("  Computing PCA contribution...")
        pca_contrib = self.compute_pca_contribution(trajectory)
        
        # Normalize features for neural network input
        rmsf_norm = (rmsf - rmsf.mean()) / (rmsf.std() + 1e-8)
        pca_norm = (pca_contrib - pca_contrib.mean()) / (pca_contrib.std() + 1e-8)
        
        # Combine into node and edge features
        node_features = np.stack([rmsf_norm, pca_norm], axis=-1)  # (N, 2)
        
        # Normalize edge features
        dcc_norm = dcc  # Already in [-1, 1]
        mi_norm = mi / (mi.max() + 1e-8)  # Normalize to [0, 1]
        cf_norm = contact_freq  # Already in [0, 1]
        
        edge_features = np.stack([dcc_norm, mi_norm, cf_norm], axis=-1)  # (N, N, 3)
        
        if verbose:
            print("  Done!")
        
        return TrajectoryFeatures(
            node_features=node_features,
            edge_features=edge_features,
            dcc=dcc,
            mi=mi,
            contact_freq=contact_freq,
            rmsf=rmsf,
            pca_contrib=pca_contrib
        )


class NormalModeGenerator:
    """
    Generate pseudo-trajectory from normal mode perturbations.
    
    Normal modes approximate low-frequency collective motions.
    Not as accurate as real MD, but:
    - Fast to compute
    - Captures large-scale motions
    - Sufficient for architecture validation
    
    Reference: Suhre & Sanejouand (2004) Nucleic Acids Research
    """
    
    def __init__(
        self,
        n_modes: int = 10,
        amplitude: float = 2.0,
        n_frames: int = 100
    ):
        """
        Args:
            n_modes: Number of normal modes to use
            amplitude: Maximum displacement amplitude (Å)
            n_frames: Number of frames to generate
        """
        self.n_modes = n_modes
        self.amplitude = amplitude
        self.n_frames = n_frames
    
    def generate(self, coords: np.ndarray) -> np.ndarray:
        """
        Generate pseudo-trajectory by perturbing along normal modes.
        
        Uses Elastic Network Model (ENM) approximation.
        
        Args:
            coords: (n_residues, 3) reference Cα coordinates
            
        Returns:
            trajectory: (n_frames, n_residues, 3) pseudo-trajectory
        """
        n_residues = len(coords)
        
        # Build Kirchhoff/Hessian matrix (simplified ENM)
        hessian = self._build_hessian(coords)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        
        # Skip first 6 modes (rigid body motion)
        mode_indices = range(6, min(6 + self.n_modes, len(eigenvalues)))
        
        # Generate trajectory by sampling along modes
        trajectory = np.zeros((self.n_frames, n_residues, 3))
        
        for f in range(self.n_frames):
            displacement = np.zeros((n_residues, 3))
            
            for mode_idx in mode_indices:
                # Random amplitude for this mode
                amp = np.random.uniform(-self.amplitude, self.amplitude)
                
                # Scale by 1/sqrt(eigenvalue) (stiffer modes, smaller displacement)
                scale = amp / np.sqrt(eigenvalues[mode_idx] + 1e-8)
                
                # Reshape mode vector to (n_residues, 3)
                mode = eigenvectors[:, mode_idx].reshape(n_residues, 3)
                
                displacement += scale * mode
            
            trajectory[f] = coords + displacement
        
        return trajectory
    
    def _build_hessian(
        self, 
        coords: np.ndarray,
        cutoff: float = 13.0,
        gamma: float = 1.0
    ) -> np.ndarray:
        """
        Build Hessian matrix using Gaussian Network Model.
        
        Simplified form for computational efficiency.
        """
        n_residues = len(coords)
        n_dof = n_residues * 3
        
        # Initialize Hessian
        hessian = np.zeros((n_dof, n_dof))
        
        # Compute pairwise distances
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                r_ij = coords[j] - coords[i]
                dist = np.linalg.norm(r_ij)
                
                if dist < cutoff:
                    # Spring constant (Gaussian decay)
                    k = gamma * np.exp(-(dist / cutoff) ** 2)
                    
                    # Unit vector
                    u_ij = r_ij / (dist + 1e-8)
                    
                    # Outer product for 3x3 block
                    block = k * np.outer(u_ij, u_ij)
                    
                    # Fill Hessian blocks
                    ii, jj = i * 3, j * 3
                    hessian[ii:ii+3, jj:jj+3] = -block
                    hessian[jj:jj+3, ii:ii+3] = -block
                    hessian[ii:ii+3, ii:ii+3] += block
                    hessian[jj:jj+3, jj:jj+3] += block
        
        return hessian


if __name__ == "__main__":
    # Quick test
    print("Testing ClassicalFeatureExtractor...")
    
    # Create synthetic trajectory
    n_frames, n_residues = 50, 30
    np.random.seed(42)
    
    # Base structure (random coil for simplicity)
    base_coords = np.cumsum(np.random.randn(n_residues, 3) * 3.8, axis=0)
    
    # Add fluctuations
    trajectory = base_coords + np.random.randn(n_frames, n_residues, 3) * 0.5
    
    # Make residues 10-15 correlated (potential pocket region)
    shared_motion = np.random.randn(n_frames, 1, 3) * 1.0
    trajectory[:, 10:15, :] += shared_motion
    
    # Extract features
    extractor = ClassicalFeatureExtractor()
    features = extractor.extract_all(trajectory, verbose=True)
    
    print(f"\nNode features shape: {features.node_features.shape}")
    print(f"Edge features shape: {features.edge_features.shape}")
    print(f"DCC shape: {features.dcc.shape}")
    print(f"DCC range: [{features.dcc.min():.3f}, {features.dcc.max():.3f}]")
    
    # Check that correlated residues have high DCC
    pocket_dcc = features.dcc[10:15, 10:15].mean()
    other_dcc = features.dcc[:10, :10].mean()
    print(f"\nPocket region DCC: {pocket_dcc:.3f}")
    print(f"Other region DCC: {other_dcc:.3f}")
    
    print("\n✓ ClassicalFeatureExtractor test complete!")
    
    # Test normal mode generator
    print("\nTesting NormalModeGenerator...")
    nmg = NormalModeGenerator(n_frames=20)
    nm_trajectory = nmg.generate(base_coords)
    print(f"Generated trajectory shape: {nm_trajectory.shape}")
    
    print("\n✓ NormalModeGenerator test complete!")
