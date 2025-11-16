"""
SOTA Enhancements for Hyaline
============================

This module contains state-of-the-art components for GPCR activation prediction:
1. RBF Distance Expansion (from TorchMD-Net)
2. Motif Attention Biasing (novel integration)
3. Multi-Scale Graph Construction

These enhancements address the limitations identified in the architectural analysis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict


# =============================================================================
# 1. RADIAL BASIS FUNCTION (RBF) EXPANSION
# =============================================================================

class CosineCutoff(nn.Module):
    """
    Smooth cosine cutoff function for distance-based interactions.
    
    Ensures interactions smoothly go to zero at the cutoff distance,
    avoiding discontinuities in energy/force predictions.
    """
    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 10.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi * (
                        2 * (distances - self.cutoff_lower) / 
                        (self.cutoff_upper - self.cutoff_lower) + 1.0
                    )
                ) + 1.0
            )
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class ExpNormalSmearing(nn.Module):
    """
    Exponential Normal Smearing RBF expansion.
    
    Transforms scalar distances into high-dimensional feature vectors
    using exponential Gaussian basis functions. This is the SOTA approach
    from TorchMD-Net for molecular GNNs.
    
    Args:
        cutoff_lower: Minimum distance (Å)
        cutoff_upper: Maximum distance (Å)  
        num_rbf: Number of RBF kernels
        trainable: Whether to learn RBF parameters
    """
    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        num_rbf: int = 64,
        trainable: bool = True
    ):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)
        
        # Initialize means and betas (inverse widths)
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)
    
    def _initial_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Start with exponentially spaced means
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas
    
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: Pairwise distances [E] or [E, 1]
            
        Returns:
            RBF features [E, num_rbf]
        """
        if dist.dim() == 2:
            dist = dist.squeeze(-1)
        
        # Apply exponential transform
        dist = dist.unsqueeze(-1)  # [E, 1]
        exp_dist = torch.exp(
            -self.alpha * (dist - self.cutoff_lower)
        )  # [E, 1]
        
        # Compute RBF values
        rbf = self.cutoff_fn(dist.squeeze(-1)).unsqueeze(-1) * torch.exp(
            -self.betas * (exp_dist - self.means) ** 2
        )  # [E, num_rbf]
        
        return rbf


class GaussianSmearing(nn.Module):
    """
    Standard Gaussian RBF expansion.
    
    Simpler alternative to ExpNormalSmearing, using evenly-spaced
    Gaussian kernels in distance space.
    """
    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        num_rbf: int = 64,
        trainable: bool = False
    ):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        
        # Evenly spaced means
        offset = torch.linspace(cutoff_lower, cutoff_upper, num_rbf)
        width = (cutoff_upper - cutoff_lower) / num_rbf
        coeff = -0.5 / (width ** 2)
        
        if trainable:
            self.register_parameter("offset", nn.Parameter(offset))
            self.register_parameter("coeff", nn.Parameter(torch.tensor(coeff)))
        else:
            self.register_buffer("offset", offset)
            self.register_buffer("coeff", torch.tensor(coeff))
    
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.dim() == 2:
            dist = dist.squeeze(-1)
        dist = dist.unsqueeze(-1)  # [E, 1]
        return torch.exp(self.coeff * (dist - self.offset) ** 2)


# =============================================================================
# 2. MOTIF ATTENTION BIASING
# =============================================================================

class MotifAttentionBias(nn.Module):
    """
    Injects GPCR domain knowledge into attention computation.
    
    Creates attention biases based on:
    1. Motif membership (DRY, NPxxY, CWxP residues get higher attention)
    2. Motif-motif interactions (edges between motif residues are emphasized)
    3. Learned per-motif importance weights
    
    This ensures the model focuses on biologically relevant regions
    for activation prediction.
    """
    def __init__(
        self,
        num_motif_types: int = 5,  # 0=none, 1=DRY, 2=NPxxY, 3=CWxP, 4=PIF
        hidden_dim: int = 256,
        temperature: float = 1.0
    ):
        super().__init__()
        self.num_motif_types = num_motif_types
        self.temperature = temperature
        
        # Learnable importance per motif type
        self.motif_importance = nn.Parameter(torch.zeros(num_motif_types))
        nn.init.constant_(self.motif_importance[0], 0.0)  # Non-motif baseline
        nn.init.constant_(self.motif_importance[1], 1.0)  # DRY (most important)
        nn.init.constant_(self.motif_importance[2], 0.8)  # NPxxY
        nn.init.constant_(self.motif_importance[3], 0.6)  # CWxP
        nn.init.constant_(self.motif_importance[4], 0.4)  # PIF
        
        # Cross-motif interaction bias
        self.cross_motif_bias = nn.Parameter(
            torch.zeros(num_motif_types, num_motif_types)
        )
        # Initialize: DRY-NPxxY and DRY-CWxP interactions are important
        self.cross_motif_bias.data[1, 2] = 1.0  # DRY → NPxxY
        self.cross_motif_bias.data[2, 1] = 1.0  # NPxxY → DRY
        self.cross_motif_bias.data[1, 3] = 0.8  # DRY → CWxP
        self.cross_motif_bias.data[3, 1] = 0.8  # CWxP → DRY
    
    def forward(
        self,
        motif_types: torch.Tensor,  # [N] motif type per node
        edge_index: torch.Tensor     # [2, E]
    ) -> torch.Tensor:
        """
        Compute edge-level attention bias.
        
        Returns:
            bias: [E] attention bias to add to logits
        """
        row, col = edge_index
        
        # Node-level importance
        node_importance_i = self.motif_importance[motif_types[row]]  # [E]
        node_importance_j = self.motif_importance[motif_types[col]]  # [E]
        
        # Cross-motif interaction
        type_i = motif_types[row]  # [E]
        type_j = motif_types[col]  # [E]
        cross_bias = self.cross_motif_bias[type_i, type_j]  # [E]
        
        # Combined bias (source + target + interaction)
        bias = (node_importance_i + node_importance_j + cross_bias) / self.temperature
        
        return bias


# =============================================================================
# 3. MULTI-SCALE GRAPH CONSTRUCTION
# =============================================================================

def build_multiscale_edges(
    pos: torch.Tensor,
    cutoffs: List[float] = [5.0, 10.0, 25.0],
    return_distances: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Build multi-scale radius graphs.
    
    Creates separate edge sets for different distance scales:
    - Fine (5Å): Side-chain contacts, local interactions
    - Medium (10Å): Tertiary structure, default GPCR scale
    - Coarse (25Å): Long-range allosteric communication
    
    Args:
        pos: Cα coordinates [N, 3]
        cutoffs: List of distance cutoffs
        return_distances: If True, include distance tensors
        
    Returns:
        Dictionary with edge_index and optional distances per scale
    """
    # Compute full pairwise distances once
    dist = torch.cdist(pos, pos)  # [N, N]
    N = pos.size(0)
    
    result = {}
    
    for i, cutoff in enumerate(cutoffs):
        scale_name = f"scale_{i}"
        
        # Find edges within cutoff (excluding self-loops)
        if i == 0:
            # Finest scale: only include edges up to this cutoff
            mask = (dist < cutoff) & (dist > 0)
        else:
            # Coarser scales: exclude finer edges (ring structure)
            prev_cutoff = cutoffs[i - 1]
            mask = (dist < cutoff) & (dist >= prev_cutoff)
        
        edge_index = mask.nonzero(as_tuple=False).T  # [2, E_i]
        result[f"{scale_name}_edge_index"] = edge_index
        
        if return_distances:
            row, col = edge_index
            result[f"{scale_name}_dist"] = dist[row, col]
    
    return result


class MultiScaleAggregator(nn.Module):
    """
    Aggregates information from multiple graph scales.
    
    Each scale captures different structural relationships:
    - Fine: Local hydrogen bonds, salt bridges
    - Medium: Secondary structure elements
    - Coarse: Domain-level communication
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_scales: int = 3,
        aggregation: str = "attention"
    ):
        super().__init__()
        self.num_scales = num_scales
        self.aggregation = aggregation
        
        if aggregation == "attention":
            # Learnable scale importance
            self.scale_attention = nn.Sequential(
                nn.Linear(hidden_dim * num_scales, num_scales),
                nn.Softmax(dim=-1)
            )
        elif aggregation == "concat":
            # Project concatenated scales back to hidden dim
            self.proj = nn.Linear(hidden_dim * num_scales, hidden_dim)
        elif aggregation == "sum":
            # Learnable scale weights
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
    
    def forward(self, scale_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate embeddings from multiple scales.
        
        Args:
            scale_embeddings: List of [N, hidden_dim] tensors per scale
            
        Returns:
            Aggregated embedding [N, hidden_dim]
        """
        stacked = torch.stack(scale_embeddings, dim=1)  # [N, num_scales, hidden_dim]
        
        if self.aggregation == "attention":
            # Compute attention over scales
            concat = torch.cat(scale_embeddings, dim=-1)  # [N, hidden_dim * num_scales]
            weights = self.scale_attention(concat)  # [N, num_scales]
            weights = weights.unsqueeze(-1)  # [N, num_scales, 1]
            return (stacked * weights).sum(dim=1)  # [N, hidden_dim]
        
        elif self.aggregation == "concat":
            concat = torch.cat(scale_embeddings, dim=-1)
            return self.proj(concat)
        
        elif self.aggregation == "sum":
            weights = F.softmax(self.scale_weights, dim=0)
            weights = weights.view(1, self.num_scales, 1)
            return (stacked * weights).sum(dim=1)
        
        else:
            # Simple mean
            return stacked.mean(dim=1)


# =============================================================================
# 4. ENHANCED EDGE UPDATE WITH RBF
# =============================================================================

class EnhancedGeometricEdgeUpdate(nn.Module):
    """
    Geometric edge update with RBF distance expansion.
    
    Replaces the simple dist² feature with rich RBF encoding,
    providing better resolution for distance-dependent predictions.
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        trainable_rbf: bool = True
    ):
        super().__init__()
        
        # RBF expansion
        self.rbf = ExpNormalSmearing(
            cutoff_lower=0.0,
            cutoff_upper=cutoff,
            num_rbf=num_rbf,
            trainable=trainable_rbf
        )
        
        # Updated input: h_i || h_j || RBF(dist) || edge_attr || u
        input_dim = node_dim * 2 + num_rbf + edge_dim + hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
    
    def forward(
        self,
        h_i: torch.Tensor,        # [E, node_dim]
        h_j: torch.Tensor,        # [E, node_dim]
        dist: torch.Tensor,       # [E] raw distances (not squared)
        edge_attr: torch.Tensor,  # [E, edge_dim]
        u: torch.Tensor           # [E, hidden_dim]
    ) -> torch.Tensor:
        # RBF expand distances
        rbf_feat = self.rbf(dist)  # [E, num_rbf]
        
        # Concatenate all features
        inputs = torch.cat([h_i, h_j, rbf_feat, edge_attr, u], dim=-1)
        return self.mlp(inputs)


# =============================================================================
# 5. ENHANCED EGNN LAYER
# =============================================================================

class EnhancedEGNNLayer(nn.Module):
    """
    Enhanced E(n)-Equivariant GNN Layer with SOTA features.
    
    Improvements over base EGNNLayer:
    1. RBF distance expansion instead of raw dist²
    2. Motif attention biasing for GPCR-aware attention
    3. Improved normalization and residual connections
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        dropout: float = 0.1,
        update_coords: bool = True,
        use_motif_bias: bool = True
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.update_coords = update_coords
        self.use_motif_bias = use_motif_bias
        
        # Enhanced edge update with RBF
        self.edge_update = EnhancedGeometricEdgeUpdate(
            node_dim, edge_dim, hidden_dim, num_rbf, cutoff
        )
        
        # Message MLP with RBF
        self.rbf = ExpNormalSmearing(0.0, cutoff, num_rbf, trainable=True)
        message_input_dim = node_dim * 2 + edge_dim + num_rbf
        
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention with optional motif bias
        self.attn_mlp = nn.Linear(hidden_dim, 1)
        
        if use_motif_bias:
            self.motif_bias = MotifAttentionBias(
                num_motif_types=5, hidden_dim=hidden_dim
            )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Coordinate update
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(node_dim * 2 + num_rbf, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
        
        self.norm = nn.LayerNorm(node_dim)
    
    def forward(
        self,
        x: torch.Tensor,           # [N, node_dim]
        pos: torch.Tensor,         # [N, 3]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, edge_dim]
        u: torch.Tensor,           # [B, hidden_dim]
        batch: torch.Tensor,       # [N]
        motif_types: Optional[torch.Tensor] = None  # [N] motif type IDs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from torch_geometric.utils import softmax
        
        row, col = edge_index
        N = x.size(0)
        
        # Compute pairwise vectors and distances
        x_diff = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(x_diff, dim=-1)  # [E]
        
        # Broadcast global state to edges
        u_edge = u[batch[row]]  # [E, hidden_dim]
        
        # Enhanced edge update with RBF
        edge_attr_new = self.edge_update(
            x[row], x[col], dist, edge_attr, u_edge
        )
        edge_attr_new = edge_attr + edge_attr_new  # Residual
        
        # Compute messages with RBF
        rbf_feat = self.rbf(dist)  # [E, num_rbf]
        msg_input = torch.cat([x[row], x[col], edge_attr_new, rbf_feat], dim=-1)
        messages = self.message_mlp(msg_input)  # [E, hidden_dim]
        
        # Attention with optional motif bias
        attn_logits = self.attn_mlp(messages).squeeze(-1)  # [E]
        
        if self.use_motif_bias and motif_types is not None:
            motif_bias = self.motif_bias(motif_types, edge_index)  # [E]
            attn_logits = attn_logits + motif_bias
        
        attn_weights = softmax(attn_logits, col, num_nodes=N)  # [E]
        
        # Message aggregation
        weighted_msg = messages * attn_weights.unsqueeze(-1)
        x_msg = torch.zeros(N, self.hidden_dim, device=x.device, dtype=x.dtype)
        x_msg.scatter_add_(0, col.unsqueeze(-1).expand(-1, self.hidden_dim), weighted_msg)
        
        # Node update
        x_new = self.norm(x + self.node_mlp(torch.cat([x, x_msg], dim=-1)))
        
        # Coordinate update
        if self.update_coords:
            coord_input = torch.cat([x[row], x[col], rbf_feat], dim=-1)
            coord_weights = self.coord_mlp(coord_input)  # [E, 1]
            coord_delta = coord_weights * x_diff  # [E, 3]
            
            pos_delta = torch.zeros_like(pos)
            pos_delta.scatter_add_(0, row.unsqueeze(-1).expand(-1, 3), coord_delta)
            pos_new = pos + pos_delta * 0.1
        else:
            pos_new = pos
        
        return x_new, pos_new, edge_attr_new, attn_weights


# =============================================================================
# UTILITY: Convert existing graphs to enhanced format
# =============================================================================

def add_motif_features_to_graph(data, sequence: str):
    """
    Add motif type features to a PyG Data object.
    
    Args:
        data: PyG Data object
        sequence: Amino acid sequence
        
    Returns:
        Updated data with motif_types attribute
    """
    from hyaline.motifs import get_motif_type_embedding
    
    motif_types = get_motif_type_embedding(sequence)
    
    # Ensure length matches
    if len(motif_types) != data.num_nodes:
        # Truncate or pad
        if len(motif_types) > data.num_nodes:
            motif_types = motif_types[:data.num_nodes]
        else:
            padding = torch.zeros(data.num_nodes - len(motif_types), dtype=torch.long)
            motif_types = torch.cat([motif_types, padding])
    
    data.motif_types = motif_types
    return data
