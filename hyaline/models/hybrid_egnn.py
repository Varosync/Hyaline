"""
Hybrid EGNN for Cryptic Pocket Detection
=========================================

E(n) Equivariant Graph Neural Network with classical MD features.

Key Innovation:
- Uses EGNN message passing for geometric learning
- Incorporates classical MD features (DCC, MI, RMSF) as inputs
- Edge features encode dynamical correlations, not just geometry
- Maintains SE(3) equivariance for coordinate updates

This is the scientifically grounded alternative to the SNN approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional, Dict


class EGNNLayer(MessagePassing):
    """
    E(n) Equivariant Graph Neural Network Layer.
    
    Based on Satorras et al. (2021) "E(n) Equivariant Graph Neural Networks"
    
    Key properties:
    - Invariant to rotations and translations
    - Updates both features and coordinates
    - Edge features can encode arbitrary pairwise information
    
    Args:
        hidden_dim: Hidden dimension
        edge_dim: Dimension of edge features
        update_coords: Whether to update coordinates
        normalize: Use LayerNorm
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 3,
        update_coords: bool = True,
        normalize: bool = True
    ):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords
        
        # Edge MLP: transforms edge features + node features + distance
        # Input: [h_i || h_j || edge_attr || distance]
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate update MLP (outputs scalar weight)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        
        # Normalization
        self.norm = nn.LayerNorm(hidden_dim) if normalize else nn.Identity()
    
    def forward(
        self,
        h: Tensor,              # [N, hidden_dim] node features
        pos: Tensor,            # [N, 3] coordinates
        edge_index: Tensor,     # [2, E] edge indices
        edge_attr: Tensor       # [E, edge_dim] edge features
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Returns:
            h_new: [N, hidden_dim] updated node features
            pos_new: [N, 3] updated coordinates
        """
        # Compute pairwise distances
        row, col = edge_index
        coord_diff = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(coord_diff, dim=-1, keepdim=True)  # [E, 1]
        
        # Message passing
        h_new = self.propagate(
            edge_index,
            h=h,
            edge_attr=edge_attr,
            dist=dist,
            coord_diff=coord_diff
        )
        
        # Skip connection + norm
        h_new = self.norm(h + h_new)
        
        # Coordinate update
        if self.update_coords:
            pos_new = self._update_coords(h, pos, edge_index, coord_diff, edge_attr, dist)
        else:
            pos_new = pos
        
        return h_new, pos_new
    
    def message(
        self,
        h_i: Tensor,
        h_j: Tensor, 
        edge_attr: Tensor,
        dist: Tensor,
        coord_diff: Tensor
    ) -> Tensor:
        """Compute messages."""
        # Concatenate all inputs
        msg_input = torch.cat([h_i, h_j, edge_attr, dist], dim=-1)
        return self.edge_mlp(msg_input)
    
    def update(self, aggr_out: Tensor, h: Tensor) -> Tensor:
        """Update node features."""
        return self.node_mlp(torch.cat([h, aggr_out], dim=-1))
    
    def _update_coords(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        coord_diff: Tensor,
        edge_attr: Tensor,
        dist: Tensor
    ) -> Tensor:
        """Update coordinates equivariantly."""
        row, col = edge_index
        
        # Compute edge weights
        msg_input = torch.cat([h[row], h[col], edge_attr, dist], dim=-1)
        edge_feat = self.edge_mlp(msg_input)
        coord_weights = self.coord_mlp(edge_feat)  # [E, 1]
        
        # Normalize by distance to avoid numerical issues
        norm_diff = coord_diff / (dist + 1e-8)
        
        # Weighted coordinate updates
        coord_updates = coord_weights * norm_diff
        
        # Aggregate
        pos_new = pos.clone()
        pos_new.index_add_(0, row, coord_updates)
        
        return pos_new


class HybridEGNN(nn.Module):
    """
    Hybrid EGNN for Cryptic Pocket Detection.
    
    Combines:
    1. EGNN message passing (SE(3) equivariant)
    2. Classical MD features (DCC, MI, contact freq) as edge features
    3. Per-residue features (RMSF, PCA) as node features
    
    The key insight is that cryptic pockets involve coordinated motion,
    which DCC/MI directly measure. The EGNN learns to combine these
    features with geometric information.
    
    Args:
        node_input_dim: Dimension of input node features
        edge_input_dim: Dimension of input edge features  
        hidden_dim: Hidden dimension
        num_layers: Number of EGNN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        node_input_dim: int = 2,    # RMSF, PCA contribution
        edge_input_dim: int = 3,    # DCC, MI, contact_freq
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge embedding (preserves original features)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim + 1, hidden_dim // 4),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, edge_input_dim)  # Keep edge_dim small
        )
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_input_dim,
                update_coords=(i < num_layers - 1),  # Don't update on last layer
                normalize=True
            )
            for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Pocket prediction head
        self.pocket_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Druggability prediction (optional, per pocket)
        self.druggability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        node_features: Tensor,      # [N, node_input_dim]
        pos: Tensor,                # [N, 3] coordinates
        edge_index: Tensor,         # [2, E]
        edge_features: Tensor,      # [E, edge_input_dim]
        batch: Optional[Tensor] = None  # [N] batch assignment
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            node_features: Per-residue features (RMSF, PCA contribution)
            pos: Cα coordinates
            edge_index: Graph connectivity
            edge_features: Per-edge features (DCC, MI, contact_freq)
            batch: Batch assignment for batched processing
            
        Returns:
            Dictionary with:
                - pocket_prob: [N] probability of being in cryptic pocket
                - druggability: [N] druggability score
                - final_coords: [N, 3] updated coordinates (if applicable)
        """
        # Encode node features
        h = self.node_encoder(node_features)
        
        # Add distance to edge features and encode
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=-1, keepdim=True)
        edge_input = torch.cat([edge_features, dist], dim=-1)
        edge_attr = self.edge_encoder(edge_input)
        
        # EGNN message passing
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)
            h = self.dropout(h)
        
        # Predictions
        pocket_prob = self.pocket_head(h).squeeze(-1)  # [N]
        druggability = self.druggability_head(h).squeeze(-1)  # [N]
        
        return {
            'pocket_prob': pocket_prob,
            'druggability': druggability,
            'node_embeddings': h,
            'final_coords': pos
        }
    
    @torch.no_grad()
    def predict(
        self,
        node_features: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        threshold: float = 0.5
    ) -> Dict[str, Tensor]:
        """
        Inference with thresholding.
        
        Returns:
            Dictionary with predictions and binary labels
        """
        self.eval()
        outputs = self.forward(node_features, pos, edge_index, edge_features)
        
        outputs['pocket_binary'] = (outputs['pocket_prob'] > threshold).long()
        outputs['druggable_binary'] = (outputs['druggability'] > threshold).long()
        
        return outputs


class HybridEGNNWithAttention(nn.Module):
    """
    Hybrid EGNN with temporal attention for multi-frame analysis.
    
    When we have multiple frames from a trajectory, this model:
    1. Processes each frame with EGNN
    2. Attends over frames to find consistent patterns
    3. Aggregates for final prediction
    
    This captures the intuition that cryptic pockets are revealed
    through dynamics - we want patterns that appear consistently
    across conformational changes.
    """
    
    def __init__(
        self,
        node_input_dim: int = 2,
        edge_input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Per-frame EGNN
        self.frame_egnn = HybridEGNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Temporal attention over frames
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final prediction heads
        self.pocket_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        frames_node_features: Tensor,    # [F, N, node_dim]
        frames_pos: Tensor,              # [F, N, 3]
        edge_index: Tensor,              # [2, E] (same for all frames)
        frames_edge_features: Tensor,    # [F, E, edge_dim]
    ) -> Dict[str, Tensor]:
        """
        Process multiple frames with temporal attention.
        
        Args:
            frames_*: Features for F frames
            edge_index: Shared graph topology
            
        Returns:
            Aggregated predictions
        """
        F, N, _ = frames_node_features.shape
        
        # Process each frame
        frame_embeddings = []
        for f in range(F):
            outputs = self.frame_egnn(
                frames_node_features[f],
                frames_pos[f],
                edge_index,
                frames_edge_features[f]
            )
            frame_embeddings.append(outputs['node_embeddings'])
        
        # Stack: [F, N, hidden_dim]
        frame_stack = torch.stack(frame_embeddings, dim=0)
        
        # Transpose for attention: [N, F, hidden_dim]
        frame_stack = frame_stack.transpose(0, 1)
        
        # Self-attention over frames for each residue
        attended, _ = self.temporal_attention(
            frame_stack, frame_stack, frame_stack
        )  # [N, F, hidden_dim]
        
        # Aggregate over frames (mean)
        aggregated = attended.mean(dim=1)  # [N, hidden_dim]
        
        # Final predictions
        pocket_prob = self.pocket_head(aggregated).squeeze(-1)
        
        return {
            'pocket_prob': pocket_prob,
            'frame_embeddings': frame_stack  # [N, F, hidden_dim]
        }


def build_graph_from_coords(
    coords: Tensor,
    cutoff: float = 10.0,
    include_sequential: bool = True
) -> Tensor:
    """
    Build graph connectivity from coordinates.
    
    Args:
        coords: [N, 3] Cα coordinates
        cutoff: Distance cutoff for edges
        include_sequential: Always connect sequential residues
        
    Returns:
        edge_index: [2, E] edge indices
    """
    N = coords.shape[0]
    
    # Pairwise distances
    dist = torch.cdist(coords, coords)  # [N, N]
    
    # Edges based on distance
    adj = dist < cutoff
    
    # Include sequential neighbors
    if include_sequential:
        for i in range(N - 1):
            adj[i, i + 1] = True
            adj[i + 1, i] = True
    
    # Remove self-loops
    adj.fill_diagonal_(False)
    
    # Convert to edge_index
    edge_index = adj.nonzero().t().contiguous()  # [2, E]
    
    return edge_index


if __name__ == "__main__":
    # Quick test
    print("Testing HybridEGNN...")
    
    torch.manual_seed(42)
    
    # Create synthetic data
    N = 50  # residues
    node_dim = 2
    edge_dim = 3
    
    node_features = torch.randn(N, node_dim)
    pos = torch.cumsum(torch.randn(N, 3) * 3.8, dim=0)  # Chain-like
    
    # Build graph
    edge_index = build_graph_from_coords(pos, cutoff=10.0)
    E = edge_index.shape[1]
    print(f"Graph: {N} nodes, {E} edges")
    
    # Random edge features (would be DCC, MI, contact_freq in practice)
    edge_features = torch.randn(E, edge_dim)
    
    # Create model
    model = HybridEGNN(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=128,
        num_layers=3
    )
    
    # Forward pass
    outputs = model(node_features, pos, edge_index, edge_features)
    
    print(f"Pocket prob shape: {outputs['pocket_prob'].shape}")
    print(f"Pocket prob range: [{outputs['pocket_prob'].min():.3f}, {outputs['pocket_prob'].max():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test backward pass
    loss = outputs['pocket_prob'].sum()
    loss.backward()
    print("✓ Backward pass successful!")
    
    print("\n✓ HybridEGNN test complete!")
