"""
TF-Modulator: SE(3) Equivariant Model for TF-DNA Pocket Detection
=================================================================

This module implements the TF-Modulator architecture - an SE(3) equivariant
graph neural network designed to predict druggable pockets in transcription
factor-DNA-coactivator ternary complexes.

Key features:
- 6 EGNN layers for deep geometric reasoning
- Two prediction heads: pocket detection + druggability scoring
- Interface-aware edge features for TF-DNA boundaries
- Optional classical MD feature integration

Architecture follows the system specification for oncology drug discovery
targeting undruggable transcription factors (MYC, KRAS, etc.)
"""

from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TFModulatorConfig:
    """Configuration for TF-Modulator architecture."""
    
    # Node features
    element_types: int = 5      # C, N, O, S, P
    hybridization_states: int = 4  # sp, sp2, sp3, other
    residue_types: int = 20     # 20 amino acids
    include_charge: bool = True
    include_dynamics: bool = True  # RMSF, PCA from MD
    
    # Edge features
    include_bond_type: bool = True  # covalent, h-bond, vdw
    include_interface_flag: bool = True  # crosses TF-DNA boundary
    include_dcc: bool = True  # dynamic cross-correlation
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 6  # Per specification
    dropout: float = 0.1
    
    # Cutoffs
    protein_cutoff: float = 8.0   # Angstroms for protein-protein edges
    dna_cutoff: float = 12.0      # Angstroms for protein-DNA edges
    
    @property
    def node_input_dim(self) -> int:
        """Calculate node feature dimension."""
        dim = self.element_types  # one-hot element
        dim += self.hybridization_states  # one-hot hybridization
        dim += self.residue_types  # one-hot residue
        if self.include_charge:
            dim += 1  # partial charge
        if self.include_dynamics:
            dim += 2  # RMSF + PCA contribution
        return dim
    
    @property
    def edge_input_dim(self) -> int:
        """Calculate edge feature dimension."""
        dim = 1  # distance
        if self.include_bond_type:
            dim += 3  # covalent, h-bond, vdw
        if self.include_interface_flag:
            dim += 1  # crosses TF-DNA boundary
        if self.include_dcc:
            dim += 2  # DCC + MI
        return dim


class EGNNLayer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network Layer.
    
    Updates both node features and coordinates while respecting SE(3) symmetry.
    The coordinate update is equivariant (rotates with input) while the
    feature update is invariant.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
        update_coords: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords
        self.residual = residual
        
        # Edge MLP: computes messages from node pairs + edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Node MLP: aggregates messages to update node features
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Coordinate MLP: computes coordinate updates (equivariant)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        h: torch.Tensor,           # [N, hidden_dim] node features
        pos: torch.Tensor,         # [N, 3] coordinates
        edge_index: torch.Tensor,  # [2, E] edge indices
        edge_attr: torch.Tensor,   # [E, edge_dim] edge features
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Updated (h, pos) tuple
        """
        row, col = edge_index
        
        # Compute distances (invariant scalar)
        rel_pos = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        # Edge messages
        edge_input = torch.cat([
            h[row],
            h[col],
            edge_attr,
            dist,
        ], dim=-1)
        
        messages = self.edge_mlp(edge_input)  # [E, hidden_dim]
        
        # Aggregate messages to nodes (handle mixed precision)
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, messages.to(agg.dtype))
        
        # Normalize by node degree to prevent gradient explosion
        degree = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        degree.index_add_(0, row, torch.ones(row.size(0), device=h.device, dtype=h.dtype))
        degree = degree.clamp(min=1).unsqueeze(-1)  # [N, 1]
        agg = agg / degree
        
        # Update node features (invariant)
        node_input = torch.cat([h, agg], dim=-1)
        h_update = self.node_mlp(node_input)
        
        if self.residual:
            h = h + h_update
        else:
            h = h_update
        
        h = self.layer_norm(h)
        
        # Update coordinates (equivariant)
        if self.update_coords:
            # Compute scalar weights
            coord_weights = self.coord_mlp(messages)  # [E, 1]
            
            # Direction-weighted update
            coord_update = coord_weights * rel_pos  # [E, 3]
            
            # Aggregate coordinate updates
            pos_agg = torch.zeros_like(pos)
            pos_agg.index_add_(0, row, coord_update.to(pos_agg.dtype))
            
            # Normalize by number of neighbors
            neighbor_count = torch.zeros(pos.size(0), device=pos.device, dtype=pos.dtype)
            neighbor_count.index_add_(0, row, torch.ones(row.size(0), device=pos.device, dtype=pos.dtype))
            neighbor_count = neighbor_count.clamp(min=1).unsqueeze(-1)
            
            pos = pos + pos_agg / neighbor_count
        
        return h, pos


class PocketHead(nn.Module):
    """Per-residue pocket prediction head."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict pocket probability per residue."""
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class DruggabilityHead(nn.Module):
    """
    Druggability scoring head.
    
    Predicts a continuous druggability score for each residue/pocket
    based on geometric and physicochemical properties.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict druggability score per residue (0-1)."""
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class TFModulator(nn.Module):
    """
    TF-Modulator: SE(3) Equivariant Model for TF-DNA Pocket Detection.
    
    This is the main model class implementing the full architecture:
    - Node/edge encoders for molecular features
    - 6 EGNN layers for geometric message passing
    - Two prediction heads: pocket detection + druggability scoring
    
    Args:
        config: TFModulatorConfig with architecture parameters
    """
    
    def __init__(self, config: Optional[TFModulatorConfig] = None):
        super().__init__()
        
        if config is None:
            config = TFModulatorConfig()
        
        self.config = config
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.edge_input_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
        )
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=config.hidden_dim,
                edge_dim=config.hidden_dim // 2,
                dropout=config.dropout,
                update_coords=(i < config.num_layers - 1),  # Don't update coords on last layer
                residual=True,
            )
            for i in range(config.num_layers)
        ])
        
        # Prediction heads
        self.pocket_head = PocketHead(config.hidden_dim, config.dropout)
        self.druggability_head = DruggabilityHead(config.hidden_dim, config.dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,   # [N, node_dim]
        pos: torch.Tensor,             # [N, 3]
        edge_index: torch.Tensor,      # [2, E]
        edge_features: torch.Tensor,   # [E, edge_dim]
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            node_features: Per-atom/residue features
            pos: 3D coordinates
            edge_index: Graph connectivity
            edge_features: Per-edge features
            return_embeddings: If True, also return learned representations
            
        Returns:
            Dictionary with:
            - pocket_prob: [N] pocket probability per residue
            - druggability: [N] druggability score per residue
            - pos_final: [N, 3] updated coordinates (optional)
            - embeddings: [N, hidden_dim] learned representations (optional)
        """
        # Encode inputs
        h = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        
        # Store initial positions
        pos_init = pos.clone()
        
        # Message passing
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)
        
        # Predictions
        pocket_prob = self.pocket_head(h)
        druggability = self.druggability_head(h)
        
        output = {
            'pocket_prob': pocket_prob,
            'druggability': druggability,
            'pos_final': pos,
        }
        
        if return_embeddings:
            output['embeddings'] = h
        
        return output
    
    @classmethod
    def from_config(cls, config_path: str) -> 'TFModulator':
        """Load model from config file."""
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = TFModulatorConfig(**config_dict)
        return cls(config)


def build_tf_graph(
    pos: torch.Tensor,
    is_dna: torch.Tensor,
    protein_cutoff: float = 8.0,
    dna_cutoff: float = 12.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build graph with interface-aware edge features.
    
    Args:
        pos: [N, 3] atomic coordinates
        is_dna: [N] boolean mask for DNA atoms
        protein_cutoff: Distance cutoff for protein-protein edges
        dna_cutoff: Distance cutoff for protein-DNA edges
        
    Returns:
        edge_index: [2, E] graph connectivity
        interface_flag: [E] 1 if edge crosses TF-DNA boundary, 0 otherwise
    """
    N = pos.size(0)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(pos, pos)
    
    # Different cutoffs for different edge types
    is_dna_f = is_dna.float().unsqueeze(1)
    
    # Edge crosses interface if one node is DNA and other is protein
    crosses_interface = (is_dna_f != is_dna_f.T).float()
    
    # Use larger cutoff for interface edges
    cutoff_matrix = torch.where(
        crosses_interface.bool(),
        torch.full_like(dist_matrix, dna_cutoff),
        torch.full_like(dist_matrix, protein_cutoff),
    )
    
    # Build adjacency
    adj = (dist_matrix < cutoff_matrix) & (dist_matrix > 0)
    
    # Convert to edge index
    edge_index = adj.nonzero(as_tuple=False).T
    
    # Get interface flags for edges
    row, col = edge_index
    interface_flag = crosses_interface[row, col]
    
    return edge_index, interface_flag


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Demonstration
    print("TF-Modulator Architecture")
    print("=" * 50)
    
    # Default config
    config = TFModulatorConfig()
    print(f"\nConfiguration:")
    print(f"  Node input dim: {config.node_input_dim}")
    print(f"  Edge input dim: {config.edge_input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    
    # Create model
    model = TFModulator(config)
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")
    
    # Synthetic forward pass
    N = 100  # residues
    E = 500  # edges
    
    node_features = torch.randn(N, config.node_input_dim)
    pos = torch.randn(N, 3) * 10
    edge_index = torch.randint(0, N, (2, E))
    edge_features = torch.randn(E, config.edge_input_dim)
    
    output = model(node_features, pos, edge_index, edge_features)
    
    print(f"\nOutput shapes:")
    print(f"  pocket_prob: {output['pocket_prob'].shape}")
    print(f"  druggability: {output['druggability'].shape}")
    print(f"  pos_final: {output['pos_final'].shape}")
    
    print("\n✓ TF-Modulator ready!")
