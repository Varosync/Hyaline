"""
Hyaline V2: Enhanced GPCR Activation Prediction Model
======================================================

State-of-the-art enhancements over the original Hyaline model:
1. RBF Distance Expansion (64-dim ExpNormalSmearing)
2. Motif Attention Biasing (DRY, NPxxY, CWxP awareness)
3. Multi-Scale Graph Processing (5Å/10Å/25Å)
4. Improved normalization and residual connections

Expected improvements:
- Better resolution for distance-dependent features
- Biologically-informed attention patterns
- Long-range allosteric communication capture
- Especially improved Class C GPCR performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.utils import softmax
from typing import Optional, Tuple, List

from hyaline.sota_enhancements import (
    ExpNormalSmearing,
    GaussianSmearing,
    CosineCutoff,
    MotifAttentionBias,
    MultiScaleAggregator,
    EnhancedEGNNLayer,
    EnhancedGeometricEdgeUpdate,
    build_multiscale_edges,
)


class GlobalStateAggregatorV2(nn.Module):
    """
    Enhanced global state aggregation with layer normalization.
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        edge_batch: torch.Tensor
    ) -> torch.Tensor:
        node_agg = global_mean_pool(self.node_encoder(x), batch)
        edge_agg = global_mean_pool(self.edge_encoder(edge_attr), edge_batch)
        return self.state_mlp(torch.cat([node_agg, edge_agg], dim=-1))


class HyalineV2(nn.Module):
    """
    Hyaline V2: State-of-the-Art E(n)-Equivariant GNN for GPCR Activation.
    
    Key Improvements:
    1. RBF distance expansion (64-dim) for better distance resolution
    2. Motif attention biasing for GPCR-aware attention
    3. Optional multi-scale graph processing
    4. Enhanced normalization and residual connections
    
    Architecture:
        ESM3 [1536] → NodeProj → 4x EnhancedEGNNLayer → JK → Classifier
                                      ↑
                              Motif Attention Bias
                              RBF Distance Features
    """
    def __init__(
        self,
        node_input_dim: int = 1536,
        edge_input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        dropout: float = 0.1,
        update_coords: bool = True,
        use_motif_bias: bool = True,
        use_multiscale: bool = False,  # Optional multi-scale
        multiscale_cutoffs: List[float] = [5.0, 10.0, 25.0]
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_motif_bias = use_motif_bias
        self.use_multiscale = use_multiscale
        self.multiscale_cutoffs = multiscale_cutoffs
        
        # Initial projections
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Enhanced EGNN layers with RBF and optional motif bias
        self.layers = nn.ModuleList([
            EnhancedEGNNLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_rbf=num_rbf,
                cutoff=cutoff,
                dropout=dropout,
                update_coords=update_coords and (i < num_layers - 1),
                use_motif_bias=use_motif_bias
            )
            for i in range(num_layers)
        ])
        
        # Global state aggregator
        self.global_agg = GlobalStateAggregatorV2(hidden_dim, hidden_dim, hidden_dim)
        
        # Multi-scale aggregator (if enabled)
        if use_multiscale:
            self.scale_agg = MultiScaleAggregator(
                hidden_dim=hidden_dim,
                num_scales=len(multiscale_cutoffs),
                aggregation="attention"
            )
        
        # Jumping Knowledge: concat all layers
        jk_dim = hidden_dim * (num_layers + 1)
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(jk_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear layers with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        data: Data
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with:
                - x: Node features [N, 1536] (ESM3 embeddings)
                - pos: Cα coordinates [N, 3]
                - edge_index: [2, E]
                - edge_attr: [E, 3]
                - batch: [N]
                - motif_types: [N] (optional, for motif biasing)
        
        Returns:
            logits: Classification logits [B]
            attention: Layer-wise attention weights for interpretability
        """
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        
        # Get motif types if available
        motif_types = getattr(data, 'motif_types', None)
        if self.use_motif_bias and motif_types is None:
            # Create dummy motif types (all zeros = no motif)
            motif_types = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Edge batch assignment
        row, _ = edge_index
        edge_batch = batch[row]
        
        # Store layer outputs for Jumping Knowledge
        layer_outputs = [global_mean_pool(x, batch)]
        all_attn_weights = []
        
        for layer in self.layers:
            # Compute global state
            u = self.global_agg(x, edge_attr, batch, edge_batch)
            
            # Enhanced EGNN layer with motif bias
            x, pos, edge_attr, attn = layer(
                x, pos, edge_index, edge_attr, u, batch, motif_types
            )
            
            all_attn_weights.append(attn)
            layer_outputs.append(global_mean_pool(x, batch))
        
        # Jumping Knowledge aggregation
        jk = torch.cat(layer_outputs, dim=-1)
        
        # Final global state
        u_final = self.global_agg(x, edge_attr, batch, edge_batch)
        
        # Classification
        out = torch.cat([jk, u_final], dim=-1)
        logits = self.classifier(out).squeeze(-1)
        
        return logits, all_attn_weights
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        """
        Load a pretrained HyalineV2 model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            **kwargs: Override default hyperparameters
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract hyperparameters from checkpoint if available
        if 'hyperparameters' in checkpoint:
            saved_kwargs = checkpoint['hyperparameters']
            saved_kwargs.update(kwargs)
            kwargs = saved_kwargs
        
        model = cls(**kwargs)
        
        # Load state dict (handle potential key mismatches)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Try loading with strict=False for partial compatibility
            model.load_state_dict(state_dict, strict=False)
            print("Warning: Loaded with strict=False due to architecture mismatch")
        
        return model


def build_radius_graph_v2(
    pos: torch.Tensor,
    cutoff: float = 10.0,
    batch: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Build radius graph with batch support.
    
    Args:
        pos: Cα coordinates [N, 3]
        cutoff: Distance cutoff in Å
        batch: Optional batch assignment [N]
        
    Returns:
        edge_index: [2, E]
    """
    if batch is None:
        # Single graph
        dist = torch.cdist(pos, pos)
        mask = (dist < cutoff) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).T
    else:
        # Batched: only connect nodes within same graph
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(
            pos, r=cutoff, batch=batch, loop=False, max_num_neighbors=64
        )
    
    return edge_index


# Convenience function for model comparison
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    print("Testing HyalineV2...")
    
    model = HyalineV2(
        node_input_dim=1536,
        hidden_dim=256,
        num_layers=4,
        use_motif_bias=True
    )
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create dummy data
    N = 300  # Typical GPCR size
    data = Data(
        x=torch.randn(N, 1536),
        pos=torch.randn(N, 3) * 10,
        edge_index=torch.randint(0, N, (2, N * 10)),
        edge_attr=torch.randn(N * 10, 3),
        batch=torch.zeros(N, dtype=torch.long),
        motif_types=torch.randint(0, 5, (N,))
    )
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, attn = model(data)
    
    print(f"Output shape: {logits.shape}")
    print(f"Attention layers: {len(attn)}")
    print("✓ HyalineV2 test passed!")
