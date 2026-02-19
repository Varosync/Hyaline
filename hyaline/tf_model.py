"""
HyalineTF: Sequence-to-Function Model for Transcription Factors
================================================================

Maps TF sequence (and optional structural priors) to functionally
meaningful outputs:
  1. TF function class  (activator / repressor / dual)
  2. DNA-binding affinity score  (continuous)
  3. Downstream regulatory impact score  (continuous)

Technical approach
------------------
- Structure available  → E(n)-equivariant GNN (EnhancedEGNNLayer)
- Sequence only        → attention-weighted pooling over ESM embeddings
- TF domain-aware attention biasing for interpretability
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.utils import softmax as pyg_softmax
from typing import Dict, List, Optional, Tuple

from hyaline.sota_enhancements import EnhancedEGNNLayer
from hyaline.motifs import NUM_TF_DOMAIN_TYPES, TF_DOMAIN_TYPE_MAPPING

# Human-readable labels for TF function classes
TF_FUNCTION_CLASSES: List[str] = ['activator', 'repressor', 'dual']


# =============================================================================
# Sub-modules
# =============================================================================

# Default per-domain importance initialisation values (learnable from this baseline)
_DEFAULT_DOMAIN_IMPORTANCE = {
    'no_domain':      0.0,
    'zinc_finger':    1.2,  # strong sequence-specific DNA binding
    'leucine_zipper': 0.8,  # dimerization interface
    'basic_helix':    1.0,  # DNA-recognition basic region
    'homeodomain':    1.1,  # sequence-specific HTH binding
    'WRKY':           0.9,  # plant TF DNA-binding domain
    'ETS_helix':      0.7,  # ETS recognition helix
}


class TFDomainAttentionBias(nn.Module):
    """
    Domain-aware attention bias for transcription factors.

    Injects prior knowledge about TF domain importance into the attention
    computation (zinc fingers, leucine zippers, homeodomains, etc.).
    Implements the same interface as MotifAttentionBias so it can be
    dropped in as a replacement inside EnhancedEGNNLayer.
    """

    def __init__(
        self,
        num_domain_types: int = NUM_TF_DOMAIN_TYPES,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

        # Learnable per-domain importance (initialised from known biology)
        self.domain_importance = nn.Parameter(torch.zeros(num_domain_types))
        for idx, (domain, value) in enumerate(
            _DEFAULT_DOMAIN_IMPORTANCE.items()
        ):
            nn.init.constant_(self.domain_importance[idx], value)

        # Cross-domain interaction bias
        _LZ = TF_DOMAIN_TYPE_MAPPING['leucine_zipper']  # 2
        _BH = TF_DOMAIN_TYPE_MAPPING['basic_helix']     # 3
        self.cross_domain_bias = nn.Parameter(
            torch.zeros(num_domain_types, num_domain_types)
        )
        # Leucine zipper and basic helix co-occur in bZIP TFs
        self.cross_domain_bias.data[_LZ, _BH] = 0.5
        self.cross_domain_bias.data[_BH, _LZ] = 0.5

    def forward(
        self,
        domain_types: torch.Tensor,  # [N]
        edge_index: torch.Tensor,    # [2, E]
    ) -> torch.Tensor:
        """Returns per-edge attention bias [E]."""
        row, col = edge_index
        imp_i = self.domain_importance[domain_types[row]]
        imp_j = self.domain_importance[domain_types[col]]
        cross = self.cross_domain_bias[domain_types[row], domain_types[col]]
        return (imp_i + imp_j + cross) / self.temperature


class SequenceEncoder(nn.Module):
    """
    Sequence-only encoder using attention-weighted pooling over ESM embeddings.

    Works natively with PyG's variable-length batched format: no padding
    required.  An optional per-node bias (e.g. from domain importance) can
    be added to the attention logits before softmax.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.attn_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,                        # [N, input_dim]
        batch: torch.Tensor,                    # [N]
        node_bias: Optional[torch.Tensor] = None,  # [N]
    ) -> torch.Tensor:
        """Returns graph-level embeddings [B, hidden_dim]."""
        h = self.proj(x)                                       # [N, hidden_dim]
        attn_logits = self.attn_head(h).squeeze(-1)            # [N]
        if node_bias is not None:
            attn_logits = attn_logits + node_bias
        attn_weights = pyg_softmax(attn_logits, batch)         # [N]
        return global_add_pool(h * attn_weights.unsqueeze(-1), batch)  # [B, hidden_dim]


class TFOutputHead(nn.Module):
    """
    Multi-output prediction head for TF functional properties.

    Predicts:
      1. TF function class (activator / repressor / dual)  — classification
      2. DNA-binding affinity score                        — regression
      3. Downstream regulatory impact score                — regression
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_function_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        half = hidden_dim // 2

        self.function_head = nn.Sequential(
            nn.Linear(hidden_dim, half),
            nn.SiLU(),
            nn.Linear(half, num_function_classes),
        )
        self.binding_head = nn.Sequential(
            nn.Linear(hidden_dim, half),
            nn.SiLU(),
            nn.Linear(half, 1),
        )
        self.regulatory_head = nn.Sequential(
            nn.Linear(hidden_dim, half),
            nn.SiLU(),
            nn.Linear(half, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Graph-level features [B, input_dim]

        Returns:
            Dict with:
              'function_logits' [B, num_function_classes]
              'binding'         [B]  DNA-binding affinity logit
              'regulatory'      [B]  regulatory impact score
        """
        h = self.trunk(x)
        return {
            'function_logits': self.function_head(h),
            'binding': self.binding_head(h).squeeze(-1),
            'regulatory': self.regulatory_head(h).squeeze(-1),
        }


# =============================================================================
# Main model
# =============================================================================

class HyalineTF(nn.Module):
    """
    HyalineTF: Sequence-to-Function Model for Transcription Factors.

    Maps TF sequence (with optional structural priors) to:
      - TF function class  (activator / repressor / dual)
      - DNA-binding affinity score
      - Downstream regulatory impact score

    Two operating modes selected automatically from input data
    ──────────────────────────────────────────────────────────
    Structure-augmented  (data.pos and data.edge_index provided):
        ESM [dim] → NodeProj → N × EnhancedEGNNLayer → JK → TFOutputHead
                                       ↑
                               TFDomainAttentionBias
                               RBF Distance Features

    Sequence-only  (no structural data):
        ESM [dim] → SequenceEncoder (attention pooling) → TFOutputHead
                           ↑
                    domain importance bias

    Args:
        node_input_dim:       ESM embedding dimension (1280 for ESM2, 1536 for ESM3)
        edge_input_dim:       Edge feature dimension (default 3)
        hidden_dim:           Internal hidden dimension
        num_layers:           Number of EGNN layers (structure path)
        num_heads:            Attention heads per EGNN layer
        num_rbf:              RBF kernel count for distance encoding
        cutoff:               Radius cutoff in Å for the graph
        dropout:              Dropout probability
        update_coords:        Whether to update Cα coordinates in EGNN layers
        use_domain_bias:      Enable TF domain-aware attention biasing
        num_function_classes: Number of TF function classes (default 3)
    """

    def __init__(
        self,
        node_input_dim: int = 1280,
        edge_input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        dropout: float = 0.1,
        update_coords: bool = True,
        use_domain_bias: bool = True,
        num_function_classes: int = 3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_domain_bias = use_domain_bias
        self.num_function_classes = num_function_classes

        # ------------------------------------------------------------------
        # Shared domain-bias module (used in both paths)
        # ------------------------------------------------------------------
        if use_domain_bias:
            self.domain_attn_bias = TFDomainAttentionBias()

        # ------------------------------------------------------------------
        # Structure-augmented path: EGNN
        # ------------------------------------------------------------------
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.egnn_layers = nn.ModuleList([
            EnhancedEGNNLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_rbf=num_rbf,
                cutoff=cutoff,
                dropout=dropout,
                update_coords=update_coords and (i < num_layers - 1),
                use_motif_bias=use_domain_bias,
            )
            for i in range(num_layers)
        ])

        # Replace GPCR motif bias with TF domain bias in every EGNN layer,
        # reusing the shared domain_attn_bias instance for parameter sharing
        if use_domain_bias:
            for layer in self.egnn_layers:
                layer.motif_bias = self.domain_attn_bias

        # Global state aggregator (node + edge → graph summary fed to layers)
        self._node_agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self._edge_agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self._global_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Jumping Knowledge: concat all layer pooled outputs + final global state
        jk_dim = hidden_dim * (num_layers + 1) + hidden_dim
        self.struct_proj = nn.Linear(jk_dim, hidden_dim)

        # ------------------------------------------------------------------
        # Sequence-only path
        # ------------------------------------------------------------------
        self.seq_encoder = SequenceEncoder(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Shared output head
        # ------------------------------------------------------------------
        self.output_head = TFOutputHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_function_classes=num_function_classes,
            dropout=dropout,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _global_aggregate(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> torch.Tensor:
        node_agg = global_mean_pool(self._node_agg(x), batch)
        edge_agg = global_mean_pool(self._edge_agg(edge_attr), edge_batch)
        return self._global_mlp(torch.cat([node_agg, edge_agg], dim=-1))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass – selects structure-augmented or sequence-only path.

        Args:
            data: PyG Data object with:
                x            [N, node_input_dim]  ESM embeddings  (required)
                batch        [N]                  graph assignment (required)
                pos          [N, 3]               Cα coords       (optional)
                edge_index   [2, E]               graph edges     (required with pos)
                edge_attr    [E, edge_input_dim]  edge features   (required with pos)
                domain_types [N]                  TF domain IDs   (optional)

        Returns:
            Dict with:
              'function_logits'  [B, num_function_classes]
              'binding'          [B]   DNA-binding affinity logit
              'regulatory'       [B]   regulatory impact score
              'attention'        list of per-layer attention weights (struct path)
        """
        has_structure = (
            getattr(data, 'pos', None) is not None
            and getattr(data, 'edge_index', None) is not None
        )
        if has_structure:
            return self._forward_structure(data)
        return self._forward_sequence(data)

    def _forward_sequence(self, data: Data) -> Dict[str, torch.Tensor]:
        """Sequence-only path: attention-pooled ESM embeddings."""
        node_bias = None
        if self.use_domain_bias:
            domain_types = getattr(data, 'domain_types', None)
            if domain_types is None:
                domain_types = torch.zeros(
                    data.x.size(0), dtype=torch.long, device=data.x.device
                )
            node_bias = self.domain_attn_bias.domain_importance[domain_types]

        graph_emb = self.seq_encoder(data.x, data.batch, node_bias)
        outputs = self.output_head(graph_emb)
        outputs['attention'] = []
        return outputs

    def _forward_structure(self, data: Data) -> Dict[str, torch.Tensor]:
        """Structure-augmented path: EGNN with TF domain attention bias."""
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch

        domain_types = getattr(data, 'domain_types', None)
        if self.use_domain_bias and domain_types is None:
            domain_types = torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )

        row, _ = edge_index
        edge_batch = batch[row]

        layer_outputs = [global_mean_pool(x, batch)]
        all_attn: List[torch.Tensor] = []

        for layer in self.egnn_layers:
            u = self._global_aggregate(x, edge_attr, batch, edge_batch)
            x, pos, edge_attr, attn = layer(
                x, pos, edge_index, edge_attr, u, batch, domain_types
            )
            all_attn.append(attn)
            layer_outputs.append(global_mean_pool(x, batch))

        jk = torch.cat(layer_outputs, dim=-1)
        u_final = self._global_aggregate(x, edge_attr, batch, edge_batch)
        graph_emb = self.struct_proj(torch.cat([jk, u_final], dim=-1))

        outputs = self.output_head(graph_emb)
        outputs['attention'] = all_attn
        return outputs

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> 'HyalineTF':
        """Load a pretrained HyalineTF checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        if 'hyperparameters' in checkpoint:
            saved = checkpoint['hyperparameters']
            saved.update(kwargs)
            kwargs = saved
        model = cls(**kwargs)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
            print("Warning: Loaded with strict=False due to architecture mismatch")
        return model


# =============================================================================
# Convenience
# =============================================================================

def count_tf_parameters(model: nn.Module) -> int:
    """Count trainable parameters in HyalineTF."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing HyalineTF...")

    model = HyalineTF(
        node_input_dim=1280,
        hidden_dim=128,
        num_layers=2,
        use_domain_bias=True,
    )
    print(f"Parameters: {count_tf_parameters(model):,}")

    N = 120  # Typical TF size
    batch = torch.zeros(N, dtype=torch.long)

    # --- Sequence-only ---
    data_seq = Data(
        x=torch.randn(N, 1280),
        batch=batch,
        domain_types=torch.randint(0, 7, (N,)),
    )
    model.eval()
    with torch.no_grad():
        out = model(data_seq)
    print(f"Sequence-only | function_logits: {out['function_logits'].shape}, "
          f"binding: {out['binding'].shape}, regulatory: {out['regulatory'].shape}")

    # --- Structure-augmented ---
    E = N * 8
    data_struct = Data(
        x=torch.randn(N, 1280),
        pos=torch.randn(N, 3) * 10,
        edge_index=torch.randint(0, N, (2, E)),
        edge_attr=torch.randn(E, 3),
        batch=batch,
        domain_types=torch.randint(0, 7, (N,)),
    )
    with torch.no_grad():
        out = model(data_struct)
    print(f"Structure-aug | function_logits: {out['function_logits'].shape}, "
          f"attention layers: {len(out['attention'])}")
    print("✓ HyalineTF test passed!")
