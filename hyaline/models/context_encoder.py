"""
ContextEncoder: Encode SCENIC+ cellular context for TF activation prediction.

This module integrates multiple sources of cellular context:
- Cell type identity (categorical)
- TF activity scores from SCENIC+ GRN inference
- Chromatin accessibility topics from cisTopic
- Coactivator expression levels

The encoded context modulates spike thresholds in the spiking GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Optional
from dataclasses import dataclass
import math


class ContextEncoderOutput(NamedTuple):
    """Output container for ContextEncoder.
    
    Attributes:
        context_embedding: Global context vector [batch, hidden_dim]
        threshold_offset: Global threshold modulation [batch, 1]
            Negative = permissive (easy firing)
            Positive = repressive (hard firing)
        node_modulation: Per-node modulation vector [batch, hidden_dim]
    """
    context_embedding: torch.Tensor
    threshold_offset: torch.Tensor
    node_modulation: torch.Tensor


@dataclass
class ContextEncoderConfig:
    """Configuration for ContextEncoder."""
    n_cell_types: int = 50
    n_tfs: int = 200
    n_topics: int = 30
    n_coactivators: int = 20
    hidden_dim: int = 128
    cell_type_dim: int = 32
    tf_proj_dim: int = 64
    chromatin_proj_dim: int = 32
    coactivator_proj_dim: int = 32
    dropout: float = 0.1
    use_layer_norm: bool = True


class ContextEncoder(nn.Module):
    """Encode SCENIC+ context into threshold modulation signals.
    
    Maps cellular context (cell type, TF activity, chromatin state)
    to threshold modulations for the spiking GNN layers.
    
    The key biological insight:
    - Permissive context (high TF activity, open chromatin) → low threshold
    - Repressive context (low TF activity, closed chromatin) → high threshold
    
    Example:
        >>> config = ContextEncoderConfig(n_cell_types=50, n_tfs=200)
        >>> encoder = ContextEncoder(config)
        >>> output = encoder(cell_type, tf_activity, chromatin, coactivators)
        >>> output.threshold_offset  # Use to modulate spike thresholds
    """
    
    def __init__(self, config: ContextEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input embeddings/projections
        self.cell_type_embed = nn.Embedding(
            config.n_cell_types, config.cell_type_dim
        )
        
        self.tf_proj = nn.Sequential(
            nn.Linear(config.n_tfs, config.tf_proj_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        
        self.chromatin_proj = nn.Sequential(
            nn.Linear(config.n_topics, config.chromatin_proj_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        
        self.coactivator_proj = nn.Sequential(
            nn.Linear(config.n_coactivators, config.coactivator_proj_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        
        # Fusion dimension
        fusion_dim = (
            config.cell_type_dim + 
            config.tf_proj_dim + 
            config.chromatin_proj_dim + 
            config.coactivator_proj_dim
        )
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Output heads
        self.threshold_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # Bound to [-1, 1], scaled later
        )
        
        self.node_mod_head = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Optional layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity()
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        cell_type_idx: torch.Tensor,
        tf_activity: torch.Tensor,
        chromatin_topics: torch.Tensor,
        coactivator_expr: torch.Tensor,
    ) -> ContextEncoderOutput:
        """Encode cellular context.
        
        Args:
            cell_type_idx: [batch] cell type indices
            tf_activity: [batch, n_tfs] TF activity scores
            chromatin_topics: [batch, n_topics] chromatin topic weights
            coactivator_expr: [batch, n_coactivators] coactivator expression
            
        Returns:
            ContextEncoderOutput with context embedding and threshold modulation
        """
        # Embed cell type
        cell_embed = self.cell_type_embed(cell_type_idx)  # [batch, cell_type_dim]
        
        # Project continuous features
        tf_embed = self.tf_proj(tf_activity)  # [batch, tf_proj_dim]
        chrom_embed = self.chromatin_proj(chromatin_topics)  # [batch, chromatin_proj_dim]
        coact_embed = self.coactivator_proj(coactivator_expr)  # [batch, coactivator_proj_dim]
        
        # Fuse all context
        fused = torch.cat([cell_embed, tf_embed, chrom_embed, coact_embed], dim=-1)
        context = self.fusion_mlp(fused)
        context = self.layer_norm(context)
        
        # Compute outputs
        threshold_offset = self.threshold_head(context)  # [batch, 1]
        node_modulation = self.node_mod_head(context)  # [batch, hidden_dim]
        
        return ContextEncoderOutput(
            context_embedding=context,
            threshold_offset=threshold_offset,
            node_modulation=node_modulation,
        )
    
    def get_threshold_for_node(
        self,
        node_features: torch.Tensor,
        context_output: ContextEncoderOutput,
        base_threshold: float = 1.0,
        scale: float = 0.5,
    ) -> torch.Tensor:
        """Compute per-node thresholds from context.
        
        Args:
            node_features: [N, hidden_dim] node embeddings
            context_output: output from forward()
            base_threshold: base spike threshold
            scale: how much context can shift threshold
            
        Returns:
            [N] per-node thresholds
        """
        # Project node features and compute node-specific offsets
        node_context = torch.matmul(
            node_features, 
            context_output.node_modulation.unsqueeze(-1)
        ).squeeze(-1)  # [N]
        
        # Combine global and node-specific offsets
        global_offset = context_output.threshold_offset.squeeze(-1)  # [1] or [batch]
        
        # Final threshold: base + global_offset + node_offset
        threshold = base_threshold + scale * (global_offset + 0.1 * torch.tanh(node_context))
        
        return threshold
