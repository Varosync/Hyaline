"""
TFActivationModel: Context-dependent TF activation predictor.

This is the main model for predicting whether a transcription factor
will activate given its 3D structure and SCENIC+ cellular context.

Architecture:
    1. ContextEncoder: Encode SCENIC+ features → threshold modulation
    2. NodeEncoder: Project node features → hidden embeddings
    3. SpikingEGNN: Equivariant message passing with spiking dynamics
    4. SynchronizationDetector: Measure coordinated spiking
    5. ActivationHead: Predict P(TF activates)

The key insight: SCENIC+ context modulates spike thresholds.
Permissive context (high TF activity, open chromatin) → low threshold → high firing.
Synchronized firing across TF-DNA-coactivator nodes → successful complex → activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, NamedTuple
from dataclasses import dataclass

from .context_encoder import ContextEncoder, ContextEncoderConfig, ContextEncoderOutput
from .spike_encoder import SpikeEncoder, SpikeEncoderConfig
from .spiking_egnn import SpikingEGNN, SpikingEGNNConfig
from .activation_head import ActivationHead, ActivationHeadConfig


class TFActivationOutput(NamedTuple):
    """Output from TFActivationModel."""
    activation_prob: torch.Tensor  # P(TF activates) [batch]
    confidence: torch.Tensor  # Uncertainty [batch]
    sync_score: torch.Tensor  # Synchronization [batch]
    spike_rate: torch.Tensor  # Mean spike rate [batch]
    node_features: torch.Tensor  # Final node embeddings [N, hidden]
    context_embedding: torch.Tensor  # Context vector [batch, hidden]


@dataclass
class TFActivationConfig:
    """Configuration for TFActivationModel."""
    # Input dimensions
    node_input_dim: int = 32
    edge_input_dim: int = 8
    
    # Context encoder
    n_cell_types: int = 50
    n_tfs: int = 200
    n_topics: int = 30
    n_coactivators: int = 20
    
    # Architecture
    hidden_dim: int = 128
    num_egnn_layers: int = 4
    dropout: float = 0.1
    
    # Spiking parameters
    n_time_steps: int = 10
    beta: float = 0.9
    base_threshold: float = 1.0
    surrogate_slope: float = 25.0


class TFActivationModel(nn.Module):
    """Predict TF activation from structure + SCENIC+ context.
    
    This model addresses the PI's core question:
    "Given this TF structure and cellular context, will it activate?"
    
    The biological mechanism modeled:
    1. Context (cell type, chromatin, coactivators) sets activation threshold
    2. TF-DNA structure determines binding geometry
    3. Spiking dynamics simulate binding kinetics
    4. Synchronization = successful complex formation
    5. High sync → TF activates transcription
    
    Example:
        >>> config = TFActivationConfig()
        >>> model = TFActivationModel(config)
        >>> output = model(
        ...     node_features, pos, edge_index, edge_attr,
        ...     cell_type, tf_activity, chromatin, coactivators
        ... )
        >>> print(f"P(activation) = {output.activation_prob.item():.3f}")
    """
    
    def __init__(self, config: TFActivationConfig):
        super().__init__()
        self.config = config
        
        # Context encoder
        context_config = ContextEncoderConfig(
            n_cell_types=config.n_cell_types,
            n_tfs=config.n_tfs,
            n_topics=config.n_topics,
            n_coactivators=config.n_coactivators,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.context_encoder = ContextEncoder(context_config)
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.edge_input_dim, config.hidden_dim),
            nn.SiLU(),
        )
        
        # Spiking EGNN
        egnn_config = SpikingEGNNConfig(
            hidden_dim=config.hidden_dim,
            edge_dim=config.hidden_dim,
            num_layers=config.num_egnn_layers,
            dropout=config.dropout,
            beta=config.beta,
            base_threshold=config.base_threshold,
            surrogate_slope=config.surrogate_slope,
        )
        self.spiking_egnn = SpikingEGNN(egnn_config)
        
        # Activation head
        activation_config = ActivationHeadConfig(
            hidden_dim=64,
            output_dim=32,
            dropout=config.dropout,
            mc_dropout=True,
            context_dim=config.hidden_dim,  # Context embedding dimension
        )
        self.activation_head = ActivationHead(activation_config)
    
    def forward(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        # SCENIC+ context inputs
        cell_type_idx: Optional[torch.Tensor] = None,
        tf_activity: Optional[torch.Tensor] = None,
        chromatin_topics: Optional[torch.Tensor] = None,
        coactivator_expr: Optional[torch.Tensor] = None,
    ) -> TFActivationOutput:
        """Forward pass.
        
        Args:
            node_features: [N, node_input_dim] node features
            pos: [N, 3] node positions
            edge_index: [2, E] edge connectivity
            edge_attr: [E, edge_input_dim] edge features
            cell_type_idx: [batch] cell type indices
            tf_activity: [batch, n_tfs] TF activity scores
            chromatin_topics: [batch, n_topics] chromatin topic weights
            coactivator_expr: [batch, n_coactivators] coactivator expression
            
        Returns:
            TFActivationOutput with activation probability and diagnostics
        """
        N = node_features.size(0)
        device = node_features.device
        dtype = node_features.dtype
        
        # Encode context if provided
        threshold_mod = None
        context_embedding = None
        
        if cell_type_idx is not None:
            # Encode SCENIC+ context
            context_output = self.context_encoder(
                cell_type_idx,
                tf_activity,
                chromatin_topics,
                coactivator_expr,
            )
            
            # Get threshold modulation
            threshold_mod = context_output.threshold_offset.squeeze(-1)  # [batch]
            context_embedding = context_output.context_embedding
            
            # Expand to per-node if single batch
            if threshold_mod.size(0) == 1:
                threshold_mod = threshold_mod.expand(N)
        else:
            # No context - use zeros
            context_embedding = torch.zeros(1, self.config.hidden_dim, device=device, dtype=dtype)
        
        # Encode node features
        h = self.node_encoder(node_features)  # [N, hidden_dim]
        
        # Encode edge features
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)  # [E, hidden_dim]
        
        # Run spiking EGNN
        h_out, pos_out, all_spikes = self.spiking_egnn(
            h, pos, edge_index, edge_attr, threshold_mod
        )
        
        # Compute synchronization
        sync_score = self.spiking_egnn.get_sync_score(all_spikes)
        
        # Compute spike statistics
        spike_rates = all_spikes.mean(dim=0)  # [N] average across layers
        mean_spike_rate = spike_rates.mean()
        
        # Predict activation
        # Need to batch spike_rates if not batched
        if spike_rates.dim() == 1:
            spike_rates_batch = spike_rates.unsqueeze(0)  # [1, N]
        else:
            spike_rates_batch = spike_rates
        
        if sync_score.dim() == 0:
            sync_score_batch = sync_score.unsqueeze(0)  # [1]
        else:
            sync_score_batch = sync_score
        
        activation_output = self.activation_head(
            sync_score_batch,
            spike_rates_batch,
            context_embedding,
        )
        
        return TFActivationOutput(
            activation_prob=activation_output.activation_prob,
            confidence=activation_output.confidence,
            sync_score=sync_score_batch,
            spike_rate=mean_spike_rate.unsqueeze(0),
            node_features=h_out,
            context_embedding=context_embedding,
        )
    
    def forward_no_context(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TFActivationOutput:
        """Forward without SCENIC+ context (for ablation studies)."""
        return self.forward(
            node_features, pos, edge_index, edge_attr,
            cell_type_idx=None,
            tf_activity=None,
            chromatin_topics=None,
            coactivator_expr=None,
        )
    
    @torch.no_grad()
    def predict_activation(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        cell_type_idx: Optional[torch.Tensor] = None,
        tf_activity: Optional[torch.Tensor] = None,
        chromatin_topics: Optional[torch.Tensor] = None,
        coactivator_expr: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Predict activation with threshold.
        
        Returns:
            Dict with 'activated' (bool), 'probability', 'confidence'
        """
        self.eval()
        output = self.forward(
            node_features, pos, edge_index, edge_attr,
            cell_type_idx, tf_activity, chromatin_topics, coactivator_expr,
        )
        
        return {
            'activated': output.activation_prob > threshold,
            'probability': output.activation_prob,
            'confidence': output.confidence,
            'sync_score': output.sync_score,
        }
