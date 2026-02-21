"""
ActivationHead: Predict TF activation probability from synchronization dynamics.

This module converts synchronization signals from the spiking GNN into
a biologically interpretable activation probability. Synchronized binding
events (high sync score) indicate successful TF-DNA-coactivator complex
formation, which predicts transcriptional activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Optional
from dataclasses import dataclass
import math


class ActivationHeadOutput(NamedTuple):
    """Output container for ActivationHead.
    
    Attributes:
        activation_prob: P(TF activates) [batch]
        confidence: Uncertainty estimate [batch] (higher = more certain)
        spike_entropy: Spike pattern entropy [batch]
    """
    activation_prob: torch.Tensor
    confidence: torch.Tensor
    spike_entropy: torch.Tensor


@dataclass
class ActivationHeadConfig:
    """Configuration for ActivationHead."""
    hidden_dim: int = 64
    output_dim: int = 32
    dropout: float = 0.1
    mc_dropout: bool = True
    n_mc_samples: int = 10
    context_dim: int = 64  # Dimension of context embedding to include


class ActivationHead(nn.Module):
    """Predict TF activation from synchronization scores.
    
    The biological interpretation:
    - High sync + high spike rate → coordinated binding → ACTIVATION
    - Low sync or low spike rate → failed complex → NO ACTIVATION
    
    Features computed from spike dynamics:
    1. Global synchronization score
    2. Mean/max spike rates  
    3. Spike entropy (synchronization quality)
    
    Example:
        >>> head = ActivationHead(ActivationHeadConfig())
        >>> output = head(global_sync, spike_rates)
        >>> print(output.activation_prob)
    """
    
    def __init__(self, config: ActivationHeadConfig):
        super().__init__()
        self.config = config
        
        # Feature dimension: sync + mean + std + max + entropy = 5
        spike_feature_dim = 5
        
        # Context projection (to combine with spike features)
        self.context_proj = nn.Sequential(
            nn.Linear(config.context_dim, config.hidden_dim // 2),
            nn.SiLU(),
        )
        
        # Spike feature MLP
        self.spike_mlp = nn.Sequential(
            nn.Linear(spike_feature_dim, config.hidden_dim // 2),
            nn.SiLU(),
        )
        
        # Combined feature MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        
        # Activation output
        self.activation_head = nn.Linear(config.output_dim, 1)
        
        # Confidence output (optional)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.output_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for balanced predictions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize activation head bias to 0 (sigmoid(0) = 0.5)
        # This ensures predictions start near 0.5, not near 0
        nn.init.zeros_(self.activation_head.bias)
    
    def compute_spike_entropy(self, spike_rates: torch.Tensor) -> torch.Tensor:
        """Compute entropy of spike rate distribution.
        
        Low entropy = synchronized (all nodes firing similarly)
        High entropy = desynchronized (random firing patterns)
        
        Args:
            spike_rates: [batch, n_nodes] spike rates
            
        Returns:
            [batch] entropy values (normalized to [0, 1])
        """
        # Normalize to probability distribution
        eps = 1e-8
        p = spike_rates / (spike_rates.sum(dim=-1, keepdim=True) + eps)
        p = p.clamp(min=eps)
        
        # Shannon entropy
        entropy = -(p * torch.log(p)).sum(dim=-1)
        
        # Normalize by max entropy (log(n_nodes))
        n_nodes = spike_rates.size(-1)
        max_entropy = math.log(n_nodes) if n_nodes > 1 else 1.0
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def forward(
        self,
        global_sync: torch.Tensor,
        spike_rates: torch.Tensor,
        context_embedding: Optional[torch.Tensor] = None,
        sync_traces: Optional[torch.Tensor] = None,
    ) -> ActivationHeadOutput:
        """Predict activation from sync dynamics and context.
        
        Args:
            global_sync: [batch] global synchronization score
            spike_rates: [batch, n_nodes] per-node spike rates
            context_embedding: [batch, context_dim] SCENIC+ context embedding
            sync_traces: [batch, time, n_nodes] optional detailed traces
            
        Returns:
            ActivationHeadOutput with activation probability and confidence
        """
        # Ensure proper dimensions
        # Flatten global_sync to 1D if needed
        global_sync = global_sync.flatten()
        if global_sync.dim() == 0:
            global_sync = global_sync.unsqueeze(0)
        
        # Handle spike_rates dimensions
        if spike_rates.dim() == 1:
            spike_rates = spike_rates.unsqueeze(0)
        batch_size = spike_rates.size(0)
        device = spike_rates.device
        dtype = spike_rates.dtype
            
        mean_rate = spike_rates.mean(dim=-1)  # [batch]
        std_rate = spike_rates.std(dim=-1)    # [batch]
        max_rate = spike_rates.max(dim=-1)[0] # [batch]
        entropy = self.compute_spike_entropy(spike_rates)  # [batch]
        
        # Ensure global_sync matches batch size
        if global_sync.size(0) == 1 and batch_size > 1:
            global_sync = global_sync.expand(batch_size)
        elif global_sync.size(0) != batch_size:
            global_sync = global_sync[:batch_size]
        
        # Spike features
        spike_features = torch.stack([
            global_sync,
            mean_rate,
            std_rate,
            max_rate,
            entropy,
        ], dim=-1)  # [batch, 5]
        
        spike_hidden = self.spike_mlp(spike_features)  # [batch, hidden_dim//2]
        
        # Context features
        if context_embedding is not None:
            # Handle context dimension
            if context_embedding.dim() == 1:
                context_embedding = context_embedding.unsqueeze(0)
            # Truncate or pad to expected dimension
            if context_embedding.size(-1) != self.config.context_dim:
                if context_embedding.size(-1) > self.config.context_dim:
                    context_embedding = context_embedding[..., :self.config.context_dim]
                else:
                    padding = torch.zeros(
                        context_embedding.size(0),
                        self.config.context_dim - context_embedding.size(-1),
                        device=device, dtype=dtype
                    )
                    context_embedding = torch.cat([context_embedding, padding], dim=-1)
            context_hidden = self.context_proj(context_embedding)  # [batch, hidden_dim//2]
        else:
            # No context - use zeros
            context_hidden = torch.zeros(batch_size, self.config.hidden_dim // 2, device=device, dtype=dtype)
        
        # Combine spike + context
        combined = torch.cat([spike_hidden, context_hidden], dim=-1)  # [batch, hidden_dim]
        
        # Forward through combined MLP
        hidden = self.feature_mlp(combined)  # [batch, output_dim]
        
        # Predictions
        activation_logit = self.activation_head(hidden).squeeze(-1)  # [batch]
        activation_prob = torch.sigmoid(activation_logit)
        
        confidence = self.confidence_head(hidden).squeeze(-1)  # [batch]
        
        return ActivationHeadOutput(
            activation_prob=activation_prob,
            confidence=confidence,
            spike_entropy=entropy,
        )
    
    def forward_with_uncertainty(
        self,
        global_sync: torch.Tensor,
        spike_rates: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> ActivationHeadOutput:
        """Forward with MC dropout for uncertainty estimation.
        
        Args:
            global_sync: [batch] global sync
            spike_rates: [batch, n_nodes] spike rates
            n_samples: number of MC samples
            
        Returns:
            Mean prediction with uncertainty-based confidence
        """
        if not self.config.mc_dropout:
            return self.forward(global_sync, spike_rates)
        
        n_samples = n_samples or self.config.n_mc_samples
        self.train()  # Enable dropout
        
        preds = []
        for _ in range(n_samples):
            out = self.forward(global_sync, spike_rates)
            preds.append(out.activation_prob)
        
        preds = torch.stack(preds, dim=0)  # [n_samples, batch]
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        
        # Confidence = 1 - uncertainty
        confidence = 1.0 - (std_pred / 0.5).clamp(0, 1)
        
        self.eval()
        
        # Get entropy from single forward
        out = self.forward(global_sync, spike_rates)
        
        return ActivationHeadOutput(
            activation_prob=mean_pred,
            confidence=confidence,
            spike_entropy=out.spike_entropy,
        )
