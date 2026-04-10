"""
SpikingEGNN: SE(3) Equivariant GNN with spiking message passing.

This module combines EGNN's equivariant message passing with
spiking neural network dynamics. Spike generation in each layer
is modulated by context from SCENIC+.

Key innovation: Message passing triggers spikes, synchronization
across nodes indicates successful binding complex formation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass


class SpikingEGNNOutput(NamedTuple):
    """Output from SpikingEGNN layer."""
    h: torch.Tensor  # Updated node features
    pos: torch.Tensor  # Updated positions
    spikes: torch.Tensor  # Spikes from this layer
    membrane: torch.Tensor  # Membrane potentials


@dataclass
class SpikingEGNNConfig:
    """Configuration for SpikingEGNN."""
    hidden_dim: int = 128
    edge_dim: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    beta: float = 0.9  # Leak factor
    base_threshold: float = 1.0
    surrogate_slope: float = 25.0


class SurrogateSpike(torch.autograd.Function):
    """Spike with surrogate gradient."""
    
    @staticmethod
    def forward(ctx, x, threshold, slope):
        ctx.save_for_backward(x, threshold)
        ctx.slope = slope
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad):
        x, threshold = ctx.saved_tensors
        slope = ctx.slope
        diff = x - threshold
        sg = slope / (2 * (1 + torch.abs(slope * diff)) ** 2)
        return sg * grad, None, None


class SpikingEGNNLayer(nn.Module):
    """Single Spiking EGNN layer.
    
    Combines equivariant message passing with LIF spiking dynamics.
    Context modulates per-node thresholds.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.base_threshold = threshold
        self.slope = slope
        
        # Edge MLP (message function)
        edge_input_dim = 2 * hidden_dim + edge_dim + 1  # h_i, h_j, edge_attr, dist
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Node MLP (update function)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Coordinate update (small for stability)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        
        # Initialize coord update to small values
        nn.init.zeros_(self.coord_mlp[-1].weight)
        
        # Learnable membrane gain - helps spikes fire (start moderate)
        self.membrane_gain = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        membrane: Optional[torch.Tensor] = None,
        threshold_mod: Optional[torch.Tensor] = None,
    ) -> SpikingEGNNOutput:
        """Forward with spiking dynamics.
        
        Args:
            h: [N, hidden_dim] node features
            pos: [N, 3] node positions
            edge_index: [2, E] edge connectivity
            edge_attr: [E, edge_dim] edge features
            membrane: [N, hidden_dim] membrane potential from previous layer
            threshold_mod: [N] or [1] threshold modulation from context
            
        Returns:
            SpikingEGNNOutput with updated features, positions, and spikes
        """
        row, col = edge_index
        N = h.size(0)
        device = h.device
        dtype = h.dtype
        
        # Initialize membrane if not provided
        if membrane is None:
            membrane = torch.zeros_like(h)
        
        # Compute distances (equivariant)
        rel_pos = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        # Edge features
        if edge_attr is None:
            edge_attr = torch.zeros(row.size(0), 0, device=device, dtype=dtype)
        
        edge_input = torch.cat([h[row], h[col], edge_attr, dist], dim=-1)
        messages = self.edge_mlp(edge_input)  # [E, hidden_dim]
        
        # Aggregate messages with degree normalization
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, messages.to(agg.dtype))
        degree = torch.zeros(N, device=device, dtype=dtype)
        degree.index_add_(0, row, torch.ones(row.size(0), device=device, dtype=dtype))
        agg = agg / degree.clamp(min=1).unsqueeze(-1)
        
        # Update node features
        node_input = torch.cat([h, agg], dim=-1)
        h_update = self.node_mlp(node_input)
        h_new = self.layer_norm(h + h_update)
        
        # === Spiking Dynamics ===
        # Integrate input into membrane
        membrane = self.beta * membrane + h_new
        
        # Compute threshold
        threshold = self._get_threshold(threshold_mod, N, device, dtype)
        
        # Generate spikes with learnable gain
        # Use sum of absolute values for more signal, scaled by learnable gain
        membrane_signal = membrane.abs().mean(dim=-1) * self.membrane_gain  # [N]
        
        spikes = SurrogateSpike.apply(membrane_signal, threshold, self.slope)
        
        # Soft reset
        membrane = membrane - threshold.unsqueeze(-1) * spikes.unsqueeze(-1)
        
        # === Equivariant coordinate update ===
        coord_weights = self.coord_mlp(messages)  # [E, 1]
        coord_weights = coord_weights.clamp(-1, 1)
        
        # Weighted direction update
        coord_update = coord_weights * rel_pos / (dist + 1e-8)  # [E, 3]
        
        pos_agg = torch.zeros_like(pos)
        pos_agg.index_add_(0, row, coord_update.to(pos_agg.dtype))
        pos_new = pos + 0.1 * pos_agg / degree.clamp(min=1).unsqueeze(-1)  # Small update
        
        return SpikingEGNNOutput(
            h=h_new,
            pos=pos_new,
            spikes=spikes,
            membrane=membrane,
        )
    
    def _get_threshold(
        self,
        mod: Optional[torch.Tensor],
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get per-node thresholds with optional modulation."""
        threshold = torch.full((N,), self.base_threshold, device=device, dtype=dtype)
        
        if mod is not None:
            mod = mod.to(device=device, dtype=dtype)
            if mod.dim() == 0:
                mod = mod.unsqueeze(0).expand(N)
            elif mod.size(0) == 1:
                mod = mod.expand(N)
            threshold = threshold + 0.5 * mod
            threshold = threshold.clamp(min=0.1)
        
        return threshold


class SpikingEGNN(nn.Module):
    """Multi-layer Spiking EGNN.
    
    Stacks multiple SpikingEGNNLayer with shared membrane dynamics.
    Context modulation applied to all layers.
    
    Example:
        >>> model = SpikingEGNN(SpikingEGNNConfig())
        >>> h, pos, all_spikes = model(h, pos, edge_index, edge_attr, threshold_mod)
    """
    
    def __init__(self, config: SpikingEGNNConfig):
        super().__init__()
        self.config = config
        
        # Stack of spiking EGNN layers
        self.layers = nn.ModuleList([
            SpikingEGNNLayer(
                hidden_dim=config.hidden_dim,
                edge_dim=config.edge_dim,
                beta=config.beta,
                threshold=config.base_threshold,
                slope=config.surrogate_slope,
            )
            for _ in range(config.num_layers)
        ])
    
    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        threshold_mod: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward through all layers.
        
        Args:
            h: [N, hidden_dim] node features
            pos: [N, 3] positions
            edge_index: [2, E] edges
            edge_attr: [E, edge_dim] edge features
            threshold_mod: [N] or [1] threshold modulation
            
        Returns:
            h: Final node features
            pos: Final positions
            all_spikes: [num_layers, N] spikes from each layer
        """
        membrane = None
        all_spikes = []
        
        for layer in self.layers:
            output = layer(h, pos, edge_index, edge_attr, membrane, threshold_mod)
            h = output.h
            pos = output.pos
            membrane = output.membrane
            all_spikes.append(output.spikes)
        
        all_spikes = torch.stack(all_spikes, dim=0)  # [num_layers, N]
        
        return h, pos, all_spikes
    
    def set_threshold(self, threshold: float) -> None:
        """Set base threshold for all layers (for annealing).
        
        Start low (0.3) so spikes fire easily with diverse patterns,
        anneal up to 1.0 so the model learns meaningful synchronization.
        """
        for layer in self.layers:
            layer.base_threshold = threshold

    def get_sync_score(self, all_spikes: torch.Tensor) -> torch.Tensor:
        """Compute synchronization from layer spikes.
        
        Synchronization = how much nodes spike together.
        High sync = nodes firing at similar rates across layers.
        Low sync = random/uncorrelated firing.
        
        Args:
            all_spikes: [num_layers, N] spikes
            
        Returns:
            Scalar synchronization score in [0, 1]
        """
        num_layers, N = all_spikes.shape
        
        if N < 2:
            return torch.tensor(0.5, device=all_spikes.device)
        
        # Total spike rate (want moderate, not 0 or 1)
        total_rate = all_spikes.mean()
        
        # Sync within each layer: how uniform is firing across nodes?
        # Low variance within layer = high sync (all fire or all quiet)
        # High variance = low sync (some fire, some don't)
        within_layer_var = all_spikes.var(dim=-1).mean()  # [num_layers] -> scalar
        
        # Sync = 1 - normalized variance (low var = high sync)
        # Variance of Bernoulli is at most 0.25 (at p=0.5)
        sync_uniformity = 1.0 - (within_layer_var / 0.25).clamp(0, 1)
        
        # Also reward moderate spike rates (not all 0 or all 1)
        # Best at 0.5, worst at 0 or 1
        rate_quality = 1.0 - (2 * total_rate - 1).abs()
        
        # Combine: high uniformity + moderate rate = good sync
        sync_score = 0.7 * sync_uniformity + 0.3 * rate_quality
        
        return sync_score.clamp(0, 1)
