"""
SpikeEncoder: Leaky Integrate-and-Fire neurons with context-dependent thresholds.

This module implements spiking neural network dynamics for encoding
structure features as spike trains. Context from SCENIC+ modulates
the spike threshold, making firing easier (permissive context) or
harder (repressive context).

Key equations:
    V[t+1] = β * V[t] + W * x[t]  (membrane integration)
    S[t] = H(V[t] - θ)            (spike generation, H = Heaviside)
    V[t] = V[t] - θ * S[t]        (soft reset)

Surrogate gradient: Fast sigmoid for backpropagation through spikes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Optional, Tuple
from dataclasses import dataclass
import math


class SpikeEncoderOutput(NamedTuple):
    """Output container for SpikeEncoder.
    
    Attributes:
        spike_counts: [batch, n_nodes] total spikes per node
        spike_rates: [batch, n_nodes] normalized spike rates
        membrane_traces: [batch, time, n_nodes] membrane potentials
        spike_trains: [batch, time, n_nodes] binary spike trains
    """
    spike_counts: torch.Tensor
    spike_rates: torch.Tensor
    membrane_traces: torch.Tensor
    spike_trains: torch.Tensor


@dataclass
class SpikeEncoderConfig:
    """Configuration for SpikeEncoder."""
    input_dim: int = 128
    hidden_dim: int = 128
    n_steps: int = 10
    beta: float = 0.9  # Leak factor
    base_threshold: float = 1.0
    threshold_scale: float = 0.5  # How much context can shift threshold
    reset_mechanism: str = 'soft'  # 'soft' or 'hard'
    surrogate_slope: float = 25.0  # Slope of surrogate gradient


class SurrogateSpike(torch.autograd.Function):
    """Spike function with surrogate gradient for backprop.
    
    Forward: Heaviside step function
    Backward: Fast sigmoid derivative
    """
    
    @staticmethod
    def forward(ctx, membrane, threshold, slope):
        ctx.save_for_backward(membrane, threshold)
        ctx.slope = slope
        return (membrane > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        slope = ctx.slope
        
        # Surrogate gradient: derivative of fast sigmoid
        diff = membrane - threshold
        grad = slope / (2 * (1 + torch.abs(slope * diff)) ** 2)
        
        return grad * grad_output, None, None


def spike_fn(membrane: torch.Tensor, threshold: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    """Generate spikes with surrogate gradient."""
    return SurrogateSpike.apply(membrane, threshold, slope)


class LIFNeuronLayer(nn.Module):
    """Leaky Integrate-and-Fire neuron layer.
    
    Implements LIF dynamics with optional context-dependent threshold.
    """
    
    def __init__(self, config: SpikeEncoderConfig):
        super().__init__()
        self.config = config
        
        self.beta = config.beta
        self.base_threshold = config.base_threshold
        self.slope = config.surrogate_slope
        self.reset_mechanism = config.reset_mechanism
    
    def forward(
        self,
        x: torch.Tensor,
        threshold_offset: Optional[torch.Tensor] = None,
    ) -> SpikeEncoderOutput:
        """Run LIF dynamics.
        
        Args:
            x: [batch, n_nodes, hidden_dim] input features
            threshold_offset: [batch] or [batch, n_nodes] threshold modulation
            
        Returns:
            SpikeEncoderOutput with spike counts, rates, and traces
        """
        batch_size, n_nodes, hidden_dim = x.shape
        n_steps = self.config.n_steps
        device = x.device
        dtype = x.dtype
        
        # Compute effective threshold
        threshold = self._compute_threshold(threshold_offset, batch_size, n_nodes, device, dtype)
        
        # Initialize membrane potential
        membrane = torch.zeros(batch_size, n_nodes, device=device, dtype=dtype)
        
        # Storage for traces
        spike_trains = []
        membrane_traces = []
        
        # Time-step simulation
        for t in range(n_steps):
            # Input current (sum over hidden dim)
            current = x.mean(dim=-1)  # [batch, n_nodes]
            
            # Membrane integration
            membrane = self.beta * membrane + current
            
            # Spike generation
            spikes = spike_fn(membrane, threshold, self.slope)
            
            # Reset
            if self.reset_mechanism == 'soft':
                membrane = membrane - threshold * spikes
            else:  # hard reset
                membrane = membrane * (1 - spikes)
            
            # Store
            spike_trains.append(spikes)
            membrane_traces.append(membrane.clone())
        
        # Stack traces: [batch, time, n_nodes]
        spike_trains = torch.stack(spike_trains, dim=1)
        membrane_traces = torch.stack(membrane_traces, dim=1)
        
        # Compute spike counts and rates
        spike_counts = spike_trains.sum(dim=1)  # [batch, n_nodes]
        spike_rates = spike_counts / n_steps
        
        return SpikeEncoderOutput(
            spike_counts=spike_counts,
            spike_rates=spike_rates,
            membrane_traces=membrane_traces,
            spike_trains=spike_trains,
        )
    
    def _compute_threshold(
        self,
        offset: Optional[torch.Tensor],
        batch_size: int,
        n_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute effective threshold with context modulation."""
        threshold = torch.full(
            (batch_size, n_nodes), 
            self.base_threshold, 
            device=device, 
            dtype=dtype
        )
        
        if offset is not None:
            offset = offset.to(device=device, dtype=dtype)
            
            if offset.dim() == 1:  # [batch]
                offset = offset.unsqueeze(-1).expand(-1, n_nodes)
            
            # Scale and apply offset
            threshold = threshold + self.config.threshold_scale * offset
            
            # Ensure threshold stays positive
            threshold = threshold.clamp(min=0.1)
        
        return threshold


class SpikeEncoder(nn.Module):
    """Encode features as spike trains with context modulation.
    
    This is the main interface for the spiking encoder. It projects
    input features and runs LIF dynamics with optional context-dependent
    threshold modulation.
    
    Example:
        >>> encoder = SpikeEncoder(SpikeEncoderConfig())
        >>> output = encoder(node_features, threshold_offset)
        >>> print(output.spike_rates)  # Use for downstream tasks
    """
    
    def __init__(self, config: SpikeEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        
        # LIF neurons
        self.lif = LIFNeuronLayer(config)
    
    def forward(
        self,
        x: torch.Tensor,
        threshold_offset: Optional[torch.Tensor] = None,
    ) -> SpikeEncoderOutput:
        """Encode features as spikes.
        
        Args:
            x: [batch, n_nodes, input_dim] or [n_nodes, input_dim] node features
            threshold_offset: [batch] global threshold offset from context
            
        Returns:
            SpikeEncoderOutput with spike dynamics
        """
        # Handle unbatched input
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Project features
        x = self.input_proj(x)  # [batch, n_nodes, hidden_dim]
        
        # Run LIF dynamics
        return self.lif(x, threshold_offset)
    
    def get_global_sync(self, output: SpikeEncoderOutput) -> torch.Tensor:
        """Compute global synchronization from spike trains.
        
        High sync = nodes firing together = coherent activation
        Low sync = random firing = incoherent activity
        
        Args:
            output: Output from forward()
            
        Returns:
            [batch] global synchronization score in [0, 1]
        """
        # Get spike trains: [batch, time, n_nodes]
        trains = output.spike_trains
        
        # Compute pairwise correlation
        # First, compute total spikes per timestep
        sync_signal = trains.sum(dim=-1)  # [batch, time]
        
        # Normalize
        max_sync = trains.size(-1)  # All nodes firing together
        sync_signal = sync_signal / max_sync
        
        # Measure variance of sync signal (high variance = synchronized bursts)
        sync_var = sync_signal.var(dim=-1)  # [batch]
        
        # Convert to 0-1 score (higher variance = more sync)
        global_sync = torch.tanh(4 * sync_var)  # Scale for reasonable range
        
        return global_sync
