"""
Type II Inhibitor Screening Model
==================================

Hybrid model that combines pocket geometric features with compound descriptors
to predict Type II kinase inhibitor binding.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Type2ScreeningModel(nn.Module):
    """Hybrid model for Type II inhibitor prediction.
    
    Architecture
    ------------
    - Pocket encoder: Processes geometric features (DFG-αC distance, angles, etc.)
    - Compound encoder: Processes molecular descriptors (MW, LogP, etc.)
    - Interaction predictor: Predicts Type II score, DFG-out probability, and affinity
    
    Inputs
    ------
    - Pocket features (16-dim): DFG-αC distance, hinge angle, volume, ESP, etc.
    - Compound features (8-dim): MW, LogP, HBD, HBA, price, etc.
    
    Outputs
    -------
    - Type II score (0-1): Probability of Type II binding mode
    - DFG-out probability (0-1): Probability of inducing DFG-out conformation
    - Binding affinity (pKi): Predicted binding affinity
    """
    
    def __init__(
        self,
        pocket_dim: int = 16,
        compound_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.pocket_dim = pocket_dim
        self.compound_dim = compound_dim
        self.hidden_dim = hidden_dim
        
        # Pocket feature encoder
        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Compound feature encoder
        self.compound_encoder = nn.Sequential(
            nn.Linear(compound_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Interaction predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [type2_score, dfg_out_prob, affinity]
        )
        
    def forward(
        self,
        pocket_features: torch.Tensor,
        compound_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pocket_features : Tensor, shape (batch, pocket_dim)
            Pocket geometric and physicochemical features
        compound_features : Tensor, shape (batch, compound_dim)
            Compound molecular descriptors
        
        Returns
        -------
        Tensor, shape (batch, 3)
            [type2_score, dfg_out_prob, binding_affinity]
        """
        # Encode features
        pocket_emb = self.pocket_encoder(pocket_features)
        compound_emb = self.compound_encoder(compound_features)
        
        # Combine and predict
        combined = torch.cat([pocket_emb, compound_emb], dim=-1)
        output = self.predictor(combined)
        
        # Apply activations
        type2_score = torch.sigmoid(output[:, 0])
        dfg_out_prob = torch.sigmoid(output[:, 1])
        affinity = output[:, 2]  # Regression (pKi)
        
        return torch.stack([type2_score, dfg_out_prob, affinity], dim=-1)
    
    def predict_type2_score(
        self,
        pocket_features: torch.Tensor,
        compound_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict only Type II score (for ranking)."""
        output = self.forward(pocket_features, compound_features)
        return output[:, 0]


def load_screening_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Type2ScreeningModel:
    """Load pre-trained screening model.
    
    Parameters
    ----------
    checkpoint_path : str, optional
        Path to model checkpoint (.pt file)
    device : str
        Device to load model on ('cpu' or 'cuda')
        
    Returns
    -------
    Type2ScreeningModel
        Loaded model in eval mode
    """
    model = Type2ScreeningModel()
    
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading model from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        
        if 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
            
        logger.info("Model loaded successfully")
    else:
        logger.warning(
            "No checkpoint provided or file not found - using randomly initialized model. "
            "Predictions will not be meaningful until model is trained."
        )
    
    model = model.to(device)
    model.eval()
    
    return model


def train_screening_model(
    model: Type2ScreeningModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "checkpoints/screening",
) -> Type2ScreeningModel:
    """Train the screening model.
    
    Parameters
    ----------
    model : Type2ScreeningModel
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    num_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    device : str
        Device to train on
    checkpoint_dir : str
        Directory to save checkpoints
        
    Returns
    -------
    Type2ScreeningModel
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Multi-task loss
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            pocket_feat, compound_feat, labels = batch
            pocket_feat = pocket_feat.to(device)
            compound_feat = compound_feat.to(device)
            labels = labels.to(device)  # [type2_label, dfg_out_label, affinity]
            
            optimizer.zero_grad()
            
            preds = model(pocket_feat, compound_feat)
            
            # Multi-task loss
            loss_type2 = bce_loss(preds[:, 0], labels[:, 0])
            loss_dfg = bce_loss(preds[:, 1], labels[:, 1])
            loss_affinity = mse_loss(preds[:, 2], labels[:, 2])
            
            loss = loss_type2 + loss_dfg + 0.5 * loss_affinity
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                pocket_feat, compound_feat, labels = batch
                pocket_feat = pocket_feat.to(device)
                compound_feat = compound_feat.to(device)
                labels = labels.to(device)
                
                preds = model(pocket_feat, compound_feat)
                
                loss_type2 = bce_loss(preds[:, 0], labels[:, 0])
                loss_dfg = bce_loss(preds[:, 1], labels[:, 1])
                loss_affinity = mse_loss(preds[:, 2], labels[:, 2])
                
                loss = loss_type2 + loss_dfg + 0.5 * loss_affinity
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, checkpoint_path / 'best_model.pt')
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")
    
    return model
