import torch
import torch.nn as nn

class CORALAlignment(nn.Module):
    """
    Optional experimental baseline CORAL (Correlation Alignment) module. 
    CORAL alignment is optimized during training by minimizing the distance between
    the covariance matrices of source and target features. This block acts as an identity
    layer in the forward pass but facilitates extracting features for CORAL loss.
    It is provided for comparative domain adaptation baseline runs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def coral_covariance(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the covariance matrix for a batch of features.
    x shape: (B, D) where B is batch size and D is feature dimension.
    """
    n = x.size(0)
    if n <= 1:
        return torch.zeros((x.size(1), x.size(1)), device=x.device, dtype=x.dtype)
        
    # Subtract mean
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    
    # Compute covariance: C = (1 / (n - 1)) * X^T * X
    cov = torch.matmul(x_centered.t(), x_centered) / (n - 1)
    return cov

def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """
    Computes the CORAL loss (squared Frobenius norm distance between covariances).
    """
    d = source_features.size(1)
    
    source_cov = coral_covariance(source_features)
    target_cov = coral_covariance(target_features)
    
    # Frobenius norm: sum of squared differences
    loss = torch.sum((source_cov - target_cov) ** 2)
    loss = loss / (4 * d * d)
    return loss
