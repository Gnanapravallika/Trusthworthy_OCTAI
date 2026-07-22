import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.coral import coral_loss

class EvidentialDirichletLoss(nn.Module):
    """
    Evidential Dirichlet Loss with KL Annealing for Evidential Deep Learning.
    References: Sensoy et al., "Evidential Deep Learning on Joint Predictions", NeurIPS 2018.
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes

    def kl_divergence(self, alpha_tilde: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence between Dirichlet(alpha_tilde) and Dirichlet(1).
        """
        device = alpha_tilde.device
        beta = torch.ones((1, self.num_classes), dtype=torch.float32, device=device)
        
        sum_alpha = torch.sum(alpha_tilde, dim=1, keepdim=True)
        sum_beta = torch.sum(beta, dim=1, keepdim=True)
        
        ln_gamma_sum_alpha = torch.lgamma(sum_alpha)
        ln_gamma_sum_beta = torch.lgamma(sum_beta)
        
        sum_ln_gamma_alpha = torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        sum_ln_gamma_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
        
        # Digamma terms
        digamma_alpha = torch.digamma(alpha_tilde)
        digamma_sum_alpha = torch.digamma(sum_alpha)
        
        kl = (ln_gamma_sum_alpha - sum_ln_gamma_alpha) - (ln_gamma_sum_beta - sum_ln_gamma_beta) + \
             torch.sum((alpha_tilde - beta) * (digamma_alpha - digamma_sum_alpha), dim=1, keepdim=True)
             
        return kl.squeeze(1)

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch: int, kl_annealing_epochs: int = 10) -> torch.Tensor:
        """
        Args:
            alpha: Dirichlet parameters of shape (B, K)
            target: Ground truth labels (B,) with integer values in [0, K-1]
            epoch: Current training epoch (0-indexed)
            kl_annealing_epochs: Epoch duration for scaling the KL regularization term to 1.0
        """
        device = alpha.device
        B = alpha.size(0)
        
        # Convert target integers to one-hot encoding
        y = F.one_hot(target, num_classes=self.num_classes).float()
        
        # Compute Dirichlet strength S
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S
        
        # 1. Mean Squared Error (MSE) / Likelihood term
        # Measures the prediction accuracy and variance
        error_term = torch.sum((y - p) ** 2, dim=1, keepdim=True)
        variance_term = torch.sum(p * (1.0 - p) / (S + 1.0), dim=1, keepdim=True)
        mse_loss = error_term + variance_term
        
        # 2. KL Divergence Regularization term
        # Penalizes evidence on incorrect classes by driving them towards alpha=1 (uniform distribution)
        alpha_tilde = y + (1.0 - y) * alpha
        kl_loss = self.kl_divergence(alpha_tilde)
        
        # Compute annealing coefficient lambda_t
        # Linearly increases from 0.0 to 1.0 over kl_annealing_epochs
        if kl_annealing_epochs > 0:
            annealing_coef = min(1.0, float(epoch) / float(kl_annealing_epochs))
        else:
            annealing_coef = 1.0
            
        loss = mse_loss.squeeze(1) + annealing_coef * kl_loss
        return torch.mean(loss)

def get_loss_function(name: str = "evidential", num_classes: int = 7) -> nn.Module:
    """
    Factory function to retrieve the configured loss module.
    """
    name = name.lower()
    if name == "evidential" or name == "edl":
        return EvidentialDirichletLoss(num_classes=num_classes)
    elif name == "cross_entropy" or name == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss name: {name}")
