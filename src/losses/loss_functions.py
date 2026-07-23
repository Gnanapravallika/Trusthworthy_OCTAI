import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialDirichletLoss(nn.Module):
    """
    Evidential Dirichlet Loss with KL Annealing for Evidential Deep Learning.
    References: Sensoy et al., "Evidential Deep Learning on Joint Predictions", NeurIPS 2018.
    """
    def __init__(self, num_classes: int = 4, kl_weight: float = 1.0, kl_annealing_epochs: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.kl_annealing_epochs = kl_annealing_epochs

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
        
        digamma_alpha = torch.digamma(alpha_tilde)
        digamma_sum_alpha = torch.digamma(sum_alpha)
        
        kl = (ln_gamma_sum_alpha - sum_ln_gamma_alpha) - (ln_gamma_sum_beta - sum_ln_gamma_beta) + \
             torch.sum((alpha_tilde - beta) * (digamma_alpha - digamma_sum_alpha), dim=1, keepdim=True)
             
        return kl.squeeze(1)

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch: int, kl_annealing_epochs: int = None, return_components: bool = False):
        """
        Args:
            alpha: Dirichlet parameters of shape (B, K)
            target: Ground truth labels (B,) with integer values in [0, K-1]
            epoch: Current training epoch (0-indexed)
            kl_annealing_epochs: Epoch duration for scaling the KL regularization term to 1.0
            return_components: If True, returns (total_loss, mean_mse_loss, mean_kl_loss)
        """
        if kl_annealing_epochs is None:
            kl_annealing_epochs = self.kl_annealing_epochs
            
        device = alpha.device
        B = alpha.size(0)
        
        y = F.one_hot(target, num_classes=self.num_classes).float()
        
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S
        
        error_term = torch.sum((y - p) ** 2, dim=1, keepdim=True)
        variance_term = torch.sum(p * (1.0 - p) / (S + 1.0), dim=1, keepdim=True)
        mse_loss = error_term + variance_term
        
        alpha_tilde = y + (1.0 - y) * alpha
        kl_loss = self.kl_divergence(alpha_tilde)
        
        if kl_annealing_epochs > 0:
            annealing_coef = min(1.0, float(epoch) / float(kl_annealing_epochs))
        else:
            annealing_coef = 1.0
            
        total_loss = mse_loss.squeeze(1) + (self.kl_weight * annealing_coef) * kl_loss
        mean_total = torch.mean(total_loss)
        
        if return_components:
            mean_mse = torch.mean(mse_loss.squeeze(1))
            mean_kl = torch.mean(kl_loss)
            return mean_total, mean_mse, mean_kl
            
        return mean_total

def get_loss_function(name: str = "evidential", num_classes: int = 4, kl_weight: float = 1.0, kl_annealing_epochs: int = 10) -> nn.Module:
    """
    Factory function to retrieve the configured loss module.
    """
    name = name.lower()
    if name in ["evidential", "edl"]:
        return EvidentialDirichletLoss(num_classes=num_classes, kl_weight=kl_weight, kl_annealing_epochs=kl_annealing_epochs)
    elif name in ["cross_entropy", "ce"]:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss name: {name}")
