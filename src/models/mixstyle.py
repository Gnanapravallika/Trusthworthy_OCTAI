import torch
import torch.nn as nn
import numpy as np

class MixStyle(nn.Module):
    """
    MixStyle: Domain Generalization by mixing instance statistics.
    References: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        if B <= 1:
            return x  # Cannot mix statistics with a batch size of 1
            
        # Compute mean and standard deviation across spatial dimensions (H, W)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        
        # Normalize features to zero mean and unit variance
        x_norm = (x - mu) / sig
        
        # Generate random permutation of batch indices
        perm = torch.randperm(B, device=x.device)
        
        # Draw mix coefficient lambda from Beta distribution
        lmda = np.random.beta(self.alpha, self.alpha)
        lmda = torch.tensor(lmda, dtype=x.dtype, device=x.device)
        
        # Mix the statistics
        mu_mix = lmda * mu + (1 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1 - lmda) * sig[perm]
        
        # Scale and shift the normalized features with the mixed statistics
        return x_norm * sig_mix + mu_mix
