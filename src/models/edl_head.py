import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialHead(nn.Module):
    """
    Evidential Deep Learning Head. Replaces standard Softmax with a Dirichlet-parameterized
    evidentiary classifier. Uses a Softplus activation to ensure non-negative evidence.
    """
    def __init__(self, in_features: int = 2048, num_classes: int = 7, dropout_rate: float = 0.5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            alpha: Dirichlet parameters shape (B, K) where alpha = evidence + 1
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        
        # Evidence must be non-negative. We use softplus to avoid flat zero gradients.
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        return alpha

def get_evidence_metrics(alpha: torch.Tensor) -> tuple:
    """
    Computes expected probabilities and epistemic uncertainty from Dirichlet parameters alpha.
    Args:
        alpha: tensor of shape (B, K)
    Returns:
        probabilities: tensor of shape (B, K)
        uncertainty: tensor of shape (B,)
    """
    K = alpha.size(1)
    S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength
    probabilities = alpha / S
    uncertainty = K / S.squeeze(1)
    return probabilities, uncertainty
