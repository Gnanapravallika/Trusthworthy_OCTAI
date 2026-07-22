import torch
import torch.nn as nn

class SoftmaxHead(nn.Module):
    """
    Standard classifier head. Applies Global Average Pooling, Dropout,
    and a Linear classification layer. Returns raw logits.
    """
    def __init__(self, in_features: int = 2048, num_classes: int = 4, dropout_rate: float = 0.5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
