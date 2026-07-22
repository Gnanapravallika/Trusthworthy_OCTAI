import torch
import torch.nn as nn

class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Feature Fusion block. Downsamples and projects shallower high-resolution
    features (Layer 3: 1024x14x14) to match deeper low-resolution features (Layer 4: 2048x7x7)
    and fuses them using element-wise addition.
    """
    def __init__(self, in_channels_l3: int = 1024, out_channels_l4: int = 2048):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels_l3, out_channels_l4, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels_l4)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        x3_down = self.fusion_conv(x3)
        return self.relu(x4 + x3_down)
