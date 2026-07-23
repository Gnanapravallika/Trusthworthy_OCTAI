import torch
import torch.nn as nn
from src.models.backbone import get_backbone
from src.models.multiscale import MultiScaleFusion
from src.models.cbam import CBAM
from src.models.mixstyle import MixStyle
from src.models.coral import CORALAlignment
from src.models.softmax_head import SoftmaxHead
from src.models.edl_head import EvidentialHead

class TrustOCTModel(nn.Module):
    """
    Modular TrustOCT Network assembled dynamically like LEGO blocks.
    Configurable components:
      - Backbone (ResNet-50)
      - Feature Module (MultiScale Fusion)
      - Attention Module (CBAM)
      - Domain Generalization Module (MixStyle, CORAL)
      - Head (Softmax vs Evidential)
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        feature_module: str = "multiscale",
        attention_module: str = "cbam",
        dg_module: str = "mixstyle",
        head_name: str = "evidential",
        num_classes: int = 4,
        pretrained: bool = True
    ):
        super().__init__()
        # 1. Initialize Backbone
        self.backbone = get_backbone(backbone_name, pretrained=pretrained)
        in_planes = self.backbone.out_channels_l4  # Default 2048 for ResNet-50
        
        # 2. Initialize Feature Module
        self.feature_module_name = feature_module.lower()
        if self.feature_module_name == "multiscale":
            self.fusion = MultiScaleFusion(
                in_channels_l3=self.backbone.out_channels_l3,
                out_channels_l4=self.backbone.out_channels_l4
            )
        else:
            self.fusion = nn.Identity()
            
        # 3. Initialize Attention Module
        self.attention_module_name = attention_module.lower()
        if self.attention_module_name == "cbam":
            self.attention = CBAM(in_planes)
        else:
            self.attention = nn.Identity()
            
        # 4. Initialize Domain Generalization Module (MixStyle applied at mid-level x3)
        self.dg_module_name = dg_module.lower()
        if self.dg_module_name == "mixstyle":
            self.dg = MixStyle(p=0.5, alpha=0.1)
        elif self.dg_module_name == "coral":
            self.dg = CORALAlignment()
        else:
            self.dg = nn.Identity()
            
        # 5. Initialize Classifier Head
        self.head_name = head_name.lower()
        if self.head_name == "evidential":
            self.head = EvidentialHead(in_features=in_planes, num_classes=num_classes)
        else:
            self.head = SoftmaxHead(in_features=in_planes, num_classes=num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # 1. Feature Extraction (x3: 1024-ch, x4: 2048-ch)
        x3, x4 = self.backbone(x)
        
        # 2. Domain Generalization: Apply MixStyle at mid-level (x3) to mix scanner style statistics
        if self.dg_module_name == "mixstyle":
            x3 = self.dg(x3)
            
        # 3. Multi-Scale Feature Fusion
        if self.feature_module_name == "multiscale":
            x_fused = self.fusion(x3, x4)
        else:
            x_fused = x4
            
        # 4. Attention gating
        x_att = self.attention(x_fused)
        
        # 5. Classification Head
        out = self.head(x_att)
        
        if return_features:
            feat = nn.AdaptiveAvgPool2d(1)(x_att)
            feat = torch.flatten(feat, 1)
            return out, feat
            
        return out

def build_model(config: dict) -> nn.Module:
    """
    Assembles a TrustOCTModel from configuration dictionary.
    """
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    
    return TrustOCTModel(
        backbone_name=model_cfg.get("backbone", "resnet50"),
        feature_module=model_cfg.get("feature_module", "multiscale"),
        attention_module=model_cfg.get("attention", "cbam"),
        dg_module=model_cfg.get("dg", "mixstyle"),
        head_name=model_cfg.get("head", "evidential"),
        num_classes=dataset_cfg.get("num_classes", 4),
        pretrained=True
    )
