import os
import sys
import torch

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.builder import TrustOCTModel

def verify_compilation():
    print("[INFO] Starting Model Compilation Verification...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 1. Test standard Softmax baseline configuration
    print("Testing Softmax baseline configuration...")
    model_softmax = TrustOCTModel(
        backbone_name="resnet50",
        feature_module="identity",
        attention_module="identity",
        dg_module="identity",
        head_name="softmax",
        num_classes=4,
        pretrained=False
    ).to(device)
    
    # Run mock forward pass
    x = torch.randn(2, 3, 224, 224, device=device)
    y_softmax = model_softmax(x)
    assert y_softmax.shape == (2, 4), f"Expected shape (2, 4), got {y_softmax.shape}"
    
    # Backward pass check
    y_softmax.sum().backward()
    print("[OK] Softmax model compiled and backpropagated successfully!")
    
    # 2. Test proposed TrustOCT evidential configuration
    print("Testing Evidential TrustOCT configuration...")
    model_evidential = TrustOCTModel(
        backbone_name="resnet50",
        feature_module="multiscale",
        attention_module="cbam",
        dg_module="mixstyle",
        head_name="evidential",
        num_classes=4,
        pretrained=False
    ).to(device)
    
    # Run mock forward pass
    y_evidential = model_evidential(x)
    assert y_evidential.shape == (2, 4), f"Expected shape (2, 4), got {y_evidential.shape}"
    assert torch.all(y_evidential >= 1.0), "Dirichlet parameters alpha must be >= 1.0"
    
    # Backward pass check
    y_evidential.sum().backward()
    print("[OK] Evidential model compiled and backpropagated successfully!")
    
    print("[SUCCESS] All model configurations compiled and validated successfully!")

if __name__ == "__main__":
    verify_compilation()
