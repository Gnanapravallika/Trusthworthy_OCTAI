import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.builder import build_model
from src.datasets.dataset import RetinalDataset
from src.preprocessing.standardizer import RetinalPipelineTransform
from src.losses.loss_functions import get_loss_function
from src.training.trainer import train_model

def run_smoke_test():
    print("[INFO] Starting End-to-End Smoke Test...")
    
    # 1. Setup temporary directory for synthetic images
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_smoke_data")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate synthetic images and records
    num_samples = 4
    records = []
    classes = ["cnv", "dme", "drusen", "normal"]
    
    print("Generating synthetic image B-scans...")
    for idx in range(num_samples):
        # Create random grayscale noise image
        img_arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        
        filename = f"scan_{idx}.jpg"
        file_path = os.path.join(temp_dir, filename)
        img.save(file_path)
        
        # Stratified mapping mock
        cls = classes[idx % len(classes)]
        records.append({
            "image_path": file_path,
            "label": idx % len(classes),
            "patient_id": f"Pat_{idx}"
        })
        
    df = pd.DataFrame(records)
    
    # 2. Mock configuration
    config = {
        "dataset": {
            "num_classes": 4
        },
        "model": {
            "backbone": "resnet50",
            "feature_module": "multiscale",
            "attention": "cbam",
            "dg": "mixstyle",
            "head": "evidential"
        }
    }
    
    print("Building modular evidential model from configurations...")
    model = build_model(config)
    
    # 3. Setup Dataset loaders
    transform = RetinalPipelineTransform(is_training=True)
    train_dataset = RetinalDataset(
        df=df, 
        transform=transform, 
        apply_bilateral=False, # Speed up for tests
        apply_clahe=False,
        apply_min_max=False
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    val_dataset = RetinalDataset(
        df=df, 
        transform=transform, 
        apply_bilateral=False,
        apply_clahe=False,
        apply_min_max=False
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # 4. Losses & Training setup
    criterion = get_loss_function("evidential", num_classes=4)
    output_dir = os.path.join(temp_dir, "outputs")
    
    print("Running 1 epoch training smoke test...")
    # Run 1 epoch
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=1,
        lr=0.0001,
        weight_decay=0.0001,
        patience=2,
        mixed_precision=False, # Disable to speed up cpu verification
        device_name="cpu",
        output_dir=output_dir,
        is_evidential=True
    )
    
    # 5. Check outputs
    weights_path = os.path.join(output_dir, "weights_best.pth")
    metrics_path = os.path.join(output_dir, "metrics.csv")
    
    assert os.path.exists(weights_path), "[ERROR] Best weights checkpoint was not saved!"
    assert os.path.exists(metrics_path), "[ERROR] Metrics log file was not saved!"
    
    print("[OK] Smoke test training artifacts saved correctly!")
    
    # Cleanup
    print("Cleaning up temporary data...")
    shutil.rmtree(temp_dir)
    print("[SUCCESS] Smoke test completed successfully!")

if __name__ == "__main__":
    run_smoke_test()
