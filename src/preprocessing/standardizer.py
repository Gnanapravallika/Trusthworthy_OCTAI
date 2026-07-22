import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from src.preprocessing.filters import bilateral_filter, clahe_filter, min_max_scale

class RetinalPipelineTransform:
    """
    Transforms and standardizes retinal OCT scans for model training and inference.
    Applies Resize, ToTensor, and optional training-time augmentations.
    """
    def __init__(self, is_training: bool = False, image_size: int = 224):
        self.is_training = is_training
        
        transform_list = [
            T.Resize((image_size, image_size)),
            T.ToTensor()  # Automatically normalizes pixel values to [0.0, 1.0]
        ]
        
        # Inject augmentations if in training mode
        if self.is_training:
            # Flips and rotations simulate scanner and patient alignment differences
            transform_list.insert(1, T.RandomHorizontalFlip(p=0.5))
            transform_list.insert(2, T.RandomRotation(degrees=15))
            transform_list.insert(3, T.ColorJitter(brightness=0.15, contrast=0.15))
            
        self.transform_pipeline = T.Compose(transform_list)

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        return self.transform_pipeline(img_pil)

def load_and_preprocess_scan(
    file_path: str, 
    apply_bilateral: bool = True, 
    apply_clahe: bool = False, 
    apply_min_max: bool = False
) -> Image.Image:
    """
    Loads a single B-scan from disk and applies selected signal preprocessing filters.
    Returns a PIL Image converted to 3-channel RGB for PyTorch backbones.
    """
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {file_path}")
        
    if apply_bilateral:
        img = bilateral_filter(img)
        
    if apply_clahe:
        img = clahe_filter(img)
        
    if apply_min_max:
        norm_img = min_max_scale(img)
        img = (norm_img * 255).astype(np.uint8)
        
    # Convert grayscale to RGB to fit standard ImageNet-pretrained backbones
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb)
