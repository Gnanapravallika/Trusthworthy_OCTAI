import cv2
import numpy as np

def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    Applies an edge-preserving Bilateral Filter to suppress speckle noise in OCT scans
    while preserving sharp boundaries of the hyperreflective retinal layers.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def clahe_filter(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    to standardize local contrast variations across different scanner brands.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def min_max_scale(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the dynamic range of pixel values across scanner domains into [0.0, 1.0].
    """
    img_min = image.min()
    img_max = image.max()
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)
    return (image.astype(np.float32) - img_min) / (img_max - img_min)
