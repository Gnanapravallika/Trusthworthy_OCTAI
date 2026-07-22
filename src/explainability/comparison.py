import torch
import numpy as np

def calculate_saliency_entropy(cam: torch.Tensor) -> float:
    """
    Computes Saliency Entropy to quantify visual focus.
    Lower entropy indicates localized attention (sharp biomarker focus);
    higher entropy indicates scattered focus (attribution drift, background noise).
    """
    cam_normalized = cam.clone()
    cam_sum = cam_normalized.sum()
    
    if cam_sum > 0:
        cam_normalized = cam_normalized / cam_sum
    else:
        return 0.0
        
    cam_flat = cam_normalized.flatten()
    non_zero = cam_flat[cam_flat > 0]
    
    entropy = -torch.sum(non_zero * torch.log(non_zero)).item()
    return float(entropy)

def run_deletion_test(
    model: torch.nn.Module, 
    input_tensor: torch.Tensor, 
    cam: torch.Tensor, 
    class_idx: int, 
    steps: int = 10
) -> tuple:
    """
    Performs the quantitative deletion test by progressively masking high-attribution
    pixels in 10% steps and measuring the model's confidence drop.
    Returns: (list_of_confidences, aopc_score, percentage_drop)
    """
    model.eval()
    device = input_tensor.device
    
    # Get baseline classification confidence
    with torch.no_grad():
        outputs = model(input_tensor)
        # Check if evidential (returns alpha) or softmax (returns logits)
        if outputs.min() >= 1.0 and outputs.sum() > outputs.size(0):  # alpha check
            probs = outputs / torch.sum(outputs, dim=1, keepdim=True)
        else:
            probs = torch.softmax(outputs, dim=1)
        baseline_conf = probs[0, class_idx].item()
        
    # Flatten CAM and sort pixel indices descending
    cam_flat = cam.flatten()
    sorted_indices = torch.argsort(cam_flat, descending=True)
    
    total_pixels = cam_flat.numel()
    confidence_scores = [baseline_conf]
    
    # Progressive deletion loop
    for step in range(1, steps + 1):
        fraction = step / steps
        num_mask = int(total_pixels * fraction)
        
        perturbed_tensor = input_tensor.clone()
        c, h, w = perturbed_tensor.shape[1], perturbed_tensor.shape[2], perturbed_tensor.shape[3]
        
        # Create mask: set top pixels to 0
        mask = torch.ones(h * w, dtype=torch.bool, device=device)
        mask[sorted_indices[:num_mask]] = False
        mask = mask.view(h, w)
        
        for channel in range(c):
            perturbed_tensor[0, channel] = perturbed_tensor[0, channel] * mask
            
        with torch.no_grad():
            perturbed_out = model(perturbed_tensor)
            if perturbed_out.min() >= 1.0 and perturbed_out.sum() > perturbed_out.size(0):
                perturbed_probs = perturbed_out / torch.sum(perturbed_out, dim=1, keepdim=True)
            else:
                perturbed_probs = torch.softmax(perturbed_out, dim=1)
            perturbed_conf = perturbed_probs[0, class_idx].item()
            
        confidence_scores.append(perturbed_conf)
        
    # Calculate Deletion AOPC
    diffs = [baseline_conf - confidence_scores[i] for i in range(1, len(confidence_scores))]
    aopc_score = sum(diffs) / (steps + 1)
    
    percentage_drop = (baseline_conf - confidence_scores[-1]) / baseline_conf if baseline_conf > 0 else 0.0
    
    return confidence_scores, float(aopc_score), float(percentage_drop)

def run_insertion_test(
    model: torch.nn.Module, 
    input_tensor: torch.Tensor, 
    cam: torch.Tensor, 
    class_idx: int, 
    steps: int = 10
) -> tuple:
    """
    Performs the quantitative insertion test by starting with a blank canvas
    and progressively inserting the highest-attribution pixels in 10% steps.
    Returns: (list_of_confidences, aopc_score, final_confidence)
    """
    model.eval()
    device = input_tensor.device
    
    cam_flat = cam.flatten()
    sorted_indices = torch.argsort(cam_flat, descending=True)
    
    total_pixels = cam_flat.numel()
    
    # Baseline: Fully masked image (all zeros)
    perturbed_tensor = input_tensor.clone()
    c, h, w = perturbed_tensor.shape[1], perturbed_tensor.shape[2], perturbed_tensor.shape[3]
    for channel in range(c):
        perturbed_tensor[0, channel] = torch.zeros(h, w, device=device)
        
    with torch.no_grad():
        perturbed_out = model(perturbed_tensor)
        if perturbed_out.min() >= 1.0 and perturbed_out.sum() > perturbed_out.size(0):
            perturbed_probs = perturbed_out / torch.sum(perturbed_out, dim=1, keepdim=True)
        else:
            perturbed_probs = torch.softmax(perturbed_out, dim=1)
        baseline_conf = perturbed_probs[0, class_idx].item()
        
    confidence_scores = [baseline_conf]
    
    # Progressive insertion loop
    for step in range(1, steps + 1):
        fraction = step / steps
        num_insert = int(total_pixels * fraction)
        
        perturbed_tensor = input_tensor.clone()
        
        # Create mask: set only top pixels to True
        mask = torch.zeros(h * w, dtype=torch.bool, device=device)
        mask[sorted_indices[:num_insert]] = True
        mask = mask.view(h, w)
        
        for channel in range(c):
            perturbed_tensor[0, channel] = perturbed_tensor[0, channel] * mask
            
        with torch.no_grad():
            perturbed_out = model(perturbed_tensor)
            if perturbed_out.min() >= 1.0 and perturbed_out.sum() > perturbed_out.size(0):
                perturbed_probs = perturbed_out / torch.sum(perturbed_out, dim=1, keepdim=True)
            else:
                perturbed_probs = torch.softmax(perturbed_out, dim=1)
            perturbed_conf = perturbed_probs[0, class_idx].item()
            
        confidence_scores.append(perturbed_conf)
        
    # Calculate Insertion AOPC
    diffs = [confidence_scores[i] - baseline_conf for i in range(1, len(confidence_scores))]
    aopc_score = sum(diffs) / (steps + 1)
    
    return confidence_scores, float(aopc_score), confidence_scores[-1]
