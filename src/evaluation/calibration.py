import numpy as np

def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE) across confidence bins.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n_samples = len(confidences)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select samples falling into the current bin
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)

def calculate_brier_score(probabilities: np.ndarray, targets: np.ndarray, num_classes: int = 7) -> float:
    """
    Computes the quadratic Brier Score for multi-class predictions.
    BS = (1/N) * sum_i (sum_k (p_ik - y_ik)^2)
    """
    N = len(targets)
    if N == 0:
        return 0.0
        
    # Convert targets to one-hot encoding
    y_one_hot = np.eye(num_classes)[targets]
    
    # Compute mean squared error of predicted probabilities
    brier_score = np.sum((probabilities - y_one_hot) ** 2) / N
    return float(brier_score)
