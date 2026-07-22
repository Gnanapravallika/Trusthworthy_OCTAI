import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_ood_detection(id_uncertainties: np.ndarray, ood_uncertainties: np.ndarray) -> float:
    """
    Evaluates OOD detection performance by computing the AUROC score.
    Higher uncertainty scores should ideally separate OOD samples from ID samples.
    
    Args:
        id_uncertainties: Uncertainty scores for In-Distribution samples.
        ood_uncertainties: Uncertainty scores for Out-of-Distribution samples.
    Returns:
        auroc: AUROC score for OOD detection.
    """
    # Create target labels: 0 for ID, 1 for OOD
    y_true = np.concatenate([np.zeros(len(id_uncertainties)), np.ones(len(ood_uncertainties))])
    y_score = np.concatenate([id_uncertainties, ood_uncertainties])
    
    if len(np.unique(y_true)) < 2:
        # Cannot calculate AUROC with only one class represented
        return 0.5
        
    auroc = roc_auc_score(y_true, y_score)
    return float(auroc)

def apply_selective_prediction(predictions: np.ndarray, uncertainties: np.ndarray, threshold: float) -> tuple:
    """
    Applies uncertainty-aware clinical selective prediction.
    If uncertainty is >= threshold, the image is routed to clinical review (referred).
    Otherwise, the model's diagnostic prediction is returned.
    
    Returns:
        final_preds: Array where deferred predictions are marked with -1
        referral_rate: Fraction of images referred for manual clinical review
    """
    referred_indices = uncertainties >= threshold
    final_preds = predictions.copy()
    final_preds[referred_indices] = -1  # -1 represents referred/uncertain prediction
    
    referral_rate = np.mean(referred_indices)
    return final_preds, float(referral_rate)
