import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

def compute_multiclass_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    """
    Computes macro-averaged Specificity (True Negative Rate) across all classes.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    
    for i in range(num_classes):
        # True Negatives (TN): sum of all elements in CM except row i and column i
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
        
    return float(np.mean(specificities))

def evaluate_classification_metrics(y_true: list, y_pred: list, num_classes: int = 4) -> dict:
    """
    Computes a complete suite of diagnostic performance metrics.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    specificity = compute_multiclass_specificity(y_true_arr, y_pred_arr, num_classes=num_classes)
    kappa = cohen_kappa_score(y_true_arr, y_pred_arr)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),          # Sensitivity
        "specificity": float(specificity),
        "f1_score": float(f1),
        "cohens_kappa": float(kappa)
    }
