import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.datasets.dataset import RetinalDataset
from src.preprocessing.standardizer import RetinalPipelineTransform
from src.models.edl_head import get_evidence_metrics
from src.evaluation.classification import evaluate_classification_metrics
from src.evaluation.calibration import calculate_ece, calculate_brier_score

@torch.no_grad()
def run_external_validation(
    model: torch.nn.Module,
    df_external: pd.DataFrame,
    batch_size: int = 32,
    apply_bilateral: bool = True,
    apply_clahe: bool = False,
    apply_min_max: bool = False,
    is_evidential: bool = True,
    device_name: str = "cuda"
) -> dict:
    """
    Runs model inference on an external cohort to evaluate generalizability under domain shift.
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    # Setup dataset and dataloader
    transform = RetinalPipelineTransform(is_training=False)
    dataset = RetinalDataset(
        df=df_external, 
        transform=transform, 
        apply_bilateral=apply_bilateral, 
        apply_clahe=apply_clahe, 
        apply_min_max=apply_min_max
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_probabilities = []
    all_uncertainties = []
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        if is_evidential:
            # outputs = alpha parameters of Dirichlet distribution
            probs, uncertainties = get_evidence_metrics(outputs)
            probs = probs.cpu().numpy()
            uncertainties = uncertainties.cpu().numpy()
            
            preds = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            
            all_preds.extend(preds)
            all_confidences.extend(confidences)
            all_probabilities.extend(probs)
            all_uncertainties.extend(uncertainties)
        else:
            # outputs = logits
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            
            all_preds.extend(preds)
            all_confidences.extend(confidences)
            all_probabilities.extend(probs)
            # Softmax models do not have built-in epistemic uncertainty (set to zero or 1-conf)
            all_uncertainties.extend(1.0 - confidences)
            
        all_labels.extend(labels.numpy())
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_probabilities = np.array(all_probabilities)
    all_uncertainties = np.array(all_uncertainties)
    
    # Calculate performance and calibration metrics
    perf = evaluate_classification_metrics(all_labels, all_preds)
    
    # Calculate calibration
    accuracies = (all_preds == all_labels).astype(int)
    ece = calculate_ece(all_confidences, accuracies)
    brier = calculate_brier_score(all_probabilities, all_labels)
    
    results = {
        "predictions": all_preds,
        "labels": all_labels,
        "confidences": all_confidences,
        "probabilities": all_probabilities,
        "uncertainties": all_uncertainties,
        "metrics": {
            **perf,
            "ece": ece,
            "brier_score": brier
        }
    }
    
    return results
