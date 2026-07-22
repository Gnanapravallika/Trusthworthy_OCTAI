import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, save_path: str):
    """
    Generates and saves a beautiful seaborn heatmap confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle division by zero
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=True,
        square=True
    )
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("True Diagnosis Label", fontsize=12, labelpad=10)
    plt.xlabel("Predicted Diagnosis Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_reliability_diagram(confidences: np.ndarray, accuracies: np.ndarray, num_bins = 10, save_path: str = None):
    """
    Generates and saves a reliability diagram for calibration diagnostics.
    """
    # Robust parameter handling if 3rd positional argument is save_path string
    if isinstance(num_bins, str):
        save_path = num_bins
        num_bins = 10

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    # Handle 2D probability matrix if passed directly
    if confidences.ndim == 2:
        preds = np.argmax(confidences, axis=1)
        accuracies = (preds == accuracies).astype(int)
        confidences = np.max(confidences, axis=1)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
            bin_sizes.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_sizes.append(0)
            
    plt.figure(figsize=(6, 6))
    
    # Perfect calibration diagonal reference line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Plot empirical calibration bars
    plt.bar(
        bin_boundaries[:-1], 
        bin_accs, 
        width=1.0/num_bins, 
        align="edge", 
        color="#1f77b4", 
        edgecolor="black", 
        alpha=0.8,
        label="Trained Model"
    )
    
    # Gap/miscalibration visualization
    for i in range(num_bins):
        if bin_sizes[i] > 0 and bin_confs[i] > bin_accs[i]:
            plt.bar(
                bin_boundaries[i], 
                bin_confs[i] - bin_accs[i], 
                width=1.0/num_bins, 
                bottom=bin_accs[i], 
                align="edge", 
                color="#d62728", 
                edgecolor="black", 
                alpha=0.3,
                hatch="//"
            )
            
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence (Predicted Probability)", fontsize=12, labelpad=10)
    plt.ylabel("Accuracy", fontsize=12, labelpad=10)
    plt.title("Reliability Diagram (Calibration)", fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
