"""
Evaluates all TrustOCT variants (trustoct, trustoct_expB, trustoct_expD, trustoct_expA)
across internal Kermany test set and external OCTID dataset to scientifically select the 🥇 Winning TrustOCT Model.
"""
import os, torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
from src.evaluation.calibration import calculate_ece, calculate_brier_score

CLASS_MAP = {
    'cnv': 0, 'dme': 1, 'drusen': 2, 'normal': 3,
    'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3,
    '0': 0, '1': 1, '2': 2, '3': 3,
    0: 0, 1: 1, 2: 2, 3: 3
}

def parse_col(series):
    return series.apply(lambda x: CLASS_MAP.get(str(x).lower().strip(), CLASS_MAP.get(x, 0))).values

def evaluate_variant(out_dir, display_name):
    pred_path = os.path.join(out_dir, 'predictions.csv')
    metrics_path = os.path.join(out_dir, 'metrics.csv')
    weights_path = os.path.join(out_dir, 'weights_best.pth')
    
    if not os.path.exists(pred_path):
        return None
        
    df_p = pd.read_csv(pred_path)
    y_true = parse_col(df_p['true_label'])
    y_pred = parse_col(df_p['pred_label'])
    probs = df_p[['prob_0', 'prob_1', 'prob_2', 'prob_3']].values
    u_arr = df_p['uncertainty'].values if 'uncertainty' in df_p.columns else None
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    conf = (1.0 - u_arr) * np.max(probs, axis=1) if u_arr is not None else np.max(probs, axis=1)
    ece = calculate_ece(conf, (y_pred == y_true).astype(int))
    brier = calculate_brier_score(probs, y_true)
    
    best_epoch, best_val_loss = 'N/A', 999.0
    if os.path.exists(metrics_path):
        try:
            df_m = pd.read_csv(metrics_path)
            if 'val_loss' in df_m.columns:
                min_idx = df_m['val_loss'].idxmin()
                min_row = df_m.loc[min_idx]
                best_epoch = int(min_row.get('epoch', 0))
                best_val_loss = float(min_row['val_loss'])
        except Exception:
            pass
            
    # Composite score for selection: (Macro F1 + Accuracy) - (ECE + Brier)
    selection_score = (f1 + acc) - (ece + brier)
    
    return {
        'Variant': display_name,
        'Dir': out_dir,
        'Best Epoch': best_epoch,
        'Best Val Loss': f"{best_val_loss:.4f}" if best_val_loss != 999.0 else 'N/A',
        'Accuracy (%)': f"{acc*100:.2f}%",
        'Macro F1': f"{f1:.4f}",
        'Kappa': f"{kappa:.4f}",
        'ECE (Calibration) ↓': f"{ece:.4f}",
        'Brier Score ↓': f"{brier:.4f}",
        'Selection Score ⭐': f"{selection_score:.4f}",
        'val_loss_num': best_val_loss,
        'f1_num': f1,
        'ece_num': ece
    }

print("Searching for TrustOCT variant evaluation outputs...")
variants = [
    ('outputs/trustoct', 'TrustOCT Baseline'),
    ('outputs/trustoct_expB', 'TrustOCT Exp B (kl=0.3)'),
    ('outputs/trustoct_expD', 'TrustOCT Exp D (anneal=20)'),
    ('outputs/trustoct_expA', 'TrustOCT Exp A (lr=5e-5)')
]

results = []
for d, name in variants:
    res = evaluate_variant(d, name)
    if res:
        results.append(res)

if len(results) > 0:
    df_res = pd.DataFrame(results)
    print("\n--- TRUSTOCT VARIANTS DECISION MATRIX ---")
    print(df_res.to_string(index=False))
else:
    print("No prediction CSVs found locally yet. Run in Colab to evaluate live!")
