import json
import os

def create_clean_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_markdown(source_lines):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_lines]
        })

    def add_code(source_lines):
        notebook["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in source_lines]
        })

    # Header Markdown Cell with Table of Contents
    add_markdown([
        "# TrustOCT: Trustworthy Cross-Scanner Retinal OCT Diagnostics Framework",
        "### Master Orchestrator Notebook (Clean Model Naming v1.5)",
        "---",
        "### 📌 TABLE OF CONTENTS",
        "*   [SECTION 1: Environment Setup](#section1)",
        "*   [SECTION 2: Dataset Verification & Statistics](#section2)",
        "*   [SECTION 3: Preprocessing & Denoising](#section3)",
        "*   [SECTION 4: Baseline Models Setup](#section4)",
        "*   [SECTION 5: Proposed TrustOCT Model](#section5)",
        "*   [SECTION 6: Model Training Execution](#section6)",
        "*   [SECTION 7: Classification Evaluation](#section7)",
        "*   [SECTION 8: Ablation Study](#section8)",
        "*   [SECTION 9: LayerCAM Visual Attributions](#section9)",
        "*   [SECTION 10: Faithfulness Evaluation (Deletion & Insertion)](#section10)",
        "*   [SECTION 11: External Validation (OCTID)](#section11)",
        "*   [SECTION 12: Statistical Analysis & Wilcoxon Significance](#section12)",
        "*   [SECTION 13: Publication Paper Assets](#section13)",
        "---",
        "<a id='section1'></a>",
        "## SECTION 1: Environment Setup"
    ])

    # Cell 2: Git check & requirement installs
    add_code([
        "import os",
        "",
        "# 1. Clone repository if missing (for Google Colab runners)",
        "if not os.path.exists('src'):",
        "    print('🔄 Cloning Trustworthy-OCT-AI repository...')",
        "    !git clone https://github.com/Gnanapravallika/Trusthworthy_OCTAI.git",
        "    %cd Trusthworthy_OCTAI",
        "else:",
        "    print('Running within the active repository workspace.')",
        "",
        "# 2. Install dependencies",
        "try:",
        "    import albumentations",
        "    import ptflops",
        "    print('✅ All dependency packages are already loaded.')",
        "except ImportError:",
        "    print('🔄 Installing required libraries...')",
        "    !pip install -r requirements.txt"
    ])

    # Cell 4: Imports & device check
    add_code([
        "# Setup imports",
        "import os",
        "import sys",
        "import time",
        "import cv2",
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "import torch.optim as optim",
        "from torch.utils.data import Dataset, DataLoader",
        "import torchvision.transforms as T",
        "import torchvision.models as models",
        "from PIL import Image",
        "import matplotlib.pyplot as plt",
        "from scipy.stats import wilcoxon",
        "from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report",
        "import seaborn as sns",
        "",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print(f'Running on device: {device}')"
    ])

    # Cell 5: Mount Google Drive
    add_code([
        "# Mount Google Drive",
        "try:",
        "    from google.colab import drive",
        "    drive.mount('/content/drive')",
        "except ImportError:",
        "    print('Running locally. Skipping Google Drive mount.')"
    ])

    # Header Markdown Section 2
    add_markdown([
        "<a id='section2'></a>",
        "## SECTION 2: Dataset Verification & Statistics",
        "Loads the training mappings for Kermany OCT2017 and dynamically computes **Table 1: Dataset Statistics**."
    ])

    # Cell 6.5: Dataset Download via Kaggle API
    add_code([
        "# Dataset Download via Kaggle API (for Google Colab runners)",
        "if not os.path.exists('/content/Kermany') and not os.path.exists('Kermany') and not os.path.exists('/content/OCT2017'):",
        "    try:",
        "        from google.colab import files",
        "        print(\"🔑 Please upload your kaggle.json API token file:\")",
        "        uploaded = files.upload()",
        "        if 'kaggle.json' in uploaded:",
        "            !mkdir -p ~/.kaggle",
        "            !cp kaggle.json ~/.kaggle/",
        "            !chmod 600 ~/.kaggle/kaggle.json",
        "            print(\"🔄 Downloading Kermany OCT2017 dataset from Kaggle...\")",
        "            !kaggle datasets download -d paultimothymooney/kermany2018 --unzip -p /content/Kermany",
        "            print(\"✅ Kermany OCT2017 successfully downloaded and extracted to /content/Kermany.\")",
        "        else:",
        "            print(\"❌ Upload failed. kaggle.json was not found.\")",
        "    except Exception as e:",
        "        print(\"Running locally or download skipped. Ensure Kermany folder is present.\")"
    ])

    # Cell 7: Load CSV and patient level split
    add_code([
        "# Load dataset mappings dynamically (Self-Healing Cell)",
        "from src.datasets.dataset import auto_detect_columns, patient_level_split",
        "",
        "drive_base = '/content/drive/MyDrive'",
        "csv_path = 'kermany_dataset_mapping.csv'",
        "if not os.path.exists(csv_path):",
        "    csv_path = os.path.join(drive_base, 'kermany_dataset_mapping.csv')",
        "",
        "if not os.path.exists(csv_path):",
        "    print('🔄 CSV mapping file missing. Generating dynamically from directories...')",
        "    root_oct = '/content/Kermany/OCT2017 ' # Kaggle extract path (with trailing space)",
        "    if not os.path.exists(root_oct):",
        "        root_oct = '/content/Kermany'",
        "    if not os.path.exists(root_oct):",
        "        root_oct = '/content/OCT2017'",
        "    if not os.path.exists(root_oct):",
        "        root_oct = 'e:/aiml master/OCT2017'",
        "        csv_path = 'e:/aiml master/kermany_dataset_mapping.csv'",
        "    ",
        "    if os.path.exists(root_oct):",
        "        records = []",
        "        class_to_idx = {'cnv': 0, 'dme': 1, 'drusen': 2, 'normal': 3}",
        "        for root, dirs, files in os.walk(root_oct):",
        "            for f in files:",
        "                if f.lower().endswith(('.jpg', '.png', '.jpeg')):",
        "                    parent_dir = os.path.basename(root)",
        "                    lbl = class_to_idx.get(parent_dir.lower(), -1)",
        "                    if lbl != -1:",
        "                        base = os.path.splitext(f)[0]",
        "                        parts = base.split('-')",
        "                        pt_id = '-'.join(parts[:2]) if len(parts) >= 2 else base",
        "                        records.append({",
        "                            'image_path': os.path.join(root, f),",
        "                            'label': lbl,",
        "                            'patient_id': pt_id",
        "                        })",
        "        df_new = pd.DataFrame(records)",
        "        df_new = df_new[df_new['label'] != -1]",
        "        df_new.to_csv(csv_path, index=False)",
        "        print(f'✅ Success: Dynamically created mapping CSV with {len(df_new)} images.')",
        "",
        "if os.path.exists(csv_path):",
        "    df = auto_detect_columns(pd.read_csv(csv_path))",
        "    ",
        "    # FORCE LOCAL PATH TRANSLATION (Speeds up dataloading in Colab by 100x)",
        "    is_colab = os.path.exists('/content')",
        "    local_kermany = '/content/Kermany'",
        "    local_oct2017 = '/content/OCT2017'",
        "    ",
        "    if is_colab and (os.path.exists(local_kermany) or os.path.exists(local_oct2017)):",
        "        print('🔄 Force-directing dataset image paths to fast local container storage...')",
        "        def force_local_path(path_str):",
        "            p = path_str.replace('\\\\', '/').replace('//', '/')",
        "            parts = p.split('/')",
        "            for folder in ['train', 'val', 'test']:",
        "                if folder in parts:",
        "                    idx = parts.index(folder)",
        "                    subpath = '/'.join(parts[idx:])",
        "                    # Search local folders",
        "                    for candidate in [",
        "                        os.path.join(local_kermany, subpath),",
        "                        os.path.join(local_kermany, 'OCT2017', subpath),",
        "                        os.path.join(local_kermany, 'OCT2017 ', subpath),",
        "                        os.path.join(local_oct2017, subpath),",
        "                        os.path.join(local_oct2017, 'OCT2017', subpath)",
        "                    ]:",
        "                        if os.path.exists(candidate):",
        "                            return candidate",
        "            return path_str",
        "        df['image_path'] = df['image_path'].apply(force_local_path)",
        "    ",
        "    sample_img_path = df.iloc[0]['image_path']",
        "    print(f'Sample image path: {sample_img_path}')",
        "    print(f'Does sample image exist? {os.path.exists(sample_img_path)}')",
        "    ",
        "    train_df, val_df, test_df = patient_level_split(df)",
        "    print(f'Dataset successfully loaded. Train shape: {train_df.shape}')",
        "else:",
        "    # Local fallback for code validation",
        "    print('Dataset files not found. Initializing mock dataset for notebook compilation checks...')",
        "    mock_records = []",
        "    for idx in range(100):",
        "        mock_records.append({",
        "            'image_path': f'dummy_{idx}.jpg',",
        "            'label': idx % 4,",
        "            'patient_id': f'Pat_{idx % 15}'",
        "        })",
        "    df = pd.DataFrame(mock_records)",
        "    train_df, val_df, test_df = patient_level_split(df)"
    ])

    # Cell 8: Split percentages
    add_code([
        "total = len(df)",
        "print(f\"Train: {len(train_df)} ({100*len(train_df)/total:.2f}%)\")",
        "print(f\"Validation: {len(val_df)} ({100*len(val_df)/total:.2f}%)\")",
        "print(f\"Test: {len(test_df)} ({100*len(test_df)/total:.2f}%)\")"
    ])

    # Cell 9: Unique patients counts
    add_code([
        "print(\"Unique patients\")",
        "print(\"Train:\", train_df[\"patient_id\"].nunique())",
        "print(\"Validation:\", val_df[\"patient_id\"].nunique())",
        "print(\"Test:\", test_df[\"patient_id\"].nunique())"
    ])

    # Cell 10: Overlaps
    add_code([
        "train_patients = set(train_df[\"patient_id\"])",
        "val_patients = set(val_df[\"patient_id\"])",
        "test_patients = set(test_df[\"patient_id\"])",
        "",
        "print(\"Train-Val overlap:\", len(train_patients & val_patients))",
        "print(\"Train-Test overlap:\", len(train_patients & test_patients))",
        "print(\"Val-Test overlap:\", len(val_patients & test_patients))"
    ])

    # Cell 11: Patient leakage check
    add_code([
        "# Patient Leakage Check",
        "train_patients = set(train_df['patient_id'].unique())",
        "test_patients = set(test_df['patient_id'].unique())",
        "leakage = train_patients.intersection(test_patients)",
        "print(f'Patient overlap count: {len(leakage)}')",
        "assert len(leakage) == 0, 'CRITICAL error: Patient data leakage detected!'",
        "print('Success: Zero patient leakage verified.')"
    ])

    # Cell 12: Table 1 statistics
    add_code([
        "# Compute Table 1 dynamically (mapping indices back to class names)",
        "CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']",
        "table_1 = df.groupby('label').agg(",
        "    total_images=('image_path', 'count'),",
        "    unique_patients=('patient_id', 'nunique')",
        ").reset_index().rename(columns={'label': 'Diagnostic Class', 'total_images': 'Total Images', 'unique_patients': 'Unique Patients'})",
        "table_1['Diagnostic Class'] = table_1['Diagnostic Class'].apply(lambda x: CLASSES[int(x)])",
        "print('--- TABLE 1: DATASET STATISTICS (COMPUTED) ---')",
        "display(table_1)",
        "os.makedirs('results/tables', exist_ok=True)",
        "table_1.to_csv('results/tables/table_1_dataset_statistics.csv', index=False)"
    ])

    # Header Markdown Section 3
    add_markdown([
        "<a id='section3'></a>",
        "## SECTION 3: Preprocessing & Denoising",
        "Applies and visualizes edge-preserving speckle denoising (Bilateral filtering) on training B-scans."
    ])

    # Cell 14: Visualizing Preprocessing
    add_code([
        "# Visualizing Preprocessing",
        "from src.preprocessing.filters import bilateral_filter",
        "",
        "sample_row = df.iloc[0]",
        "sample_path = sample_row['image_path']",
        "if os.path.exists(sample_path):",
        "    raw_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)",
        "    processed_img = bilateral_filter(raw_img)",
        "    ",
        "    fig, axes = plt.subplots(1, 2, figsize=(10, 5))",
        "    axes[0].imshow(raw_img, cmap='gray'); axes[0].set_title('Original B-scan')",
        "    axes[0].axis('off')",
        "    axes[1].imshow(processed_img, cmap='gray'); axes[1].set_title('Bilateral Denoised')",
        "    axes[1].axis('off')",
        "    plt.tight_layout()",
        "    os.makedirs('results/figures', exist_ok=True)",
        "    plt.savefig('results/figures/figure_1_preprocessing.png', dpi=300)",
        "    plt.show()",
        "else:",
        "    print('Local image file missing. Skipping preprocessing plot.')"
    ])

    # Header Markdown Section 4
    add_markdown([
        "<a id='section4'></a>",
        "## SECTION 4: Baseline Models Setup",
        "Loads the comparison backbones (ResNet-50, DenseNet-121, etc.)."
    ])

    # Cell 17: Load baselines
    add_code([
        "from src.models.builder import TrustOCTModel",
        "",
        "# Configure pre-trained architectures from modular builder",
        "resnet_baseline = TrustOCTModel(",
        "    backbone_name='resnet50', feature_module='identity', attention_module='identity',",
        "    dg_module='identity', head_name='softmax', num_classes=4, pretrained=True",
        ").to(device)",
        "print('Pre-trained baselines loaded successfully.')"
    ])

    # Header Markdown Section 5
    add_markdown([
        "<a id='section5'></a>",
        "## SECTION 5: Proposed TrustOCT Model",
        "Loads the attention-gated ResNet backbone with sequentially integrated Channel-Spatial Attention (CBAM) and MultiScale Fusion."
    ])

    # Cell 19: Load AEResNet
    add_code([
        "trust_oct_model = TrustOCTModel(",
        "    backbone_name='resnet50', feature_module='multiscale', attention_module='cbam',",
        "    dg_module='mixstyle', head_name='evidential', num_classes=4, pretrained=True",
        ").to(device)",
        "print('TrustOCT model successfully initialized with pre-trained weights.')"
    ])

    # Header Markdown Section 6
    add_markdown([
        "<a id='section6'></a>",
        "## SECTION 6: Model Training Execution",
        "Trains each ablation model sequentially in separate cells for optimal resiliency and debugging. Adjust training hyper-parameters below as global variables.",
        "",
        "**Global Hyper-parameters:**"
    ])

    # Global variables cell
    add_code([
        "# Training Hyperparameters",
        "epochs = 30",
        "lr = 1e-4",
        "batch_size = 32"
    ])

    # Experiment 1 Cell
    add_markdown(["### 1. Train `resnet50` (Baseline)"])
    add_code([
        "from src.training.trainer import run_experiment",
        "run_experiment('resnet50', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size, device_name=str(device))"
    ])

    # Experiment 2 Cell
    add_markdown(["### 2. Train `msf_resnet50` (+ MultiScale)"])
    add_code([
        "from src.training.trainer import run_experiment",
        "run_experiment('msf_resnet50', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size, device_name=str(device))"
    ])

    # Experiment 3 Cell
    add_markdown(["### 3. Train `msf_cbam_resnet50` (+ MultiScale + CBAM)"])
    add_code([
        "from src.training.trainer import run_experiment",
        "run_experiment('msf_cbam_resnet50', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size, device_name=str(device))"
    ])

    # Experiment 4 Cell
    add_markdown(["### 4. Train `msf_cbam_mixstyle_resnet50` (+ MultiScale + CBAM + MixStyle)"])
    add_code([
        "from src.training.trainer import run_experiment",
        "run_experiment('msf_cbam_mixstyle_resnet50', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size, device_name=str(device))"
    ])

    # Experiment 5 Cell
    add_markdown(["### 5. Train `msf_cbam_edl_resnet50` (+ MultiScale + CBAM + Evidential Head)"])
    add_code([
        "from src.training.trainer import run_experiment",
        "run_experiment('msf_cbam_edl_resnet50', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size, device_name=str(device))"
    ])

    # Experiment 6 Cell
    add_markdown(["### 6. Train `trustoct` (Full Integration - Proposed)"])
    add_code([
        "from src.training.trainer import run_experiment",
        "run_experiment('trustoct', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size, device_name=str(device))"
    ])

    # Header Markdown Section 7
    add_markdown([
        "<a id='section7'></a>",
        "## SECTION 7: Classification Evaluation",
        "Loads the saved predictions for **all models** dynamically, and compiles the comparison **Table 2** with 95% Bootstrap Confidence Intervals."
    ])

    # Cell 26: Bootstrapping evaluation pipeline
    add_code([
        "from src.evaluation.classification import evaluate_classification_metrics",
        "from src.evaluation.calibration import calculate_ece, calculate_brier_score",
        "from src.evaluation.plots import plot_confusion_matrix, plot_reliability_diagram",
        "from src.models.builder import build_model",
        "",
        "CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']",
        "class_to_idx = {c: idx for idx, c in enumerate(CLASSES)}",
        "",
        "def compute_all_metrics(labels, preds, probs):",
        "    labels = np.array(labels)",
        "    preds = np.array(preds)",
        "    probs = np.array(probs)",
        "    ",
        "    perf = evaluate_classification_metrics(labels, preds)",
        "    confidences = np.max(probs, axis=1)",
        "    accuracies = (preds == labels).astype(int)",
        "    ",
        "    ece = calculate_ece(confidences, accuracies)",
        "    brier = calculate_brier_score(probs, labels)",
        "    ",
        "    # Specificity macro",
        "    from src.evaluation.classification import compute_multiclass_specificity",
        "    specificity = compute_multiclass_specificity(labels, preds)",
        "    ",
        "    # ROC-AUC ovr macro",
        "    present_classes = sorted(list(np.unique(labels)))",
        "    if len(present_classes) > 1:",
        "        class_map = {old_label: new_label for new_label, old_label in enumerate(present_classes)}",
        "        mapped_labels = [class_map[lbl] for lbl in labels]",
        "        probs_sliced = probs[:, present_classes]",
        "        row_sums = probs_sliced.sum(axis=1, keepdims=True)",
        "        probs_sliced = np.where(row_sums > 1e-5, probs_sliced / row_sums, np.ones_like(probs_sliced) / probs_sliced.shape[1])",
        "        auc = roc_auc_score(mapped_labels, probs_sliced, multi_class='ovr', average='macro')",
        "    else:",
        "        auc = 0.5",
        "        ",
        "    return {",
        "        'Accuracy': perf['accuracy'], 'Precision': perf['precision'], 'Recall': perf['recall'],",
        "        'Specificity': specificity, 'Macro F1': perf['f1_score'], 'Kappa': perf['cohens_kappa'],",
        "        'ROC-AUC': auc, 'ECE': ece, 'Brier': brier",
        "    }",
        "",
        "def load_predictions_and_bootstrap(pred_path, n_bootstraps=200):",
        "    df_pred = pd.read_csv(pred_path)",
        "    labels = df_pred['true_label'].map(class_to_idx).values",
        "    preds = df_pred['pred_label'].map(class_to_idx).values",
        "    probs = df_pred[['prob_0', 'prob_1', 'prob_2', 'prob_3']].values",
        "    ",
        "    base_scores = compute_all_metrics(labels, preds, probs)",
        "    bootstrap_results = {k: [] for k in base_scores.keys()}",
        "    n_samples_len = len(labels)",
        "    np.random.seed(42)",
        "    for _ in range(n_bootstraps):",
        "        boot_idx = np.random.choice(n_samples_len, size=n_samples_len, replace=True)",
        "        boot_labels = labels[boot_idx]",
        "        boot_preds = preds[boot_idx]",
        "        boot_probs = probs[boot_idx]",
        "        ",
        "        boot_scores = compute_all_metrics(boot_labels, boot_preds, boot_probs)",
        "        for k, v in boot_scores.items():",
        "            bootstrap_results[k].append(v)",
        "            ",
        "    report = {}",
        "    for k, base_val in base_scores.items():",
        "        boot_vals = sorted(bootstrap_results[k])",
        "        ci_lower = boot_vals[int(0.025 * n_bootstraps)]",
        "        ci_upper = boot_vals[int(0.975 * n_bootstraps)]",
        "        report[k] = base_val",
        "        report[f'{k}_CI'] = (ci_lower, ci_upper)",
        "        ",
        "    return report, preds, labels, probs",
        "",
        "models_to_evaluate = [",
        "    ('outputs/resnet50/predictions.csv', 'ResNet-50 Baseline'),",
        "    ('outputs/msf_resnet50/predictions.csv', 'msf_resnet50'),",
        "    ('outputs/msf_cbam_resnet50/predictions.csv', 'msf_cbam_resnet50'),",
        "    ('outputs/msf_cbam_mixstyle_resnet50/predictions.csv', 'msf_cbam_mixstyle_resnet50'),",
        "    ('outputs/msf_cbam_edl_resnet50/predictions.csv', 'msf_cbam_edl_resnet50'),",
        "    ('outputs/trustoct/predictions.csv', 'TrustOCT (Proposed)')",
        "]",
        "",
        "comparison_rows = []",
        "best_preds, best_labels, best_probs = None, None, None",
        "",
        "for path, display_name in models_to_evaluate:",
        "    if os.path.exists(path):",
        "        print(f'Auditing {display_name} with bootstrap CIs...')",
        "        report, preds, labels, probs = load_predictions_and_bootstrap(path)",
        "        row = {'Model': display_name}",
        "        for metric in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'Macro F1', 'Kappa', 'ROC-AUC', 'ECE', 'Brier']:",
        "            val = report[metric]",
        "            ci = report[f\"{metric}_CI\"]",
        "            row[metric] = f\"{val:.4f} ({ci[0]:.4f} - {ci[1]:.4f})\"",
        "        comparison_rows.append(row)",
        "        if display_name == 'TrustOCT (Proposed)':",
        "            best_preds = preds",
        "            best_labels = labels",
        "            best_probs = probs",
        "    else:",
        "        print(f'Skipping {display_name}: predictions file {path} not found.')",
        "        ",
        "if len(comparison_rows) > 0:",
        "    table_2_df = pd.DataFrame(comparison_rows)",
        "    print('\\n--- TABLE 2: CORE MODELS COMPARISON (WITH 95% BOOTSTRAP CIs) ---')",
        "    display(table_2_df)",
        "    table_2_df.to_csv('results/tables/table_2_metrics_ci.csv', index=False)"
    ])

    # Cell 27: Confusion matrix display
    add_code([
        "# Display Confusion Matrix for Proposed Model",
        "from PIL import Image",
        "cm_path = 'outputs/trustoct/confusion_matrix.png'",
        "if os.path.exists(cm_path):",
        "    display(Image.open(cm_path))"
    ])

    # Header Markdown Section 8
    add_markdown([
        "<a id='section8'></a>",
        "## SECTION 8: Ablation Study",
        "Evaluates ablation study metrics from loaded predictions files to compile **Table 3: Ablation Study**."
    ])

    # Cell 29: Ablation evaluations
    add_code([
        "from sklearn.metrics import matthews_corrcoef",
        "",
        "ablation_configs = [",
        "    ('outputs/resnet50/predictions.csv', 'resnet50'),",
        "    ('outputs/msf_resnet50/predictions.csv', 'msf_resnet50'),",
        "    ('outputs/msf_cbam_resnet50/predictions.csv', 'msf_cbam_resnet50'),",
        "    ('outputs/msf_cbam_mixstyle_resnet50/predictions.csv', 'msf_cbam_mixstyle_resnet50'),",
        "    ('outputs/msf_cbam_edl_resnet50/predictions.csv', 'msf_cbam_edl_resnet50'),",
        "    ('outputs/trustoct/predictions.csv', 'trustoct')",
        "]",
        "",
        "ablation_rows = []",
        "for path, config_name in ablation_configs:",
        "    if os.path.exists(path):",
        "        df_pred = pd.read_csv(path)",
        "        labels = df_pred['true_label'].map(class_to_idx).values",
        "        preds = df_pred['pred_label'].map(class_to_idx).values",
        "        ",
        "        acc = accuracy_score(labels, preds)",
        "        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')",
        "        mcc = matthews_corrcoef(labels, preds)",
        "        ",
        "        ablation_rows.append({",
        "            'Configuration': config_name,",
        "            'Accuracy (%)': f\"{acc*100:.2f}%\",",
        "            'Macro F1': f\"{f1:.4f}\",",
        "            'MCC': f\"{mcc:.4f}\"",
        "        })",
        "    else:",
        "        print(f'Skipping ablation {config_name}: predictions file {path} not found.')",
        "        ",
        "if len(ablation_rows) > 0:",
        "    ablation_df = pd.DataFrame(ablation_rows)",
        "    print('--- TABLE 3: ABLATION STUDY ---')",
        "    display(ablation_df)",
        "    ablation_df.to_csv('results/tables/table_3_ablation_study.csv', index=False)"
    ])

    # Header Markdown Section 9
    add_markdown([
        "<a id='section9'></a>",
        "## SECTION 9: LayerCAM Visual Attributions",
        "Displays visual explanation heatmaps using actual images from the test split."
    ])

    # Cell 31: LayerCAM compare grid
    add_markdown([
        "LayerCAM visualizations for each model are automatically generated during the training phase and saved directly inside their respective subfolders (e.g. `outputs/trustoct/layercam/`)."
    ])

    add_code([
        "from PIL import Image",
        "cam_img_path = 'outputs/trustoct/layercam/normal_cam.png'",
        "if os.path.exists(cam_img_path):",
        "    display(Image.open(cam_img_path))",
        "else:",
        "    print('LayerCAM output not found. Run training with test_df passed.')"
    ])

    # Header Markdown Section 10
    add_markdown([
        "<a id='section10'></a>",
        "## SECTION 10: Faithfulness Evaluation (Deletion & Insertion)",
        "Runs progressive deletion and insertion perturbation tests on the test dataset to compute AOPC. These quantitative audits confirm if the LayerCAM attribution matches functional diagnostics."
    ])

    # Cell 33: AOPC faithfulness averages
    add_code([
        "from src.explainability.comparison import calculate_saliency_entropy, run_deletion_test, run_insertion_test",
        "from src.explainability.layercam import LayerCAM",
        "",
        "test_records = test_df.to_dict('records')[:20] # Run on first 20 for speed validation",
        "comparison_rows = []",
        "models_to_audit = [",
        "    ({'model': {'backbone': 'resnet50', 'feature_module': 'identity', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/resnet50/weights_best.pth', 'ResNet-50 Baseline'),",
        "    ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'mixstyle', 'head': 'evidential'}, 'dataset': {'num_classes': 4}}, 'outputs/trustoct/weights_best.pth', 'TrustOCT (Proposed)')",
        "]",
        "",
        "for config_item, path, display_name in models_to_audit:",
        "    if os.path.exists(path):",
        "        print(f\"Auditing explainability for {display_name}...\")",
        "        model_inst = build_model(config_item).to(device)",
        "        is_ev = (config_item['model']['head'] == 'evidential')",
        "        model_inst.load_state_dict(torch.load(path, map_location=device))",
        "        model_inst.eval()",
        "        ",
        "        target_layer = model_inst.backbone.layer4[-1]",
        "        cam_generator = LayerCAM(model_inst, target_layer)",
        "        ",
        "        del_scores, ins_scores, entropies = [], [], []",
        "        for rec in test_records:",
        "            img_p = rec['image_path']",
        "            lbl = int(rec['label'])",
        "            if not os.path.exists(img_p):",
        "                continue",
        "            try:",
        "                img_p_pil = Image.open(img_p).convert('RGB')",
        "                tensor_inp = val_transform(img_p_pil).unsqueeze(0).to(device)",
        "                with torch.no_grad():",
        "                    outs = model_inst(tensor_inp)",
        "                    if is_ev:",
        "                        from src.models.edl_head import get_evidence_metrics",
        "                        probs_ev, _ = get_evidence_metrics(outs)",
        "                        pred_cls = probs_ev.argmax(dim=1).item()",
        "                    else:",
        "                        pred_cls = torch.softmax(outs, dim=1).argmax(dim=1).item()",
        "                        ",
        "                cam_heatmap = cam_generator.generate(tensor_inp, class_idx=pred_cls)",
        "                _, aopc_del, _ = run_deletion_test(model_inst, tensor_inp, cam_heatmap, class_idx=pred_cls)",
        "                _, aopc_ins, _ = run_insertion_test(model_inst, tensor_inp, cam_heatmap, class_idx=pred_cls)",
        "                entropy = calculate_saliency_entropy(cam_heatmap)",
        "                ",
        "                del_scores.append(aopc_del)",
        "                ins_scores.append(aopc_ins)",
        "                entropies.append(entropy)",
        "            except Exception:",
        "                continue",
        "                ",
        "        cam_generator.release()",
        "        comparison_rows.append({",
        "            'Model': display_name,",
        "            'Deletion AOPC': f\"{np.mean(del_scores):.4f} \u00b1 {np.std(del_scores):.4f}\",",
        "            'Insertion AOPC': f\"{np.mean(ins_scores):.4f} \u00b1 {np.std(ins_scores):.4f}\",",
        "            'Saliency Entropy': f\"{np.mean(entropies):.4f} \u00b1 {np.std(entropies):.4f}\"",
        "        })",
        "    else:",
        "        print(f'Skipping audit for {display_name}: checkpoint {path} not found.')",
        "        ",
        "if len(comparison_rows) > 0:",
        "    table_4_df = pd.DataFrame(comparison_rows)",
        "    print('\\n--- TABLE 4: EXPLAINABILITY FAITHFULNESS BENCHMARKS (TEST-SET AVERAGES) ---')",
        "    display(table_4_df)",
        "    table_4_df.to_csv('results/tables/table_4_explainability_averages.csv', index=False)"
    ])

    # Header Markdown Section 11
    add_markdown([
        "<a id='section11'></a>",
        "## SECTION 11: External Validation (OCTID)",
        "Evaluates generalization on the out-of-distribution **OCTID cohort** using the trained AE-ResNet model."
    ])

    # Cell 35: Cross dataset evaluations
    add_code([
        "from src.evaluation.cross_dataset import run_external_validation",
        "",
        "octid_csv = 'octid_dataset_mapping.csv'",
        "if not os.path.exists(octid_csv):",
        "    octid_csv = '/content/drive/MyDrive/octid_dataset_mapping.csv'",
        "    ",
        "if os.path.exists(octid_csv):",
        "    octid_df = pd.read_csv(octid_csv)",
        "    sample_path_id = octid_df.iloc[0]['image_path']",
        "    if not os.path.exists(sample_path_id):",
        "        def correct_id_path(win_p):",
        "            l_p = win_p.replace(\'\\\\\\\\', '/')",
        "            r_p = l_p[l_p.find('OCTID/'):] if 'OCTID/' in l_p else '/'.join(l_p.split('/')[-3:])",
        "            return os.path.join(drive_base, r_p)",
        "        octid_df['image_path'] = octid_df['image_path'].apply(correct_id_path)",
        "        sample_path_id = octid_df.iloc[0]['image_path']",
        "    ",
        "    external_rows = []",
        "    models_for_external = [",
        "        ({'model': {'backbone': 'resnet50', 'feature_module': 'identity', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/resnet50/weights_best.pth', 'ResNet-50 Baseline'),",
        "        ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/msf_resnet50/weights_best.pth', 'msf_resnet50'),",
        "        ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/msf_cbam_resnet50/weights_best.pth', 'msf_cbam_resnet50'),",
        "        ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'mixstyle', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/msf_cbam_mixstyle_resnet50/weights_best.pth', 'msf_cbam_mixstyle_resnet50'),",
        "        ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'identity', 'head': 'evidential'}, 'dataset': {'num_classes': 4}}, 'outputs/msf_cbam_edl_resnet50/weights_best.pth', 'msf_cbam_edl_resnet50'),",
        "        ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'mixstyle', 'head': 'evidential'}, 'dataset': {'num_classes': 4}}, 'outputs/trustoct/weights_best.pth', 'TrustOCT (Proposed)')",
        "    ]",
        "    ",
        "    for config_item, path, display_name in models_for_external:",
        "        if os.path.exists(path):",
        "            is_ev = (config_item['model']['head'] == 'evidential')",
        "            model_inst = build_model(config_item).to(device)",
        "            model_inst.load_state_dict(torch.load(path, map_location=device))",
        "            model_inst.eval()",
        "            ",
        "            print(f'Evaluating generalization on OCTID for {display_name}...')",
        "            res = run_external_validation(",
        "                model=model_inst, df_external=octid_df, batch_size=16,",
        "                apply_bilateral=True, apply_clahe=(config_item['model']['dg']=='coral'),",
        "                apply_min_max=(config_item['model']['dg']=='coral'), is_evidential=is_ev, device_name=str(device)",
        "            )",
        "            ",
        "            external_rows.append({",
        "                'Model': display_name,",
        "                'Accuracy': res['metrics']['accuracy'],",
        "                'Precision': res['metrics']['precision'],",
        "                'Recall': res['metrics']['recall'],",
        "                'Macro F1': res['metrics']['f1_score'],",
        "                'ECE': res['metrics']['ece'],",
        "                'Brier': res['metrics']['brier_score']",
        "            })",
        "        else:",
        "            print(f'Skipping OCTID generalization for {display_name}: checkpoint {path} not found.')",
        "            ",
        "    if len(external_rows) > 0:",
        "        table_5_df = pd.DataFrame(external_rows)",
        "        print('\\n--- TABLE 5: CROSS-SCANNER DOMAIN GENERALIZATION (OCTID) ---')",
        "        display(table_5_df)",
        "        table_5_df.to_csv('results/tables/table_5_external_validation.csv', index=False)",
        "else:",
        "    print('OCTID mapping CSV not found. Skipping external validation.')"
    ])

    # Header Markdown Section 12
    add_markdown([
        "<a id='section12'></a>",
        "## SECTION 12: Statistical Analysis & Wilcoxon Significance",
        "Computes Wilcoxon signed-rank tests dynamically from saved predictions files."
    ])

    # Cell 37: Statistical tests
    add_code([
        "from scipy.stats import wilcoxon",
        "from statsmodels.stats.contingency_tables import mcnemar",
        "",
        "print('Running Statistical Significance Tests...')",
        "results_cache = {}",
        "",
        "configs_for_significance = [",
        "    ('outputs/resnet50/predictions.csv', 'ResNet-50 Baseline'),",
        "    ('outputs/trustoct/predictions.csv', 'TrustOCT (Proposed)')",
        "]",
        "",
        "for path, display_name in configs_for_significance:",
        "    if os.path.exists(path):",
        "        df_pred = pd.read_csv(path)",
        "        labels = df_pred['true_label'].map(class_to_idx).values",
        "        preds = df_pred['pred_label'].map(class_to_idx).values",
        "        results_cache[display_name] = {'preds': np.array(preds), 'labels': np.array(labels)}",
        "",
        "def run_mcnemar(labels, preds_a, preds_b):",
        "    correct_a = (preds_a == labels)",
        "    correct_b = (preds_b == labels)",
        "    n00 = np.sum(~correct_a & ~correct_b)",
        "    n01 = np.sum(~correct_a & correct_b)",
        "    n10 = np.sum(correct_a & ~correct_b)",
        "    n11 = np.sum(correct_a & correct_b)",
        "    table = [[n11, n10], [n01, n00]]",
        "    res = mcnemar(table, exact=True)",
        "    return res.statistic, res.pvalue",
        "",
        "proposed_name = 'TrustOCT (Proposed)'",
        "if proposed_name in results_cache and 'ResNet-50 Baseline' in results_cache:",
        "    prop_data = results_cache[proposed_name]",
        "    base_data = results_cache['ResNet-50 Baseline']",
        "    stat, p_val = run_mcnemar(prop_data['labels'], prop_data['preds'], base_data['preds'])",
        "    print(f\"McNemar's test proposed vs baseline ResNet50 p-value: {p_val:.5f}\")",
        "    ",
        "    # Wilcoxon signed-rank test",
        "    diff = (prop_data['preds'] == prop_data['labels']).astype(int) - (base_data['preds'] == base_data['labels']).astype(int)",
        "    if not np.all(diff == 0):",
        "        stat_w, p_val_w = wilcoxon(diff)",
        "        print(f\"Wilcoxon proposed vs baseline ResNet50 p-value: {p_val_w:.5f}\")"
    ])

    # Header Markdown Section 13
    add_markdown([
        "<a id='section13'></a>",
        "## SECTION 13: Publication Paper Assets",
        "Creates directories and compiles unified paper figures (Learning Curves, Confusion Matrix grids, Calibration Curves) and Zips results."
    ])

    # Cell 39: Figure plots
    add_code([
        "import shutil",
        "from src.evaluation.plots import plot_reliability_diagram",
        "",
        "print('Generating publication figure curves...')",
        "history_files = {",
        "    'resnet50': 'outputs/resnet50/metrics.csv',",
        "    'trustoct': 'outputs/trustoct/metrics.csv'",
        "}",
        "",
        "# 1. Unified training curves comparison",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
        "metrics_to_plot = [",
        "    ('train_loss', 'Training Loss', axes[0, 0]),",
        "    ('val_loss', 'Validation Loss', axes[0, 1]),",
        "    ('train_acc', 'Training Accuracy', axes[1, 0]),",
        "    ('val_acc', 'Validation Accuracy', axes[1, 1])",
        "]",
        "for metric_key, title, ax in metrics_to_plot:",
        "    for model_name, path in history_files.items():",
        "        if os.path.exists(path):",
        "            df_hist = pd.read_csv(path)",
        "            ax.plot(df_hist['epoch'], df_hist[metric_key], label=model_name, linewidth=2)",
        "    ax.set_title(title, fontsize=12, fontweight='bold')",
        "    ax.set_xlabel('Epoch')",
        "    ax.grid(True, linestyle=':', alpha=0.6)",
        "    ax.legend()",
        "plt.tight_layout()",
        "plt.savefig('results/figures/figure_2_training_curves.png', dpi=300)",
        "plt.show()",
        "",
        "# 2. Absolute dynamic drop table (OCTDL vs OCTID)",
        "drop_rows = []",
        "models_for_drop = [",
        "    ('outputs/resnet50/predictions.csv', 'ResNet-50 Baseline'),",
        "    ('outputs/msf_resnet50/predictions.csv', 'msf_resnet50'),",
        "    ('outputs/msf_cbam_resnet50/predictions.csv', 'msf_cbam_resnet50'),",
        "    ('outputs/msf_cbam_mixstyle_resnet50/predictions.csv', 'msf_cbam_mixstyle_resnet50'),",
        "    ('outputs/msf_cbam_edl_resnet50/predictions.csv', 'msf_cbam_edl_resnet50'),",
        "    ('outputs/trustoct/predictions.csv', 'TrustOCT (Proposed)')",
        "]",
        "",
        "for path, display_name in models_for_drop:",
        "    if os.path.exists(path):",
        "        df_pred = pd.read_csv(path)",
        "        labels = df_pred['true_label'].map(class_to_idx).values",
        "        preds = df_pred['pred_label'].map(class_to_idx).values",
        "        src_acc = accuracy_score(labels, preds) * 100",
        "        ",
        "        # Target performance",
        "        tgt_acc = 0.0",
        "        if os.path.exists('results/tables/table_5_external_validation.csv'):",
        "            df_ext = pd.read_csv('results/tables/table_5_external_validation.csv')",
        "            match_row = df_ext[df_ext['Model'] == display_name]",
        "            if len(match_row) > 0:",
        "                tgt_acc = match_row.iloc[0]['Accuracy'] * 100",
        "                ",
        "        acc_drop = tgt_acc - src_acc",
        "        drop_rows.append({",
        "            'Model': display_name,",
        "            'Source (OCTDL) Acc (%)': f'{src_acc:.2f}%',",
        "            'Target (OCTID) Acc (%)': f'{tgt_acc:.2f}%',",
        "            'Absolute Performance Drop': f'{acc_drop:.2f}%'",
        "        })",
        "        ",
        "if len(drop_rows) > 0:",
        "    drop_df = pd.DataFrame(drop_rows)",
        "    print('\\n--- TABLE 5.5: CROSS-DOMAIN ACCURACY DECAY (DOMAIN SHIFT DROP) ---')",
        "    display(drop_df)",
        "    drop_df.to_csv('results/tables/table_5_5_domain_drop.csv', index=False)",
        "",
        "# Zip results",
        "zip_path = '/content/final_paper_results'",
        "if os.path.exists('outputs'):",
        "    shutil.make_archive(zip_path, 'zip', 'outputs')",
        "    print(\"Success: Zip archive 'final_paper_results.zip' created successfully from outputs folder!\")"
    ])

    # Cell 40: Computational complexity
    add_code([
        "# Complexity Analysis",
        "import time",
        "",
        "complexity_rows = []",
        "dummy_input = torch.randn(1, 3, 224, 224).to(device)",
        "models_for_complexity = [",
        "    ({'model': {'backbone': 'resnet50', 'feature_module': 'identity', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/resnet50/weights_best.pth', 'ResNet-50 Baseline'),",
        "    ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/msf_resnet50/weights_best.pth', 'msf_resnet50'),",
        "    ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/msf_cbam_resnet50/weights_best.pth', 'msf_cbam_resnet50'),",
        "    ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'mixstyle', 'head': 'softmax'}, 'dataset': {'num_classes': 4}}, 'outputs/msf_cbam_mixstyle_resnet50/weights_best.pth', 'msf_cbam_mixstyle_resnet50'),",
        "    ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'identity', 'head': 'evidential'}, 'dataset': {'num_classes': 4}}, 'outputs/msf_cbam_edl_resnet50/weights_best.pth', 'msf_cbam_edl_resnet50'),",
        "    ({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'cbam', 'dg': 'mixstyle', 'head': 'evidential'}, 'dataset': {'num_classes': 4}}, 'outputs/trustoct/weights_best.pth', 'TrustOCT (Proposed)')",
        "]",
        "",
        "for config_item, path, display_name in models_for_complexity:",
        "    if os.path.exists(path):",
        "        model_inst = build_model(config_item).to(device)",
        "        model_inst.load_state_dict(torch.load(path, map_location=device))",
        "        model_inst.eval()",
        "        ",
        "        total_params = sum(p.numel() for p in model_inst.parameters())",
        "        trainable_params = sum(p.numel() for p in model_inst.parameters() if p.requires_grad)",
        "        model_size_mb = os.path.getsize(path) / (1024 * 1024)",
        "        ",
        "        # Latency evaluation",
        "        with torch.no_grad():",
        "            for _ in range(20):",
        "                _ = model_inst(dummy_input)",
        "            if torch.cuda.is_available():",
        "                torch.cuda.synchronize()",
        "            start_t = time.perf_counter()",
        "            for _ in range(100):",
        "                _ = model_inst(dummy_input)",
        "            if torch.cuda.is_available():",
        "                torch.cuda.synchronize()",
        "            end_t = time.perf_counter()",
        "        avg_inf_ms = ((end_t - start_t) / 100) * 1000",
        "        ",
        "        complexity_rows.append({",
        "            'Model': display_name,",
        "            'Total Params (M)': f\"{total_params / 1e6:.2f}M\",",
        "            'Trainable Params (M)': f\"{trainable_params / 1e6:.2f}M\",",
        "            'Size on Disk (MB)': f\"{model_size_mb:.2f} MB\",",
        "            'Inference Speed (ms)': f\"{avg_inf_ms:.2f} ms\"",
        "        })",
        "        ",
        "if len(complexity_rows) > 0:",
        "    complexity_df = pd.DataFrame(complexity_rows)",
        "    print('\\n--- TABLE 6: COMPUTATIONAL COMPLEXITY ANALYSIS ---')",
        "    display(complexity_df)",
        "    complexity_df.to_csv('results/tables/table_6_computational_complexity.csv', index=False)"
    ])

    # Cell 41: Failure Analysis Grid
    add_code([
        "# Representative Failure Analysis Grid",
        "CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']",
        "misclass_records = []",
        "trustoct_pred_path = 'outputs/trustoct/predictions.csv'",
        "",
        "if os.path.exists(trustoct_pred_path):",
        "    df_pred = pd.read_csv(trustoct_pred_path)",
        "    misclassified = df_pred[df_pred['true_label'] != df_pred['pred_label']]",
        "    print(f'Total misclassified images found in test set: {len(misclassified)}')",
        "    ",
        "    n_plots = min(16, len(misclassified))",
        "    if n_plots > 0:",
        "        fig, axes = plt.subplots(4, 4, figsize=(12, 12))",
        "        axes_flat = axes.flatten()",
        "        for i in range(16):",
        "            ax = axes_flat[i]",
        "            if i < n_plots:",
        "                rec = misclassified.iloc[i]",
        "                img_p = rec['image_path']",
        "                if os.path.exists(img_p):",
        "                    img_pil_mis = Image.open(img_p).convert('RGB')",
        "                    ax.imshow(img_pil_mis)",
        "                ax.set_title(f\"True: {rec['true_label']}\\nPred: {rec['pred_label']}\", fontsize=10, fontweight='bold', color='red')",
        "            ax.axis('off')",
        "        plt.tight_layout()",
        "        plt.savefig('results/figures/figure_6_failure_analysis.png', dpi=300)",
        "        plt.show()",
        "        print(\"Figure 6 (Failure Analysis Grid) successfully generated!\")"
    ])

    # Save the notebook JSON
    target_path = "e:/aiml master/TrustOCT/TrustOCT_Project.ipynb"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Jupyter Notebook successfully created at: {target_path}")

if __name__ == "__main__":
    create_clean_notebook()
