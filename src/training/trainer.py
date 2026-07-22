import os
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from src.models.edl_head import get_evidence_metrics

def enforce_seeds(seed: int = 42):
    """
    Locks all random seeds to guarantee reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """
    Halt training if validation loss does not decrease for a set patience.
    Restores the best weights at termination.
    """
    def __init__(self, patience: int = 5, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

def train_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device, 
    epoch: int,
    is_evidential: bool,
    scaler: GradScaler = None
) -> tuple:
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # GPU Mixed Precision Autocast
        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                if is_evidential:
                    loss = criterion(outputs, targets, epoch)
                else:
                    loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if is_evidential:
                loss = criterion(outputs, targets, epoch)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate training accuracy
        if is_evidential:
            # outputs = alpha parameters of Dirichlet distribution
            probs, _ = get_evidence_metrics(outputs)
            preds = torch.argmax(probs, dim=1)
        else:
            # outputs = logits
            preds = torch.argmax(outputs, dim=1)
            
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device, 
    epoch: int,
    is_evidential: bool
) -> tuple:
    """
    Evaluates the model on validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        if is_evidential:
            loss = criterion(outputs, targets, epoch)
            probs, _ = get_evidence_metrics(outputs)
            preds = torch.argmax(probs, dim=1)
        else:
            loss = criterion(outputs, targets)
            preds = torch.argmax(outputs, dim=1)
            
        running_loss += loss.item() * inputs.size(0)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    mixed_precision: bool,
    device_name: str,
    output_dir: str,
    is_evidential: bool = True
) -> dict:
    """
    Coordinates training, validation, early stopping, schedulers, and weight checkpoints.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)
    
    # 1. Initialize Optimizer (AdamW) and Cosine Annealing Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 2. Setup Mixed Precision and Early Stopping
    scaler = GradScaler() if (mixed_precision and device.type == "cuda") else None
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    history = []
    best_val_loss = float("inf")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, is_evidential, scaler
        )
        
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch, is_evidential
        )
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(
            f"Epoch [{epoch+1:02d}/{epochs:02d}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })
        
        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model checkpoint
            torch.save(model.state_dict(), os.path.join(output_dir, "weights_best.pth"))
            print(f"[BEST] Saved new best model checkpoint (Val Loss: {val_loss:.4f})")
            
        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("[EARLY STOP] Early stopping triggered. Restoration of best parameters...")
            model.load_state_dict(early_stopping.best_weights)
            # Re-save best weights locally
            torch.save(model.state_dict(), os.path.join(output_dir, "weights_best.pth"))
            break
            
    total_time = time.time() - start_time
    print(f"[INFO] Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s.")
    
    # Save training history log
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    return history

def run_experiment(
    experiment_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 32,
    device_name: str = 'cuda'
) -> dict:
    """
    Simpler master runner function to execute individual dissertation experiments cleanly.
    """
    from src.models.builder import build_model
    from src.preprocessing.standardizer import RetinalPipelineTransform
    from src.datasets.dataset import RetinalDataset
    from src.losses.loss_functions import get_loss_function

    # Map experiment IDs to structural configurations
    experiment_configs = {
        'resnet50': {'feature': 'identity', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'},
        'msf_resnet50': {'feature': 'multiscale', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'},
        'msf_cbam_resnet50': {'feature': 'multiscale', 'attention': 'cbam', 'dg': 'identity', 'head': 'softmax'},
        'msf_cbam_mixstyle_resnet50': {'feature': 'multiscale', 'attention': 'cbam', 'dg': 'mixstyle', 'head': 'softmax'},
        'msf_cbam_edl_resnet50': {'feature': 'multiscale', 'attention': 'cbam', 'dg': 'identity', 'head': 'evidential'},
        'trustoct': {'feature': 'multiscale', 'attention': 'cbam', 'dg': 'mixstyle', 'head': 'evidential'}
    }

    if experiment_id not in experiment_configs:
        raise ValueError(f"Unknown experiment ID: {experiment_id}. Must be one of {list(experiment_configs.keys())}")

    exp = experiment_configs[experiment_id]
    config = {
        'model': {
            'backbone': 'resnet50',
            'feature_module': exp['feature'],
            'attention': exp['attention'],
            'dg': exp['dg'],
            'head': exp['head']
        },
        'dataset': {
            'num_classes': 4
        }
    }

    # 1. Print configuration summary
    loss_name = 'Evidential Loss' if exp['head'] == 'evidential' else 'Cross Entropy'
    print('+' + '-'*40 + '+')
    print(f'| Experiment : {experiment_id:<27} |')
    print(f'| Backbone   : resnet50                     |')
    print(f'| Feature    : {exp["feature"]:<27} |')
    print(f'| Attention  : {exp["attention"]:<27} |')
    print(f'| DG         : {exp["dg"]:<27} |')
    print(f'| Head       : {exp["head"]:<27} |')
    print(f'| Loss       : {loss_name:<27} |')
    print(f'| Optimizer  : AdamW                       |')
    print(f'| Scheduler  : Cosine Annealing            |')
    print('+' + '-'*40 + '+')

    # 2. Setup Reproducibility & Build Model
    enforce_seeds(42)
    model = build_model(config)
    is_evidential = (exp['head'] == 'evidential')
    criterion = get_loss_function('evidential' if is_evidential else 'cross_entropy', num_classes=4)

    # 3. Setup Dataset Loaders
    transform_train = RetinalPipelineTransform(is_training=True)
    transform_val = RetinalPipelineTransform(is_training=False)
    train_dataset = RetinalDataset(train_df, transform=transform_train, apply_bilateral=True)
    val_dataset = RetinalDataset(val_df, transform=transform_val, apply_bilateral=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Trigger Training (save to outputs/)
    output_dir = f'outputs/{experiment_id}'
    history = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion,
        epochs=epochs, lr=lr, weight_decay=1e-4, patience=5, mixed_precision=True,
        device_name=device_name, output_dir=output_dir, is_evidential=is_evidential
    )

    # 5. Optional evaluation on the test set
    if test_df is not None:
        print(f"🔄 Executing complete test evaluation for {experiment_id}...")
        import yaml
        import shutil
        import cv2
        from PIL import Image
        from src.evaluation.plots import plot_confusion_matrix, plot_reliability_diagram
        from src.explainability.layercam import LayerCAM

        # Load best saved weights
        weights_path = os.path.join(output_dir, "weights_best.pth")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device_name))
            print(f"Loaded best weights from {weights_path}")
        model.eval()

        # Copy metrics.csv to history.csv
        metrics_csv_path = os.path.join(output_dir, "metrics.csv")
        if os.path.exists(metrics_csv_path):
            shutil.copyfile(metrics_csv_path, os.path.join(output_dir, "history.csv"))

        # Save config.yaml
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # Setup test loader
        test_dataset = RetinalDataset(test_df, transform=transform_val, apply_bilateral=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Gather predictions
        all_paths = []
        all_pt_ids = []
        all_true = []
        all_pred = []
        all_probs = []
        all_uncertainties = []

        # Map classes back
        CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images = images.to(device_name)
                outputs = model(images)
                
                if is_evidential:
                    probs, var = get_evidence_metrics(outputs)
                    # EDL uncertainty: u = K / S where S = alpha.sum()
                    evidence = torch.relu(outputs)
                    alpha = evidence + 1.0
                    S = torch.sum(alpha, dim=1)
                    uncertainty = 4.0 / S
                else:
                    probs = torch.softmax(outputs, dim=1)
                    # Softmax uncertainty: normalized entropy
                    ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    uncertainty = ent / np.log(4.0)

                _, preds_idx = torch.max(probs, 1)

                all_probs.extend(probs.cpu().numpy())
                all_pred.extend(preds_idx.cpu().numpy())
                all_true.extend(targets.numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())

        # Match back index mappings to patient IDs and paths
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            all_paths.append(row.get('image_path', ''))
            all_pt_ids.append(row.get('patient_id', ''))

        # Build predictions.csv
        pred_records = []
        for i in range(len(all_true)):
            record = {
                'image_path': all_paths[i],
                'patient_id': all_pt_ids[i],
                'true_label': CLASSES[all_true[i]],
                'pred_label': CLASSES[all_pred[i]],
                'prob_0': all_probs[i][0],
                'prob_1': all_probs[i][1],
                'prob_2': all_probs[i][2],
                'prob_3': all_probs[i][3],
                'uncertainty': all_uncertainties[i]
            }
            pred_records.append(record)

        df_pred = pd.DataFrame(pred_records)
        df_pred.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
        print(f"Saved predictions to outputs/{experiment_id}/predictions.csv")

        # Save confusion matrix and reliability diagram
        plot_confusion_matrix(np.array(all_true), np.array(all_pred), CLASSES, os.path.join(output_dir, "confusion_matrix.png"))
        plot_reliability_diagram(np.array(all_probs), np.array(all_true), os.path.join(output_dir, "reliability_diagram.png"))
        print(f"Saved confusion matrix and reliability diagram inside outputs/{experiment_id}/")

        # Save LayerCAM attributions in layercam/ folder
        cam_dir = os.path.join(output_dir, "layercam")
        os.makedirs(cam_dir, exist_ok=True)
        
        # We select a target layer (e.g. layer4[-1] for resnet)
        target_layer = model.backbone.layer4[-1]
        cam_generator = LayerCAM(model, target_layer)

        for c_idx, c_name in enumerate(CLASSES):
            match_rows = test_df[test_df['label'] == c_idx]
            if len(match_rows) > 0:
                sample_row = match_rows.iloc[0]
                img_path = sample_row['image_path']
                if os.path.exists(img_path):
                    try:
                        img_pil = Image.open(img_path).convert('RGB')
                        tensor_inp = transform_val(img_pil).unsqueeze(0).to(device_name)
                        
                        # Generate CAM heatmap for the true class
                        cam_heatmap = cam_generator.generate(tensor_inp, class_idx=c_idx)
                        cam_np = cam_heatmap.detach().cpu().numpy()

                        # Resize and overlay
                        orig_cv = cv2.imread(img_path)
                        orig_cv = cv2.resize(orig_cv, (224, 224))
                        heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
                        overlay_img = cv2.addWeighted(orig_cv, 0.6, heatmap_color, 0.4, 0)

                        cv2.imwrite(os.path.join(cam_dir, f"{c_name.lower()}_cam.png"), overlay_img)
                    except Exception as e:
                        print(f"Failed to generate LayerCAM for {c_name}: {e}")
                        
        cam_generator.release()
        print(f"Saved LayerCAM heatmaps in outputs/{experiment_id}/layercam/")

    return history
