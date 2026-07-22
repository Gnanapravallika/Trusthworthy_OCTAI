import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.preprocessing.standardizer import load_and_preprocess_scan

CLASSES = ["cnv", "dme", "drusen", "normal"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

class RetinalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading retinal OCT scans from metadata dataframe.
    """
    def __init__(self, df: pd.DataFrame, transform=None, apply_bilateral: bool = True, apply_clahe: bool = False, apply_min_max: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.apply_bilateral = apply_bilateral
        self.apply_clahe = apply_clahe
        self.apply_min_max = apply_min_max

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['image_path']
        label = int(row['label'])
        
        try:
            img = load_and_preprocess_scan(
                file_path, 
                apply_bilateral=self.apply_bilateral,
                apply_clahe=self.apply_clahe,
                apply_min_max=self.apply_min_max
            )
        except Exception as e:
            # Fallback to grey placeholder image on failure
            img = Image.new('RGB', (224, 224), color=128)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def patient_level_split(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15) -> tuple:
    """
    Divides patients strictly (70:15:15) to prevent data leakage and ensure stratified splits.
    """
    np.random.seed(42)
    
    train_patients = set()
    val_patients = set()
    test_patients = set()
    
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        unique_pts = label_df['patient_id'].unique()
        np.random.shuffle(unique_pts)
        
        n_pts = len(unique_pts)
        if n_pts >= 3:
            train_patients.add(unique_pts[0])
            val_patients.add(unique_pts[1])
            test_patients.add(unique_pts[2])
            
            remaining = unique_pts[3:]
            n_rem = len(remaining)
            n_tr = int(n_rem * train_ratio)
            n_vl = int(n_rem * val_ratio)
            
            train_patients.update(remaining[:n_tr])
            val_patients.update(remaining[n_tr:n_tr+n_vl])
            test_patients.update(remaining[n_tr+n_vl:])
        else:
            for i, pt in enumerate(unique_pts):
                if i % 3 == 0:
                    train_patients.add(pt)
                elif i % 3 == 1:
                    val_patients.add(pt)
                else:
                    test_patients.add(pt)
                    
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]
    
    return train_df, val_df, test_df

def auto_detect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically maps dataset CSV columns to standardized keys: image_path, label, patient_id.
    """
    cols = df.columns
    rename_dict = {}
    
    path_col = next((c for c in cols if any(k in c.lower() for k in ["path", "file", "image", "filename"])), None)
    if path_col:
        rename_dict[path_col] = 'image_path'
        
    label_col = next((c for c in cols if any(k in c.lower() for k in ["label", "class", "disease", "category", "target"])), None)
    if label_col:
        rename_dict[label_col] = 'label'
        
    patient_col = next((c for c in cols if any(k in c.lower() for k in ["patient", "subject", "id", "user"])), None)
    if patient_col:
        rename_dict[patient_col] = 'patient_id'
        
    df = df.rename(columns=rename_dict)
    
    for req in ['image_path', 'label', 'patient_id']:
        if req not in df.columns:
            if req == 'patient_id':
                df['patient_id'] = df.index.astype(str)
            else:
                raise ValueError(f"Required column '{req}' could not be detected in DataFrame.")
                
    if df['label'].dtype == object or df['label'].dtype == str:
        df['label'] = df['label'].apply(lambda x: CLASS_TO_IDX.get(str(x).strip().lower(), -1))
        df = df[df['label'] != -1]
        
    return df
