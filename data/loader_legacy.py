# loader_torch.py

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd

# -------------------------------
# 1. Directory to Label Mapping
# -------------------------------

# alpha_map = {"0.2": 0, "0.3": 1, "0.4": 2}
# q0_map = {"1": 0, "1.5": 1, "2.0": 2, "2.5": 3}

alpha_map = {"0.2": 0, "0.25": 1, "0.3": 2, "0.35": 3, "0.4": 4, "0.45": 5, "0.5": 6}
q0_map = {"1": 0, "1.5": 1, "2.0": 2, "2.5": 3, "3.0": 4}


def parse_labels_from_dir(dir_name):
    """Parse directory name into label tuple (energy_loss, alpha_s, q0)."""
    energy_loss_str, alpha_str, q0_str = dir_name.split('_')
    energy_loss = 0 if energy_loss_str == "MMAT" else 1
    alpha = alpha_map[alpha_str]
    q0 = q0_map[q0_str]
    return (energy_loss, alpha, q0)


# -----------------------------------------
# 2. File and Label Generator (Python level)
# -----------------------------------------

def file_label_generator(root_dir):
    """Yield (file_path, label_tuple) for all files in dataset."""
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            label_tuple = parse_labels_from_dir(dir_name)
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(dir_path, file_name)
                    yield file_path, label_tuple

def save_file_label_list(file_label_list, filename, root_dir):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'energy_loss', 'alpha', 'q0'])
        for file_path, (energy_loss, alpha, q0) in file_label_list:
            relative_path = os.path.relpath(file_path, root_dir)
            writer.writerow([relative_path, energy_loss, alpha, q0])

def load_file_label_list(filename, root_dir):
    result = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            relative_path = row['file_path']
            absolute_path = os.path.join(root_dir, relative_path)
            label = (int(row['energy_loss']), int(row['alpha']), int(row['q0']))
            result.append((absolute_path, label))
    return result


# ----------------------------------------------------------
# 3. Stratified Split for Train/Val/Test (Balanced and Stable)
# ----------------------------------------------------------

def split_file_list(file_label_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """Stratified split on (file_path, label_tuple)."""
    np.random.seed(random_seed)
    label_to_files = defaultdict(list)
    for file_path, label in file_label_list:
        label_to_files[label].append(file_path)

    train_list, val_list, test_list = [], [], []

    for label, files in label_to_files.items():
        files = np.array(files)
        np.random.shuffle(files)
        total = len(files)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        if total < 3:
            train_split = files[:1]
            val_split = files[1:2] if total > 1 else []
            test_split = files[2:] if total > 2 else []
        else:
            train_split = files[:train_end]
            val_split = files[train_end:val_end]
            test_split = files[val_end:]

        train_list.extend([(fp, label) for fp in train_split])
        val_list.extend([(fp, label) for fp in val_split])
        test_list.extend([(fp, label) for fp in test_split])

    np.random.shuffle(train_list)
    np.random.shuffle(val_list)
    np.random.shuffle(test_list)

    return train_list, val_list, test_list


# --------------------------------------------------------
# 4. PyTorch Dataset Generator (Lazy loading .npy)
# --------------------------------------------------------

class JetDataset(Dataset):
    """PyTorch Dataset for loading event images and multi-output labels."""
    
    def __init__(self, file_label_list, global_max, device='cuda',transform=None):
        self.file_label_list = file_label_list
        self.global_max = global_max
        self.transform = transform
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.file_label_list)

    def __getitem__(self, idx):
        file_path, label = self.file_label_list[idx]
        
        # Load the event image from .npy file and normalize
        if np.isfinite(self.global_max) and self.global_max>0 :
            event = np.load(file_path).astype(np.float32) / self.global_max
        else:
            event = np.load(file_path).astype(np.float32)
            # print(f"[WARNING] global_max <=0 or NaN, no normalization applied")
        event = np.expand_dims(event, axis=0)  # Shape: (1, 32, 32) for PyTorch
        
        # Convert labels to tensors
        energy_loss_label = torch.tensor([label[0]], dtype=torch.long, device=self.device)  # (1,)
        alpha_label = torch.tensor([label[1]], dtype=torch.long, device=self.device)             # (3-class)
        q0_label = torch.tensor([label[2]], dtype=torch.long, device=self.device)                # (4-class)

        labels = {
            'energy_loss_output': energy_loss_label,
            'alpha_output': alpha_label,
            'q0_output': q0_label
        }

        # Apply any transformations (e.g., augmentation)
        if self.transform:
            event = self.transform(event)

        return torch.tensor(event, device=self.device), labels

class AggregatedJetDataset(Dataset):
    def __init__(self, agg_csv, root_dir, global_max, device='cuda'):
        self.df = pd.read_csv(agg_csv)
        self.root_dir = root_dir
        self.global_max = global_max
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_list = row['file_paths'].split('|')
        imgs = []
        for rel_path in file_list:
            abs_path = os.path.join(self.root_dir, rel_path)
            arr = np.load(abs_path).astype(np.float32) / self.global_max
            imgs.append(torch.tensor(arr, device=self.device).unsqueeze(0))
        img_avg = torch.stack(imgs).mean(dim=0)
        labels = {
            'energy_loss_output': torch.tensor([row['energy_loss']], dtype=torch.long, device=self.device),
            'alpha_output':       torch.tensor([row['alpha']],       dtype=torch.long, device=self.device),
            'q0_output':          torch.tensor([row['q0']],          dtype=torch.long, device=self.device)
        }
        return img_avg, labels
# -------------------------------
# Split Saving/Loading Utilities
# -------------------------------

def save_split_to_csv(file_label_list, filename, root_dir):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'energy_loss', 'alpha', 'q0'])
        for file_path, (energy_loss, alpha, q0) in file_label_list:
            relative_path = os.path.relpath(file_path, root_dir)  # Make path relative
            writer.writerow([relative_path, energy_loss, alpha, q0])


def load_split_from_csv(filename, root_dir):
    result = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Support both non-aggregated ('file_path') and aggregated ('file_paths')
            rel = row.get('file_path') or row.get('file_paths') or row.get('agg_file_path')
            if rel is None:
                raise KeyError(f"CSV {filename} missing 'file_path' or 'file_paths' column.")
            absolute_path = os.path.join(root_dir, rel)
            # relative_path = row['file_paths']
            # absolute_path = os.path.join(root_dir, relative_path)  # Rebuild full path
            label = (int(row['eloss']), int(row['alpha']), int(row['q0']))
            result.append((absolute_path, label))
    return result

def get_dataloaders(cfg, device='cuda'):
    """
    Create DataLoader objects for training, validation, and testing datasets.
    If cfg.group_size > 1, uses AggregatedJetDataset; otherwise uses JetDataset.
    """
    # Choose dataset class based on whether we're aggregating
    if cfg.group_size> 1:
        # Aggregated splits are CSVs with columns: agg_id, file_paths, energy_loss, alpha, q0
        train_ds = AggregatedJetDataset(cfg.train_csv, cfg.dataset_root_dir,cfg.global_max, device=device)
        val_ds   = AggregatedJetDataset(cfg.val_csv, cfg.dataset_root_dir,cfg.global_max, device=device)
        test_ds  = AggregatedJetDataset(cfg.test_csv, cfg.dataset_root_dir,cfg.global_max, device=device)

    else:
        '''Create DataLoader objects for training, validation, and testing datasets.'''
        train_list = load_split_from_csv(cfg.train_csv, cfg.dataset_root_dir)
        val_list = load_split_from_csv(cfg.val_csv, cfg.dataset_root_dir)
        test_list = load_split_from_csv(cfg.test_csv, cfg.dataset_root_dir)

        train_ds = JetDataset(train_list, global_max=cfg.global_max, device=device)
        val_ds = JetDataset(val_list, global_max=cfg.global_max, device=device)
        test_ds = JetDataset(test_list, global_max=cfg.global_max, device=device)
    print(f"[INFO] Training samples: {len(train_ds)}")
    print(f"[INFO] Validation samples: {len(val_ds)}")
    print(f"[INFO] Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    # Show the length of the dataloaders
    print(f"[INFO] Length of training dataloader: {len(train_loader)}")
    print(f"[INFO] Length of validation dataloader: {len(val_loader)}")
    print(f"[INFO] Length of test dataloader: {len(test_loader)}")

    return train_loader, val_loader, test_loader

# -------------------------------
# 5. Main Function for Testing
# -------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch DataLoader for ML-JET dataset with smart caching and splits")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--global_max', type=float, required=True, help='Global max for normalization')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for DataLoader')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Shuffle buffer size')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # File names inside dataset root
    train_file = os.path.join(args.root_dir, "train_files.csv")
    val_file = os.path.join(args.root_dir, "val_files.csv")
    test_file = os.path.join(args.root_dir, "test_files.csv")
    file_label_cache = os.path.join(args.root_dir, "file_labels.csv")

    # -------------------------------
    # Priority Check 1: Splits exist?
    # -------------------------------
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print(f"[INFO] Found existing splits in \n'{args.root_dir}'.\n Loading splits directly...")
        train_list = load_split_from_csv(train_file, args.root_dir)
        val_list = load_split_from_csv(val_file, args.root_dir)
        test_list = load_split_from_csv(test_file, args.root_dir)
    else:
        print(f"[INFO] Splits not found. Checking for cached file-label list...")

        # -------------------------------
        # Priority Check 2: File label list exists?
        # -------------------------------
        if os.path.exists(file_label_cache):
            print(f"[INFO] Found cached file-label list '{file_label_cache}'.")
            file_label_list = load_file_label_list(file_label_cache, args.root_dir)
        else:
            print(f"[INFO] Cached file-label list not found. Scanning dataset directory to generate...")
            file_label_list = list(file_label_generator(args.root_dir))
            print(f"[INFO] Total files found: {len(file_label_list)}")
            save_file_label_list(file_label_list, file_label_cache, args.root_dir)
            print(f"[INFO] File-label list cached to '{file_label_cache}'.")

        # Now split the loaded/generated file-label list
        print("[INFO] Performing stratified split...")
        train_list, val_list, test_list = split_file_list(file_label_list, random_seed=args.random_seed)

        print(f"Training set size: {len(train_list)}")
        print(f"Validation set size: {len(val_list)}")
        print(f"Test set size: {len(test_list)}")

        # Save splits for future use
        save_split_to_csv(train_list, train_file, args.root_dir)
        save_split_to_csv(val_list, val_file, args.root_dir)
        save_split_to_csv(test_list, test_file, args.root_dir)
        print(f"[INFO] Splits saved inside dataset root '{args.root_dir}'.")

    # -------------------------------
    # PyTorch Dataset Pipeline
    # -------------------------------
    print("[INFO] Building PyTorch datasets for training/validation/testing...")
    train_dataset = JetDataset(train_list, global_max=args.global_max)
    val_dataset = JetDataset(val_list, global_max=args.global_max)
    test_dataset = JetDataset(test_list, global_max=args.global_max)

    print(f"[INFO] Dataset pipeline built successfully. Example batch:")

    # Testing Dataset
    x, labels = train_dataset[0]
    print(f"Input batch shape: {x.shape}")
    for key, value in labels.items():
        print(f"Label - {key}: {value.size()}")

    print("✅ DataLoader pipeline ready with smart caching and split management.")


if __name__ == "__main__":
    main()

#python data/loader_torch.py --root_dir ~/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled --global_max 121.79151153564453 --batch_size 512
