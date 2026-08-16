import os
import sys

# Add project root to sys.path

import logging
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tileselect.data.dataset import WsiTrainingDataset
from tileselect.models.clam import CLAM_SB
# Import RL Agent for filtered mode

# Import SmoothTop1SVM from local topk package (copied from CLAM)
from tileselect.vendored.topk.svm import SmoothTop1SVM


def setup_logger(log_file):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.handlers:
        root_logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)
    
    return root_logger

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- REPLICATED EarlyStopping from CLAM ---
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class ClamFeatureDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        h5_path, svs_path, label, num_patches = self.ds[idx]
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                features = f[self.ds.h5_key][:]
        except Exception as e:
            logging.error(f"Error reading {h5_path}: {e}")
            features = np.zeros((1, 768), dtype=np.float32)
        return torch.tensor(features, dtype=torch.float32), label

# --- NEW: Train Loop for CLAM ---
def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, loss_fn, device):
    model.train()
    
    train_loss = 0.
    train_inst_loss = 0.
    
    # Simple metrics
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}")
    
    for batch_idx, (data, label) in enumerate(pbar):
        data = data.squeeze(0).to(device)
        label_t = label.long().to(device)
        lbl = label_t.item()
        
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label_t, instance_eval=True)
        
        # Loss calculation matches CLAM
        loss = loss_fn(logits, label_t.view(1)) # Bag loss
        loss_value = loss.item()
        
        instance_loss = instance_dict['instance_loss']
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        train_loss += loss_value
        
        # Accuracy
        pred = Y_hat.item()
        if pred == lbl:
            correct += 1
        total += 1
        
        pbar.set_postfix(loss=f"{loss_value:.4f}", inst=f"{instance_loss_value:.4f}")

    train_loss /= len(loader)
    train_inst_loss /= len(loader)
    train_acc = correct / total if total > 0 else 0
    
    return train_loss, train_inst_loss, train_acc

# --- Validation Loop for CLAM ---
def validate_clam(epoch, model, loader, n_classes, loss_fn, device):
    model.eval()
    val_loss = 0.
    val_inst_loss = 0.
    
    correct = 0
    total = 0
    
    val_probs = []
    val_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.squeeze(0).to(device)
            label_t = label.long().to(device)
            lbl = label_t.item()
            try:
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label_t, instance_eval=True)
                
                loss = loss_fn(logits, label_t.view(1))
                val_loss += loss.item()
                
                instance_loss = instance_dict['instance_loss']
                val_inst_loss += instance_loss.item()
                
                val_probs.append(Y_prob.cpu().numpy())
                val_labels.append(lbl)
                
                if Y_hat.item() == lbl:
                    correct += 1
                total += 1
            except:
                continue

    val_loss /= len(loader)
    val_inst_loss /= len(loader)
    val_acc = correct / total if total > 0 else 0
    
    val_auc = 0.0
    if len(val_labels) > 0:
        val_labels = np.array(val_labels)
        val_probs = np.concatenate(val_probs, axis=0)
        if val_probs.shape[1] == 2:
            try:
                if len(np.unique(val_labels)) > 1:
                    val_auc = roc_auc_score(val_labels, val_probs[:, 1])
            except:
                pass

    return val_loss, val_inst_loss, val_acc, val_auc


def run_train_clam(cfg):
    output_dir = cfg['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(output_dir, 'train_clam.log'))

    logger.info("Script started with K-Fold support and CLAM configurations")

    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Param extraction
    k_folds = cfg['training'].get('k', 1)
    bag_weight = cfg['model'].get('bag_weight', 0.7)
    inst_loss_name = cfg['model'].get('inst_loss', 'ce')
    bag_loss_name = cfg['model'].get('bag_loss', 'ce')
    
    logger.info(f"Settings: K={k_folds}, BagWeight={bag_weight}, InstLoss={inst_loss_name}")
    
    # 1. Determine Splits Strategy
    # CLAM original loads 'splits_{i}.csv'. 
    # We will look for csvs in cfg['data']['split_dir'] if exists, compatible with k loop.
    split_dir = cfg['data'].get('split_dir')
    use_folds = cfg.get('experiment', {}).get('use_folds', False)
    if use_folds:
        # The per-fold splits sit next to the inherited CLAM checkpoints.
        split_dir = cfg.get('experiment', {}).get('checkpoint_dir') or split_dir
        if not split_dir:
            raise ValueError("use_folds is set but experiment.checkpoint_dir is missing")
        logger.info(f"use_folds is True, overriding split_dir to {split_dir}")

    for i in range(k_folds):
        logger.info(f"\n=== Training Fold {i} ===")
        
        # 1. Load Datasets for this Fold
        # If running K-folds, we expect specific split files
        train_csv_path = None
        val_csv_path = None
        
        if (k_folds > 1 or use_folds) and split_dir:
            # Try loading typical CLAM split files 'splits_{i}.csv' (contains train/val/test bools)
            # BUT WsiTrainingDataset expects a CSV with file paths/labels.
            # The CLAM 'splits_{i}.csv' usually has slide_ids.
            # This adaptation is tricky because WsiTrainingDataset is built around a single CSV + data_dir.
            # WORKAROUND: assume the pre-generated per-fold splits
            # (splits_{i}.csv) live under the directory the config points at.
            
            # If not, we fall back to the single train/val/test split defined in config (only valid for k=1? Or repeated?)
            # But the user asked for K-fold.
            
            # We will try to construct the path
            potential_split_file = os.path.join(split_dir, f'splits_{i}.csv')
            if os.path.exists(potential_split_file):
                 # This file usually needs parsing to extract train/val IDs, 
                 # then filter the master labels.csv.
                 # Using helper from CLAM/dataset_modules/dataset_generic.py would be ideal but hard to import.
                 # Let's implement a simple pandas filter.
                 import pandas as pd
                 split_df = pd.read_csv(potential_split_file)
                 # Expect columns: train, val, test (containing slide_ids)
                 # Wait, CLAM splits_{i}.csv format:
                 # index | train | val | test
                 # 0     | id1   | id2 | id3 ...
                 
                 # We need to filter the master labels_csv
                 master_df = pd.read_csv(cfg['data']['labels_csv'])
                 # Rename 'patient' to 'case_id' if needed or standardize
                 if 'patient' in master_df.columns:
                     master_df = master_df.rename(columns={'patient': 'case_id'})
                 if 'features_path' not in master_df.columns:
                      # Try to infer or assume standard structure? 
                      # For compatibility, assume features_path is present as user fixed it.
                      pass
                      
                 train_ids = split_df['train'].dropna().tolist()
                 val_ids = split_df['val'].dropna().tolist()
                 # test_ids = split_df['test'].dropna().tolist()
                 
                 train_df = master_df[master_df['slide_id'].isin(train_ids)]
                 val_df = master_df[master_df['slide_id'].isin(val_ids)]
                 
                 # We can instantiate WsiTrainingDataset from dataframe directly if supported?
                 # src/lib/dataset.py might need checking. 
                 # Assuming it takes a list of dicts/items or valid csv path.
                 # Let's verify WsiTrainingDataset.
                 # Just in case, we'll save temp CSVs for this fold.
                 fold_dir = os.path.join(output_dir, f'splits_temp_{i}')
                 os.makedirs(fold_dir, exist_ok=True)
                 train_csv_path = os.path.join(fold_dir, 'train.csv')
                 val_csv_path = os.path.join(fold_dir, 'val.csv')
                 train_df.to_csv(train_csv_path, index=False)
                 val_df.to_csv(val_csv_path, index=False)
            else:
                if k_folds > 1:
                     logger.warning(f"Split file {potential_split_file} not found. Configuring K-Fold but falling back to fixed splits (duplicate run).")
                train_csv_path = cfg['data']['train_split']
                val_csv_path = cfg['data']['val_split']
        else:
            train_csv_path = cfg['data']['train_split']
            val_csv_path = cfg['data']['val_split']

        train_ds = WsiTrainingDataset.from_directory(cfg['data']['dataset_dir'], train_csv_path, h5_key=cfg['data']['h5_key'])
        val_ds = WsiTrainingDataset.from_directory(cfg['data']['dataset_dir'], val_csv_path, h5_key=cfg['data']['h5_key'])

        # CLAM Wrapper Dataset to load features in worker processes
        train_ds_wrapped = ClamFeatureDataset(train_ds)
        val_ds_wrapped = ClamFeatureDataset(val_ds)

        # Use RandomSampler for CLAM parity
        sampler = RandomSampler(train_ds_wrapped)

        # Loaders - using the wrapped dataset
        train_loader = DataLoader(train_ds_wrapped, batch_size=1, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds_wrapped, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # 2. Model Init
        model = CLAM_SB(
            n_classes=cfg['model']['n_classes'], 
            embed_dim=cfg['model']['feature_dim'],
            dropout=cfg['model']['dropout'],
            instance_loss_fn = SmoothTop1SVM(n_classes=2).to(device) if inst_loss_name == 'svm' else nn.CrossEntropyLoss(),
            subtyping=cfg['model'].get('subtyping', False)
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['reg']))
        
        if bag_loss_name == 'svm':
             criterion = SmoothTop1SVM(n_classes=cfg['model']['n_classes']).to(device)
        else:
             criterion = nn.CrossEntropyLoss()
        
        # 3. Output for Fold
        fold_output_dir = os.path.join(output_dir, f'fold_{i}')
        os.makedirs(fold_output_dir, exist_ok=True)
        plots_dir = os.path.join(fold_output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 4. Early Stopping
        patience = cfg['training'].get('patience', 20)
        early_stopping = EarlyStopping(patience=patience, stop_epoch=50, verbose=True)
        
        # History
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
        best_epochs = []
        
        # 5. Training Loop
        for epoch in range(cfg['training']['epochs']):
            train_loss, train_inst_loss, train_acc = train_loop_clam(epoch, model, train_loader, optimizer, 
                                                                     cfg['model']['n_classes'], bag_weight, criterion, device)
            
            logger.info(f"Ep {epoch}: Loss={train_loss:.4f}, InstLoss={train_inst_loss:.4f}, Acc={train_acc:.4f}")
            
            val_loss, val_inst_loss, val_acc, val_auc = validate_clam(epoch, model, val_loader, 
                                                                      cfg['model']['n_classes'], criterion, device)
            
            logger.info(f"  Val: Loss={val_loss:.4f}, InstLoss={val_inst_loss:.4f}, Acc={val_acc:.4f}, AUROC={val_auc:.4f}")
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # Checkpoint
            if val_loss < early_stopping.val_loss_min:
                best_epochs.append(epoch)
            
            checkpoint_path = os.path.join(fold_output_dir, 'checkpoint.pt')
            early_stopping(epoch, val_loss, model, ckpt_name=checkpoint_path)
            
            # Plotting per fold (UPDATED: Saves every epoch)
            try:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(history['train_loss'], label='Train')
                plt.plot(history['val_loss'], label='Val')
                
                if len(best_epochs) > 0:
                    best_loss_values = [history['val_loss'][e] for e in best_epochs]
                    plt.scatter(best_epochs, best_loss_values, c='red', s=30, zorder=5, label='Best')

                plt.title('Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(history['train_acc'], label='Train Acc')
                plt.plot(history['val_acc'], label='Val Acc')
                plt.plot(history['val_auc'], label='Val AUC')
                plt.title('Metrics')
                plt.legend()
                
                plt.savefig(os.path.join(plots_dir, 'curves.png'))
                plt.close()
            except:
                pass

            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
            
    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_train_clam(cfg)

if __name__ == "__main__":
    main()

