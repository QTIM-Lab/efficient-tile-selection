import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root
from tileselect.models.unet import SimpleUNet
from tileselect import paths


# H&E stain-jitter (HED-light, Ruifrok-Johnston stain matrix) for stain-colour robustness.
# skimage is unavailable, so the RGB<->HED transform is done directly in optical-density space.
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]], dtype=np.float32)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def hed_stain_jitter(thumb_uint8, sigma_alpha=0.06, sigma_beta=0.015):
    """Randomly perturb the H/E/D stain channels (Tellez HED-light): OD = -log(I), project
    to stain space, apply per-channel gain alpha~U(1-s,1+s) and bias beta~U(-s,s), reconstruct.
    White background stays ~unchanged (OD~0). Photometric — thumbnail only, never the heatmap."""
    rgb = np.clip(thumb_uint8.astype(np.float32) / 255.0, 1e-6, 1.0)
    od = -np.log(rgb)                                   # optical density [H,W,3]
    hed = od.reshape(-1, 3) @ _HED_FROM_RGB             # to stain space
    # Jitter only the Haematoxylin and Eosin channels; leave the 3rd (DAB/residual) alone,
    # so H&E slides stay in a realistic pink/purple gamut (no green casts).
    alpha = 1.0 + np.random.uniform(-sigma_alpha, sigma_alpha, 3).astype(np.float32)
    beta = np.random.uniform(-sigma_beta, sigma_beta, 3).astype(np.float32)
    alpha[2] = 1.0
    beta[2] = 0.0
    hed = hed * alpha + beta
    od2 = (hed @ _RGB_FROM_HED).reshape(od.shape)
    rgb2 = np.exp(-od2)
    return np.clip(rgb2 * 255.0, 0, 255).astype(np.uint8)

class UNetDataset(Dataset):
    def __init__(self, data_dir, augment=False, strong=False, max_norm=False):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.augment = augment
        # DiceBCE compares per-cell values, so it needs the peak at 1.0. ListNet consumes
        # the target as a distribution that already sums to 1 and must not rescale it.
        self.max_norm = max_norm
        # 'strong' adds 90-degree rotations (geometric) + H&E stain jitter (photometric),
        # on top of the basic flips + colour jitter.
        self.strong = strong
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ) if augment else None
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        # Trust local data
        data = torch.load(path, weights_only=False)
        
        # Input: Thumbnail (H, W, 3) -> (3, H, W)
        thumb = data['thumbnail']
        if self.augment and self.strong:
            thumb = hed_stain_jitter(thumb)   # H&E stain-colour jitter (thumbnail only)
        thumb_pil = Image.fromarray(thumb)
        
        # Target: Heatmap (H, W) -> (1, H, W)
        heatmap = data['heatmap']
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        # Only DiceBCE needs the peak at 1.0 (see __init__). Under ListNet the target is
        # left exactly as generated — softmax over tiles, scattered and blurred, summing
        # to 1 — because the loss reads it as a distribution.
        if self.max_norm:
            h_max = heatmap.max()
            if h_max > 0:
                heatmap = heatmap / h_max
        
        # Data augmentation: apply same spatial transforms to both
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                thumb_pil = thumb_pil.transpose(Image.FLIP_LEFT_RIGHT)
                heatmap = heatmap.flip(-1)
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                thumb_pil = thumb_pil.transpose(Image.FLIP_TOP_BOTTOM)
                heatmap = heatmap.flip(-2)
            # Random 90-degree rotation — WSI has no canonical orientation (strong only).
            # Same counterclockwise rotation on both thumbnail and heatmap.
            if self.strong:
                k = int(torch.randint(0, 4, (1,)).item())
                for _ in range(k):
                    thumb_pil = thumb_pil.transpose(Image.ROTATE_90)
                if k:
                    heatmap = torch.rot90(heatmap, k, dims=(-2, -1))
            # Color jitter (only on thumbnail, not heatmap)
            thumb_pil = self.color_jitter(thumb_pil)
        
        thumb = transforms.ToTensor()(thumb_pil) # [3, 224, 224], 0-1
        heatmap = heatmap.unsqueeze(0)
        
        return thumb, heatmap

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for sparse heatmap regression.
    
    Dice loss handles class imbalance by measuring overlap as a proportion,
    while BCE provides pixel-level gradient signal. The combination trains
    both the spatial structure (Dice) and the exact values (BCE).
    """
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice
        
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class ListwiseRankingLoss(nn.Module):
    """Listwise ranking loss (ListNet-style): cross-entropy between the target attention
    distribution and the U-Net's predicted distribution over the grid cells. It matches
    the *shape* of attention rather than per-cell values, so it optimizes the ORDER of
    cells — which is what top-k tile selection needs, unlike MSE/Dice.

    The target is consumed as-is: softmax over the slide's tiles already sums to 1, the
    scatter keeps one value per cell and the Gaussian blur (mode='reflect') conserves
    mass, so it arrives here as a distribution and nothing is renormalized.
    """
    def forward(self, logits, targets):
        b = logits.shape[0]
        log_p = F.log_softmax(logits.view(b, -1), dim=1)           # predicted log-distribution
        return -(targets.view(b, -1) * log_p).sum(dim=1).mean()    # cross-entropy


class HeatmapMSELoss(nn.Module):
    """The textbook heatmap-regression baseline: plain MSE against a Gaussian-blurred
    target scaled to a peak of 1.0 (max_norm applies here, see UNetDataset).

    This is what pose estimation and landmark detection optimise, so it is the
    conventional reference the distribution losses (ListNet, L1) should be measured
    against. The model emits raw logits, so they are squashed with a sigmoid first —
    the other losses apply their own transform internally for the same reason.

    Note this compares per-cell VALUES, not shape: unlike ListNet it is not invariant to
    rescaling the map, which is why the target has to be max-normalised for it to mean
    anything.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        return self.mse(torch.sigmoid(logits), targets)


class L1DistributionLoss(nn.Module):
    """L1 between the target attention distribution and the predicted one — twice the
    total variation distance. SUMMED over cells, not averaged: both sides sum to 1, so the
    value lands in [0, 2] whatever the grid size, with no dependence on the bag.

    Why try it over ListNet: cross-entropy's gradient fades as the error shrinks, so
    background cells settle at small non-zero values and the predicted map stays flat
    (normalised entropy 0.92 after pretraining, against CLAM's own 0.72). L1's gradient
    does not fade with the error, so the background keeps being pushed towards zero — the
    same reason LASSO gives sparse solutions and ridge does not. On a realistic target the
    background-to-signal gradient ratio is 0.049 here against 0.017 for ListNet.

    Caveat: the softmax Jacobian scales every cell by its own probability, so the absolute
    gradients run ~50x smaller than ListNet's. Adam normalises by the gradient RMS, so a
    uniform rescaling like this is largely absorbed and training.lr can stay as it is.
    """
    def forward(self, logits, targets):
        b = logits.shape[0]
        p = F.softmax(logits.view(b, -1), dim=1)                   # predicted distribution
        return (p - targets.view(b, -1)).abs().sum(dim=1).mean()


def top_k_overlap(preds, targets, k_percent=0.10):
    B = preds.shape[0]
    N = preds.shape[2] * preds.shape[3]
    k = int(N * k_percent)
    if k == 0: return 0.0
    
    overlap_sum = 0.0
    for i in range(B):
        p_flat = preds[i].view(-1)
        t_flat = targets[i].view(-1)
        
        _, p_top = torch.topk(p_flat, k)
        _, t_top = torch.topk(t_flat, k)
        
        # Count intersection
        p_top_mask = torch.zeros(N, device=preds.device, dtype=torch.bool)
        t_top_mask = torch.zeros(N, device=preds.device, dtype=torch.bool)
        
        p_top_mask[p_top] = True
        t_top_mask[t_top] = True
        
        intersection = (p_top_mask & t_top_mask).sum().item()
        overlap_sum += intersection / k
    return overlap_sum / B

def run_train_unet(cfg, fold=None):
    """
    fold=None : train on the original unet_data_dir, save to unet_path.
    fold=0..9 : train on unet_data_dir_fold_{fold}, save to unet_path with _fold_{fold} suffix.
    """
    base_data_dir = cfg['data'].get('unet_data_dir', os.path.join(cfg['data']['dataset_dir'], 'unet_data'))
    base_save_path = cfg['model']['unet_path']

    if fold is not None:
        data_dir  = base_data_dir + f'_fold_{fold}'
        save_path = base_save_path.replace('.pt', f'_fold_{fold}.pt')
        print(f"[Fold {fold}] data_dir={data_dir}  save_path={save_path}")
    else:
        data_dir  = base_data_dir
        save_path = base_save_path

    epochs = int(cfg['training'].get('epochs', 20))
    batch_size = int(cfg['training'].get('batch_size', 16))
    lr = float(cfg['training'].get('lr', 1e-4))
    # U-Net output grid (attention resolution); may differ from the thumbnail input size.
    grid_size = int(cfg['data'].get('unet_grid_size', cfg['data'].get('thumb_size', 224)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data — use augmentation for train, none for val
    full_dataset = UNetDataset(data_dir, augment=False)
    print(f"Found {len(full_dataset)} samples in {data_dir}.")
    
    if len(full_dataset) == 0:
        print("No data found. Please run generate_unet_data first.")
        return

    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    
    # Get indices for train/val split
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )
    
    # Create separate datasets with/without augmentation.
    # training.augment: 'basic' (default: flips + colour jitter) or 'strong' (+ 90-deg
    # rotations + H&E stain jitter).
    strong_aug = str(cfg['training'].get('augment', 'basic')).lower() == 'strong'
    if strong_aug:
        print("Augmentation: STRONG (flips + colour jitter + 90-deg rotations + HED stain jitter)")
    loss_name = cfg['training'].get('loss', 'dicebce')
    # DiceBCE needs the peak at 1.0; the distribution losses need the target left summing to 1.
    max_norm = loss_name not in ('ranking', 'l1')
    train_ds = UNetDataset(data_dir, augment=True, strong=strong_aug, max_norm=max_norm)
    val_ds = UNetDataset(data_dir, augment=False, max_norm=max_norm)
    
    train_subset = torch.utils.data.Subset(train_ds, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_ds, val_indices.indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # --- FIXED BATCH FOR MONITORING ---
    monitor_batch = next(iter(val_loader))
    mon_thumbs, mon_maps = monitor_batch[0][:5].to(device), monitor_batch[1][:5].to(device)

    # Model — input = thumbnail (thumb_size), output = attention grid (grid_size)
    model = SimpleUNet(n_channels=3, n_classes=1, out_size=grid_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    if loss_name == 'ranking':
        criterion = ListwiseRankingLoss()
        print("Loss: ListwiseRankingLoss (target consumed as a distribution, no rescaling)")
    elif loss_name == 'l1':
        criterion = L1DistributionLoss()
        print("Loss: L1DistributionLoss (total variation between the two distributions)")
    elif loss_name == 'mse':
        criterion = HeatmapMSELoss()
        print("Loss: HeatmapMSELoss (standard heatmap regression on the peak-1 map)")
    else:
        criterion = DiceBCELoss(bce_weight=0.5, smooth=1.0)
        print("Loss: DiceBCELoss")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_loss = float('inf')
    patience = int(cfg['training'].get('patience', 20))
    no_improve = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_overlap': [], 'val_overlap': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_overlap_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for thumbs, maps in pbar:
            thumbs, maps = thumbs.to(device), maps.to(device)
            
            optimizer.zero_grad()
            outputs = model(thumbs)
            loss = criterion(outputs, maps)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * thumbs.size(0)
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                ov = top_k_overlap(preds, maps, k_percent=0.10)
                train_overlap_sum += ov * thumbs.size(0)
                
            pbar.set_postfix(loss=loss.item(), ov=ov)
            
        train_loss /= len(train_subset)
        train_ov = train_overlap_sum / len(train_subset)
        
        # Val
        model.eval()
        val_loss = 0.0
        val_overlap_sum = 0.0
        with torch.no_grad():
            for thumbs, maps in val_loader:
                thumbs, maps = thumbs.to(device), maps.to(device)
                outputs = model(thumbs)
                loss = criterion(outputs, maps)
                val_loss += loss.item() * thumbs.size(0)
                
                preds = torch.sigmoid(outputs)
                ov = top_k_overlap(preds, maps, k_percent=0.10)
                val_overlap_sum += ov * thumbs.size(0)
                
        val_loss /= len(val_subset)
        val_ov = val_overlap_sum / len(val_subset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_overlap'].append(train_ov)
        history['val_overlap'].append(val_ov)
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} (Ov: {train_ov:.4f}), Val Loss {val_loss:.4f} (Ov: {val_ov:.4f})")
        
        # LR scheduler
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
            
        # --- UPDATE PLOTLY MONITORING EVERY EPOCH ---
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plots_dir = paths.plots_dir(cfg)
        os.makedirs(plots_dir, exist_ok=True)
        out_curves_html = os.path.join(plots_dir, 'unet_training_curves.html')
        
        # Accumulate prediction for 1 example at each epoch
        with torch.no_grad():
            mon_pred_now = torch.sigmoid(model(mon_thumbs[:1]))[0, 0].cpu().numpy()
        if not hasattr(run_train_unet, '_epoch_preds'):
            run_train_unet._epoch_preds = []
        run_train_unet._epoch_preds.append((epoch + 1, mon_pred_now))
        
        # Show every 10 epochs + always the latest
        display_epochs = []
        for ep_num, pred in run_train_unet._epoch_preds:
            if ep_num == 1 or ep_num % 10 == 0 or ep_num == epoch + 1:
                if ep_num not in [e for e, _ in display_epochs]:
                    display_epochs.append((ep_num, pred))
        
        n_rows = 1 + len(display_epochs)  # Row 1 for curves, then 1 row per displayed epoch
        
        specs = [[{"colspan": 1}, {"colspan": 1}, None]]  # Row 1: curves
        for _ in display_epochs:
            specs.append([{}, {}, {}])
        
        titles = ['Dice+BCE Loss', 'Top-10% Tile Overlap']
        for ep_num, _ in display_epochs:
            titles += [f'Ep {ep_num}: Thumb', f'Ep {ep_num}: GT', f'Ep {ep_num}: Pred']

        fig = make_subplots(
            rows=n_rows, cols=3,
            subplot_titles=titles,
            specs=specs,
            vertical_spacing=max(0.01, 0.3 / n_rows)
        )
        
        cur_epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Row 1: Curves
        fig.add_trace(go.Scatter(x=cur_epochs, y=history['train_loss'], mode='lines', name='Train Loss', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=cur_epochs, y=history['val_loss'], mode='lines', name='Val Loss', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=cur_epochs, y=history['train_overlap'], mode='lines', name='Train Ov', line=dict(color='green')), row=1, col=2)
        fig.add_trace(go.Scatter(x=cur_epochs, y=history['val_overlap'], mode='lines', name='Val Ov', line=dict(color='red')), row=1, col=2)
        
        # One row per displayed epoch: Thumb, GT, Pred at that epoch
        thumb_img = (mon_thumbs[0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        gt_map = mon_maps[0, 0].cpu().numpy()
        
        import matplotlib.cm as cm
        
        for idx, (ep_num, pred) in enumerate(display_epochs):
            r = idx + 2
            fig.add_trace(go.Image(z=thumb_img), row=r, col=1)
            
            # Map GT to RGB using matplotlib Jet
            gt_rgb = cm.jet(gt_map) # Returns (H, W, 4) in [0, 1]
            gt_rgb = (gt_rgb[..., :3] * 255).astype('uint8')
            fig.add_trace(go.Image(z=gt_rgb), row=r, col=2)
            
            # Map Pred to RGB using matplotlib Jet
            pr_rgb = cm.jet(np.clip(pred, 0, 1)) # Returns (H, W, 4) in [0, 1]
            pr_rgb = (pr_rgb[..., :3] * 255).astype('uint8')
            fig.add_trace(go.Image(z=pr_rgb), row=r, col=3)
            
            for c in [1, 2, 3]:
                fig.update_xaxes(showticklabels=False, row=r, col=c)
                fig.update_yaxes(showticklabels=False, autorange='reversed', row=r, col=c)

        fig.update_layout(height=400 + len(display_epochs)*280, title_text=f"U-Net Training Monitor (Epoch {epoch+1})", template="plotly_white", showlegend=True)
        fig.write_html(out_curves_html)

    print("Training Complete.")
    
    # --- VISUALIZATION ---
    print("Generating sample visualization...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    # Get a batch
    thumbs, maps = next(iter(val_loader))
    thumbs, maps = thumbs.to(device), maps.to(device)
    with torch.no_grad():
        logits = model(thumbs)
        preds = torch.sigmoid(logits)
    
    # Plot top 4
    N = min(4, thumbs.size(0))
    fig, axes = plt.subplots(N, 4, figsize=(16, 4*N), squeeze=False)
    
    for i in range(N):
        # 1. Thumbnail
        t = thumbs[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(t, interpolation='bilinear')
        axes[i, 0].set_title("Thumbnail")
        axes[i, 0].axis('off')
        
        # 2. Ground Truth
        m = maps[i, 0].cpu().numpy()
        axes[i, 1].imshow(m, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
        axes[i, 1].set_title("CLAM Attention (GT)")
        axes[i, 1].axis('off')
        
        # 3. U-Net Prediction (High Res)
        p = preds[i, 0].cpu().numpy()
        axes[i, 2].imshow(p, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
        axes[i, 2].set_title("U-Net Pred (224x224)")
        axes[i, 2].axis('off')
        
        # 4. Agent Input (Low Res 28x28)
        # Compute downsample exactly as in Env
        p_tensor = preds[i:i+1] # [1, 1, 224, 224]
        p_small = F.adaptive_avg_pool2d(p_tensor, (28, 28)) # [1, 1, 28, 28]
        p_small_np = p_small[0, 0].cpu().numpy()
        
        axes[i, 3].imshow(p_small_np, cmap='jet', vmin=0, vmax=1, interpolation='nearest')
        axes[i, 3].set_title("Agent Input (28x28)")
        axes[i, 3].axis('off')
        
    plots_dir = paths.plots_dir(cfg)
    out_png = os.path.join(plots_dir, 'unet_comparison.png')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved visualization to {out_png}")

def load_config(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index (0-9). Omit for original single-model behaviour.')
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_train_unet(cfg, fold=args.fold)

if __name__ == "__main__":
    main()
