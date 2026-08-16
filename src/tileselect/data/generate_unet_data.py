import os
import sys
import ctypes
import contextlib
import argparse
import numpy as np
import torch
import h5py
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from tileselect.utils.suppress_tiff import openslide  # noqa: F401


from tileselect.data.dataset import WsiTrainingDataset
from tileselect.models.clam import CLAM_SB

def get_args():
    parser = argparse.ArgumentParser(description="Generate U-Net Training Data (Thumbnail -> Attention Map)")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--output_dir', type=str, default='data/unet_data')
    parser.add_argument('--target_size', type=int, default=224, help="Size of thumbnail and heatmap (square)")
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_heatmap(coords, attention_scores, slide_dim, target_size, sigma=3.0):
    """
    Map attention scores to a 2D grid based on coordinates,
    then apply Gaussian smoothing to create a continuous, learnable target.
    
    Args:
        coords: (N, 2) array of (x, y) coordinates (level 0).
        attention_scores: (N,) array of attention scores [0, 1].
        slide_dim: (w, h) tuple of slide dimensions at level 0.
        target_size: Int, output grid size (target_size x target_size).
        sigma: Float, Gaussian blur sigma. Controls how far each tile's
               attention spreads spatially. Larger = smoother.
        
    Returns:
        heatmap: (target_size, target_size) numpy array, smoothed.
    """
    w, h = slide_dim
    heatmap = np.zeros((target_size, target_size), dtype=np.float32)
    
    # Scale factor
    scale_x = target_size / w
    scale_y = target_size / h
    
    for coord, score in zip(coords, attention_scores):
        x, y = coord
        gx = int(x * scale_x)
        gy = int(y * scale_y)
        gx = min(max(0, gx), target_size - 1)
        gy = min(max(0, gy), target_size - 1)
        heatmap[gy, gx] = max(heatmap[gy, gx], score)
    
    # Apply Gaussian smoothing to turn sparse points into smooth blobs.
    # This makes the target learnable by a CNN — convolutions naturally
    # produce smooth outputs, not isolated pixel spikes.
    # mode='reflect' is scipy's default and is stated here because the loss depends on
    # it: reflect conserves total mass, so the heatmap keeps summing to 1 even for
    # attention sitting on the slide border ('constant'/'nearest' would leak it away).
    # sigma=0 disables the blur, leaving CLAM's per-tile distribution untouched.
    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma, mode='reflect')
    
    return heatmap

def _load_clam(clam_weights, feature_dim, n_classes, device):
    model = CLAM_SB(n_classes=n_classes, embed_dim=feature_dim)
    ckpt = torch.load(clam_weights, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    cleaned = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    return model.to(device).eval()


def _thumb_cache_path(cache_dir, slide_id):
    return os.path.join(cache_dir, f"{slide_id}.pt")


def _cache_thumbnail(slide_id, h5_path, data_cfg, extra_svs_dir, target_size, cache_dir):
    """Extract thumbnail@target_size + slide dims ONCE and cache to disk (keyed by slide_id).

    The thumbnail is fold-independent, so this is reused across every fold — only the heatmap
    (this fold's CLAM attention) is recomputed per fold. Returns True on success.
    """
    cpath = _thumb_cache_path(cache_dir, slide_id)
    if os.path.exists(cpath):
        return True
    thumb, w, h = None, None, None
    # Option A: pre-extracted thumbnail PNG (dims estimated from coords)
    if data_cfg.get('thumbnail_dir'):
        tp = os.path.join(data_cfg['thumbnail_dir'], f"{slide_id}_thumbnail.png")
        if os.path.exists(tp):
            try:
                thumb = Image.open(tp).convert("RGB").resize((target_size, target_size), Image.Resampling.LANCZOS)
                coord_h5 = (os.path.join(data_cfg['patches_dir'], f"{slide_id}.h5")
                            if data_cfg.get('patches_dir') else h5_path)
                with h5py.File(coord_h5, 'r') as f:
                    c = np.array(f['coords'])
                w, h = int(np.max(c[:, 0])) + 256, int(np.max(c[:, 1])) + 256
            except Exception:
                thumb = None
    # Option B: SVS
    if thumb is None:
        svs_candidate = h5_path.replace('.h5', '.svs')
        if not os.path.exists(svs_candidate) and extra_svs_dir:
            svs_candidate = os.path.join(extra_svs_dir, slide_id + '.svs')
        if not os.path.exists(svs_candidate):
            return False
        try:
            slide = openslide.OpenSlide(svs_candidate)
            thumb = slide.get_thumbnail((target_size, target_size)).convert("RGB")
            thumb = thumb.resize((target_size, target_size), Image.Resampling.LANCZOS)
            w, h = slide.dimensions
            slide.close()
        except Exception as e:
            print(f"  thumb cache skip {slide_id}: {e}")
            return False
    tmp = f"{cpath}.tmp{os.getpid()}"
    torch.save({'thumb': np.asarray(thumb, dtype=np.uint8), 'w': int(w), 'h': int(h)}, tmp)
    os.replace(tmp, cpath)
    return True


def _process_slides(model, slide_list, output_dir, data_cfg, target_size, grid_size,
                    extra_svs_dir, device, thumb_cache_dir, n_workers=16):
    """Generate (thumbnail@target_size, heatmap@grid_size) pairs.

    Thumbnails are fold-independent → cached once (in parallel) in ``thumb_cache_dir`` and
    REUSED across folds; only the heatmap (this fold's CLAM attention) is recomputed per
    fold. target_size = thumbnail resolution (U-Net input); grid_size = attention grid (GT).
    """
    os.makedirs(thumb_cache_dir, exist_ok=True)

    # 1. Ensure thumbnails are cached (parallel — the SVS open is the bottleneck, paid once ever).
    todo = [(sid, h5) for sid, h5, _ in slide_list
            if not os.path.exists(_thumb_cache_path(thumb_cache_dir, sid))]
    if todo:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_cache_thumbnail, sid, h5, data_cfg, extra_svs_dir, target_size, thumb_cache_dir)
                    for sid, h5 in todo]
            for _ in tqdm(as_completed(futs), total=len(futs), desc=f"Caching thumbnails x{n_workers}"):
                pass

    # 2. Per-fold heatmaps, reusing the cached thumbnails (only a CLAM forward per slide).
    processed = 0
    for slide_id, h5_path, label in tqdm(slide_list, desc="Generating heatmaps"):
        save_path = os.path.join(output_dir, f"{slide_id}.pt")
        if os.path.exists(save_path):
            continue
        try:
            cpath = _thumb_cache_path(thumb_cache_dir, slide_id)
            if not os.path.exists(cpath):
                continue                                        # no thumbnail (no SVS/PNG)
            td = torch.load(cpath, weights_only=False)
            thumb, slide_dim = td['thumb'], (td['w'], td['h'])

            coord_h5 = (os.path.join(data_cfg['patches_dir'], f"{slide_id}.h5")
                        if data_cfg.get('patches_dir') else h5_path)
            with h5py.File(coord_h5, 'r') as f:
                coords = np.array(f['coords'])
            with h5py.File(h5_path, 'r') as f:
                features = torch.tensor(np.array(f[data_cfg['h5_key']]), dtype=torch.float32).to(device)
            if len(features) == 0:
                continue

            with torch.no_grad():
                out = model(features)
                A_raw = out[3] if len(out) >= 4 else out[1]   # CLAM_SB: logits, Y_prob, Y_hat, A_raw, results
                # data.attn_transform picks the space the U-Net is trained to imitate:
                #   'softmax' — the distribution CLAM actually pools with
                #   'sigmoid' — per-tile squashing, independent of bag size
                # Both are monotone in A, so the tile ORDER (and any top-k) is identical;
                # they differ only in the SHAPE of the target.
                # Either way the scores leave here summing to 1, so the heatmap below is a
                # distribution and ListNet can consume it without renormalizing.
                if str(data_cfg.get('attn_transform', 'softmax')).lower() == 'sigmoid':
                    att_scores = torch.sigmoid(A_raw).squeeze(0).cpu().numpy()
                    att_scores = att_scores / max(att_scores.sum(), 1e-12)
                else:
                    att_scores = torch.softmax(A_raw.float(), dim=-1).squeeze(0).cpu().numpy()
                if att_scores.ndim > 1:                        # multiclass → mean over classes
                    att_scores = att_scores.mean(axis=0)
                    att_scores = att_scores / max(att_scores.sum(), 1e-12)

            # data.blur_sigma: how far each tile's attention spreads on the grid.
            # 0 keeps CLAM's per-tile distribution exactly as it is.
            heatmap = generate_heatmap(coords, att_scores, slide_dim, grid_size,
                                       sigma=float(data_cfg.get('blur_sigma', 3.0)))
            torch.save({"thumbnail": thumb, "heatmap": heatmap,
                        "slide_id": slide_id, "label": int(label)}, save_path)
            processed += 1
        except Exception as e:
            print(f"  Error processing {slide_id}: {e}")

    return processed


def run_generate_unet_data(cfg, fold=None):
    """
    Generate U-Net training data (thumbnail → attention heatmap pairs).

    fold=None  : original behaviour — uses cfg's single CLAM model on train+val splits.
    fold=0..9  : cross-validation mode — uses CLAM_fold on that fold's train+val slides only,
                 saving to a fold-specific output directory to avoid test-set leakage.
    """
    target_size  = int(cfg.get('data', {}).get('thumb_size', 224))          # thumbnail (U-Net input)
    grid_size    = int(cfg.get('data', {}).get('unet_grid_size', target_size))  # attention grid (GT)
    extra_svs_dir = cfg['data'].get('svs_dir', None)
    _udd = cfg['data'].get('unet_data_dir', 'data/unet_data')
    thumb_cache_dir = os.path.join(os.path.dirname(_udd), f'thumb_cache_{target_size}')  # shared across folds
    feature_dim  = cfg['model']['feature_dim']
    n_classes    = cfg['model']['n_classes']
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fold is not None:
        import pandas as pd
        ckpt_dir_fold   = cfg['experiment']['checkpoint_dir']
        clam_weights = os.path.join(ckpt_dir_fold, f's_{fold}_checkpoint.pt')
        split_file  = os.path.join(ckpt_dir_fold, f'splits_{fold}.csv')
        output_dir  = cfg['data'].get('unet_data_dir', 'data/unet_data') + f'_fold_{fold}'

        print(f"[Fold {fold}] cross-validation mode: CLAM={clam_weights}")
        splits_df   = pd.read_csv(split_file)
        # Only train+val slides of this fold — never touch test slides
        train_val = set(splits_df['train'].dropna().astype(str).tolist()) | set(splits_df['val'].dropna().astype(str).tolist())
        
        # Build list of (h5_path, label) 
        csv_path = cfg['data'].get('csv_path')
        if csv_path and os.path.exists(csv_path):
            domain_df = pd.read_csv(csv_path)
            # handle different column names
            label_col = 'label' if 'label' in domain_df.columns else domain_df.columns[2]
            id_col = 'slide_id' if 'slide_id' in domain_df.columns else 'image_id'
            slide_label = dict(zip(domain_df[id_col].astype(str), domain_df[label_col]))
            dataset_dir = cfg['data']['dataset_dir']
            
            # also support the nested layout: <data_dir>/<slide_id>/original.h5
            slide_h5 = {}
            for sid in domain_df[id_col]:
                sid = str(sid)
                cand1 = os.path.join(dataset_dir, sid, "original.h5")
                cand2 = os.path.join(dataset_dir, f"{sid}.h5")
                if os.path.exists(cand1):
                    slide_h5[sid] = cand1
                elif os.path.exists(cand2):
                    slide_h5[sid] = cand2
        else:
            slide_label = {}
            slide_h5 = {}

        slide_list  = []
        for slide_id in train_val:
            h5_path = slide_h5.get(slide_id)
            if h5_path is None:
                dataset_dir = cfg['data'].get('dataset_dir', '')
                cand1 = os.path.join(dataset_dir, slide_id, "original.h5")
                cand2 = os.path.join(dataset_dir, f"{slide_id}.h5")
                h5_path = cand1 if os.path.exists(cand1) else cand2
                
            if h5_path and os.path.exists(h5_path):
                slide_list.append((slide_id, h5_path, slide_label.get(slide_id, 0)))
            else:
                print(f"  H5 not found for {slide_id}, skipping. Checked {h5_path}")
        print(f"[Fold {fold}] {len(slide_list)} slides with H5 features found.")
    else:
        clam_weights = cfg['model']['clam_pretrained_path']
        output_dir   = cfg['data'].get('unet_data_dir', 'data/unet_data')
        print("Original mode: using single CLAM on train+val splits.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading CLAM from {clam_weights} ...")
    model = _load_clam(clam_weights, feature_dim, n_classes, device)

    if fold is not None:
        processed = _process_slides(model, slide_list, output_dir,
                                    cfg['data'], target_size, grid_size, extra_svs_dir, device, thumb_cache_dir)
    else:
        # Original behaviour: iterate over train+val split CSVs
        data_cfg = cfg['data']
        processed = 0
        for split_name in ['train', 'val']:
            split_csv = data_cfg[f'{split_name}_split']
            print(f"Processing split: {split_name} ({split_csv})")
            ds = WsiTrainingDataset.from_directory(
                data_cfg['dataset_dir'], split_csv, h5_key=data_cfg['h5_key'])
            slide_list_split = []
            for i in range(len(ds)):
                h5_path, _, label, _ = ds[i]
                slide_id = os.path.splitext(os.path.basename(h5_path))[0]
                if slide_id == 'original':
                    slide_id = os.path.basename(os.path.dirname(h5_path))
                slide_list_split.append((slide_id, h5_path, label))
            processed += _process_slides(model, slide_list_split, output_dir,
                                         data_cfg, target_size, grid_size, extra_svs_dir, device, thumb_cache_dir)

    print(f"Done. Generated {processed} new pairs in {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index (0-9). Omit for original single-CLAM behaviour.')
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_generate_unet_data(cfg, fold=args.fold)

if __name__ == "__main__":
    main()
