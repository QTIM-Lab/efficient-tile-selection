import os
import sys
import glob
import h5py
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np

from tileselect.utils.suppress_tiff import openslide  # noqa: F401
import pandas as pd
import argparse


from tileselect.utils.conch_model import create_model_from_pretrained
from tileselect.data.dataset import WsiTrainingDataset # Reuse robust logic

def get_args():
    parser = argparse.ArgumentParser(description="Extract Thumbnail Features using Conch")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml (to resolve paths)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to CONCH weights (default: model.conch_weights in the config)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()

def load_config(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def extract_for_dataset(dataset, model, transform, device):
    """
    Iterates over a WsiTrainingDataset and processes thumbnails.
    """
    success = 0
    skipped = 0
    failed = 0
    
    print(f"Dataset has {len(dataset)} slides.")
    
    for i in tqdm(range(len(dataset))):
        try:
            # dataset[i] returns (h5_path, svs_path, label, num_patches)
            h5_path, svs_path, label, num_patches = dataset[i]
            
            if not os.path.exists(h5_path):
                failed += 1
                continue
                
            # Check if done
            needs_processing = True
            with h5py.File(h5_path, 'r') as f:
                if 'thumbnail_feature' in f and f['thumbnail_feature'].shape[0] == 768:
                    # check dim
                    needs_processing = False
            
            if not needs_processing:
                skipped += 1
                continue
                
            # Extract Thumbnail
            if not os.path.exists(svs_path):
                # print(f"SVS missing: {svs_path}")
                failed += 1
                continue
                
            slide = openslide.OpenSlide(svs_path)
            # Use same size logic as in Conch usage (224 or 448?) -> model was loaded with img_size=224 in main
            # We get a slightly larger thumb and resize cleanly
            thumb = slide.get_thumbnail((1024, 1024)).convert("RGB") 
            slide.close()
            
            # Transform
            # The transform expects a PIL image
            input_tensor = transform(thumb).unsqueeze(0).to(device) # [1, 3, 224, 224]
            
            # Inference
            with torch.no_grad():
                # Conch output shape [1, 768]
                feature = model(input_tensor)
                feature_np = feature.cpu().numpy()[0] # [768]
                
            # Provide normalization? Conch usually outputs normalized?
            # It has a layer norm at the end, so yes.
            
            # Write to H5
            with h5py.File(h5_path, 'r+') as f:
                if 'thumbnail_feature' in f:
                    del f['thumbnail_feature']
                f.create_dataset('thumbnail_feature', data=feature_np)
                
            success += 1
            
        except Exception as e:
            # print(f"Error processing {i}: {e}")
            failed += 1
            
    print(f"Finished: {success} processed, {skipped} skipped, {failed} failed.")

def run_extract_thumbs(cfg, device='cuda', weights=None):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # 1. Load Model
    if weights is None:
        # Try config or fallback
        weights = cfg.get('model', {}).get('conch_weights')
        if weights is None:
            raise ValueError("No CONCH weights given: pass --weights or set model.conch_weights")

    print(f"Loading Conch model from {weights}...")
    try:
        model, transform = create_model_from_pretrained(weights, img_size=224) 
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. Iterate ALL splits to ensure coverage
    splits = ['train_split', 'val_split', 'test_split']
    
    for split_key in splits:
        if split_key in cfg['data']:
            csv_path = cfg['data'][split_key]
            print(f"\n--- Processing {split_key}: {csv_path} ---")
            
            ds = WsiTrainingDataset.from_directory(
                cfg['data']['dataset_dir'],
                csv_path,
                h5_key=cfg['data']['h5_key']
            )
            
            extract_for_dataset(ds, model, transform, device)

def main():
    args = get_args()
    cfg = load_config(args.config)
    
    run_extract_thumbs(cfg, args.device, args.weights)

if __name__ == "__main__":
    main()
