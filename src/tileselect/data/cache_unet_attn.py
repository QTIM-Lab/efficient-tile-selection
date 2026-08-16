import os, sys, argparse, glob, yaml, h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from tileselect.utils.suppress_tiff import openslide  # noqa: F401
from tileselect.models.unet import SimpleUNet

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def find_svs(h5_path):
    attempts = [
        h5_path.replace('.h5', '.svs'),
        h5_path.replace('/features/', '/data/').replace('.h5', '.svs'),
        h5_path.replace('/features/', '/data/').replace('.h5', '.ndpi'),
        h5_path.replace('/features/', '/raw_data/').replace('.h5', '.svs')
    ]
    for atm in attempts:
        if os.path.exists(atm): return atm
    
    d = os.path.dirname(h5_path).replace('/features/', '/data/')
    b = os.path.basename(h5_path).split('.')[0]
    matched = glob.glob(f"{d}/*{b}*")
    for m in matched:
        if m.endswith(('.svs', '.ndpi', '.tiff')): return m
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--domain', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = SimpleUNet(n_channels=3, n_classes=1)
    unet.load_state_dict(torch.load(cfg['model']['unet_path'], map_location=device))
    unet.to(device)
    unet.eval()

    output_dir = os.path.join("CLAM", "attn_cache_unet", args.domain)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = cfg['data']['labels_csv']
    df = pd.read_csv(csv_path)
    # A multi-cohort CSV can carry a 'domain' column; keep only the requested one.
    if 'domain' in df.columns:
        df = df[df['domain'] == args.domain]

    print(f"Generating UNet attention cache for domain {args.domain}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        slide_id = str(row['slide_id'])
        cache_path = os.path.join(output_dir, f"{slide_id}.pt")
        if os.path.exists(cache_path): continue

        if 'features_path' in row: h5_path = row['features_path']
        else: h5_path = os.path.join(cfg['data']['dataset_dir'], f"{slide_id}.h5")
        if not os.path.exists(h5_path): continue

        svs_path = None
        if 'svs_path' in row and pd.notnull(row['svs_path']) and os.path.exists(str(row['svs_path'])):
            svs_path = str(row['svs_path'])
        else:
            svs_path = find_svs(h5_path)

        if svs_path is None or not os.path.exists(svs_path): continue

        try:
            with h5py.File(h5_path, 'r') as f: coords = np.array(f['coords'])
            slide = openslide.OpenSlide(svs_path)
            target_size = 224
            thumb = slide.get_thumbnail((target_size, target_size)).convert("RGB")
            thumb = thumb.resize((target_size, target_size), Image.Resampling.LANCZOS)
            thumb_tensor = transforms.ToTensor()(thumb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = unet(thumb_tensor)
                prob = torch.sigmoid(logits)
                heatmap_np = prob[0, 0].cpu().numpy()

            w_slide, h_slide = slide.dimensions
            unet_scores = []
            
            for index in range(len(coords)):
                tx, ty = coords[index]
                nx = tx / w_slide
                ny = ty / h_slide
                gx = min(target_size-1, int(nx * target_size))
                gy = min(target_size-1, int(ny * target_size))
                unet_scores.append(float(heatmap_np[gy, gx]))
                
            A_scores = torch.tensor(unet_scores, dtype=torch.float32)
            torch.save(A_scores, cache_path)
            slide.close()
        except Exception as e:
            pass

if __name__ == '__main__':
    main()
