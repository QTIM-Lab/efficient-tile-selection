import os
import csv
import h5py
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset

class WsiTrainingDataset(Dataset):
    """
    Flexible dataset to feed WSIs (precomputed H5 features) to the pipeline.

    Two usage modes:
    - Single-slide mode: pass explicit paths + label.
      WsiTrainingDataset.from_single(svs_path, h5_path, label, h5_key='features')
    - Multi-slide mode: pass a directory and a CSV with labels.
      The directory must contain matching pairs of .svs and .h5; pairs are matched
      by filename stem (e.g., "slide_001.svs" ↔ "slide_001.h5").
      The CSV must have columns: `slide_id` (the stem) and `label` (0/1).

    Returns: (h5_path, svs_path, label, num_patches), where num_patches is read
    from the H5 dataset `h5_key` (default 'features').
    """

    def __init__(self, items: List[Dict], h5_key: str = 'features'):
        self.items = items
        self.h5_key = h5_key

    @classmethod
    def from_single(cls, svs_path: str, h5_path: str, label: int, h5_key: str = 'features'):
        svs_path = os.path.abspath(svs_path)
        h5_path = os.path.abspath(h5_path)
        item = {'svs_path': svs_path, 'h5_path': h5_path, 'label': int(label)}
        return cls([item], h5_key=h5_key)

    @classmethod
    def from_directory(cls, data_dir: str, labels_csv: str, h5_key: str = 'features'):
        data_dir = os.path.abspath(data_dir)
        labels_csv = os.path.abspath(labels_csv)

        items: List[Dict] = []
        with open(labels_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            # Check required columns
            if 'slide_id' not in reader.fieldnames or 'label' not in reader.fieldnames:
                raise ValueError("labels_csv must have columns: slide_id,label")
            
            has_feat_path = 'features_path' in reader.fieldnames
            
            for row in reader:
                sid = str(row['slide_id']).strip()
                lbl = int(row['label'])
                
                # Logic 1: Use absolute path from CSV if available
                if has_feat_path and row['features_path']:
                    h5_path = row['features_path']
                    # Verify existence
                    if not os.path.exists(h5_path):
                         # Fallback to local 'data_dir' search if absolute fails?
                         # Or just warn.
                         # Try joining with data_dir if relative?
                         pass
                else:
                    # Logic 2: Search in data_dir
                    h5_path = os.path.join(data_dir, f"{sid}.h5")
                    if not os.path.exists(h5_path):
                        pt_path = os.path.join(data_dir, f"{sid}.pt")
                        nested_path = os.path.join(data_dir, str(sid), "original.h5")
                        if os.path.exists(pt_path):
                            h5_path = pt_path
                        elif os.path.exists(nested_path):
                            h5_path = nested_path
                        else:
                            # Check inside subdirectory if matches sid
                            sub_h5 = os.path.join(data_dir, str(sid), f"{sid}.h5")
                            if os.path.exists(sub_h5):
                                h5_path = sub_h5

                # SVS Path
                svs_path = None
                if 'svs_path' in row and row['svs_path']:
                    svs_path = row['svs_path']
                
                if not svs_path or not os.path.exists(svs_path):
                     svs_path = h5_path.replace('.h5', '.svs')
                     if not os.path.exists(svs_path):
                        # Try data dir
                        svs_path = os.path.join(data_dir, f"{sid}.svs")
                
                # If still not found, just use None or a placeholder since user says they don't need it
                if not os.path.exists(svs_path):
                    svs_path = "MISSING_SVS_FILE"
                
                # Check existence of H5
                if os.path.exists(h5_path):
                     items.append({'svs_path': svs_path, 'h5_path': h5_path, 'label': int(lbl)})

        if not items:
            raise RuntimeError(f"No valid items found in {labels_csv}. Checked paths.")

        return cls(items, h5_key=h5_key)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, str, int, int]:
        rec = self.items[idx]
        h5_path = rec['h5_path']
        svs_path = rec['svs_path']
        label = rec['label']
        # Get real number of patches (before padding)
        num_patches = 0
        try:
            if h5_path.endswith('.pt'):
                import torch
                num_patches = int(torch.load(h5_path).shape[0])
            else:
                with h5py.File(h5_path, 'r') as f:
                    if self.h5_key not in f:
                        raise KeyError(f"H5 missing dataset '{self.h5_key}' in {h5_path}")
                    num_patches = int(f[self.h5_key].shape[0])
        except Exception as e:
            print(f"[WsiTrainingDataset] ERROR reading num_patches from {h5_path}: {e}")
        return h5_path, svs_path, label, num_patches
