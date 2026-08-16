import os
import yaml
import sys

# Add project root to sys.path

import argparse
import csv
import random
import math

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def write_csv(path, headers, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def create_splits(config, domain):
    data_cfg = config['data']
    experiment_cfg = config['experiment']
    
    seed = experiment_cfg.get('seed', 42)
    random.seed(seed)
    
    from tileselect import paths
    labels_csv = data_cfg.get('labels_csv') or os.path.join(paths.data_dir(config), 'domain_labels.csv')
    if not os.path.exists(labels_csv):
         raise FileNotFoundError(f"Labels CSV not found at: {labels_csv}")
    
    rows = []
    headers = []
    
    print(f"Reading data from {labels_csv} for domain: {domain}")
    with open(labels_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
            if 'domain' not in headers:
                raise ValueError("CSV must have a 'domain' column.")
            domain_idx = headers.index('domain')
            for row in reader:
                if row and row[domain_idx] == domain:
                    rows.append(row)
        except Exception as e:
            raise RuntimeError(f"Error reading CSV: {e}")
        
    if not rows:
         raise RuntimeError(f"Could not find any data for domain '{domain}'.")
         
    # Identify column indices
    try:
        label_idx = headers.index('label')
        slide_id_idx = headers.index('slide_id')
        if 'patient' in headers:
            patient_idx = headers.index('patient')
        else:
            print("WARNING: 'patient' column not found. Attempting to deduce from 'slide_id'.")
            patient_idx = -1 
    except ValueError:
        raise ValueError("CSV must have 'slide_id' and 'label' columns")
            
    # 1. Group by Patient
    by_patient = {}
    
    for row in rows:
        slide_id = row[slide_id_idx]
        label = int(row[label_idx])
        
        if patient_idx != -1:
            patient_id = row[patient_idx]
        else:
            patient_id = slide_id[:12]
            
        if patient_id not in by_patient:
            by_patient[patient_id] = {'label': label, 'slides': []}
        
        if by_patient[patient_id]['label'] != label:
            print(f"WARNING: Patient {patient_id} has conflicting labels! Skipping slide {slide_id}.")
            continue
            
        by_patient[patient_id]['slides'].append(row)

    print(f"Found {len(by_patient)} unique patients.")
    
    # Stratify by Label
    patients_pos = [pid for pid, data in by_patient.items() if data['label'] == 1]
    patients_neg = [pid for pid, data in by_patient.items() if data['label'] == 0]
    
    print(f"Patient Balance: {len(patients_pos)} Positive, {len(patients_neg)} Negative.")
    
    # Split Ratios
    ratios = data_cfg.get('split_ratios', [0.7, 0.15, 0.15])
    total_r = sum(ratios)
    ratios = [r/total_r for r in ratios]
    
    def split_list(lst, rs):
        random.shuffle(lst)
        n = len(lst)
        n_train = int(n * rs[0])
        n_val = int(n * rs[1])
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    pos_train, pos_val, pos_test = split_list(patients_pos, ratios)
    neg_train, neg_val, neg_test = split_list(patients_neg, ratios)
    
    # 3. Construct Final Splits
    train_patients = pos_train + neg_train
    val_patients = pos_val + neg_val
    test_patients = pos_test + neg_test
    
    train_patients_balanced = list(train_patients)
    random.shuffle(train_patients_balanced)
    
    def expand(p_list):
        slides = []
        for pid in p_list:
            slides.extend(by_patient[pid]['slides'])
        return slides
        
    train_rows = expand(train_patients_balanced)
    val_rows = expand(val_patients)
    test_rows = expand(test_patients)
    
    # Save splits
    split_dir = data_cfg.get('split_dir', 'results/splits')
    os.makedirs(split_dir, exist_ok=True)
    
    train_path = os.path.join(split_dir, 'train.csv')
    val_path = os.path.join(split_dir, 'val.csv')
    test_path = os.path.join(split_dir, 'test.csv')

    write_csv(train_path, headers, train_rows)
    write_csv(val_path, headers, val_rows)
    write_csv(test_path, headers, test_rows)
    
    print(f"Splits created in {split_dir}:")
    print(f"  Train:            {len(train_rows)} slides (Patients: {len(train_patients)}) -> {train_path}")
    print(f"  Val:              {len(val_rows)} slides (Patients: {len(val_patients)}) -> {val_path}")
    print(f"  Test:             {len(test_rows)} slides (Patients: {len(test_patients)}) -> {test_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
    parser.add_argument('--domain', type=str, default=None,
                        help="Cohort tag to select from the CSV's optional domain column")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    create_splits(cfg, args.domain)

if __name__ == "__main__":
    main()
