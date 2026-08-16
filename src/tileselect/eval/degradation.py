import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import h5py
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import glob
from tileselect.utils.suppress_tiff import openslide  # noqa: F401
from tileselect.data.dataset import WsiTrainingDataset
from tileselect.models.clam import CLAM_SB
from tileselect.models.unet import SimpleUNet
from tileselect.utils.tiles import coords_to_grid
from tileselect import paths
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_subset(model, features, label):
    with torch.no_grad():
        logits, Y_prob, Y_hat, _, _ = model(features, instance_eval=False)
        return Y_prob.cpu().numpy()[0], Y_hat.item()


def macenko_stains(img, beta=0.15, alpha=1.0):
    """Per-slide H&E stain vectors by the Macenko method (PCA on optical density).

    The standard unsupervised deconvolution in digital pathology: project the optical
    density of the tissue pixels onto its two principal directions and take the extreme
    angles as the stain vectors. Estimating them per slide absorbs staining and scanner
    variation that a fixed matrix cannot. Returns [3, 2] with haematoxylin first, or None
    when the thumbnail holds too little tissue.
    """
    od = -np.log(np.clip(img.reshape(-1, 3), 1e-6, 1.0))
    od = od[od.sum(1) > beta]
    if len(od) < 100:
        return None
    _, V = np.linalg.eigh(np.cov(od.T))
    V = V[:, [2, 1]]
    for j in (0, 1):
        if V[0, j] < 0:
            V[:, j] *= -1
    phi = np.arctan2(*(od @ V)[:, ::-1].T)
    a, b = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = V @ np.array([np.cos(a), np.sin(a)])
    v2 = V @ np.array([np.cos(b), np.sin(b)])
    HE = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
    return HE / np.linalg.norm(HE, axis=0)


def stain_map_from_thumb(thumb_tensor, out_size, channel=0, sigma=8.0, macenko=True):
    """Stain-concentration map of the thumbnail, smoothed, resized to the attention grid.

    A training-free selector: it scores tiles by stain color alone, from the same
    thumbnail the U-Net sees. It answers whether the U-Net learns anything a plain color
    heuristic would not already give.

    channel 0 = haematoxylin, which binds nucleic acids and so tracks cellularity. On
    these slides a tile covers only 1.6-5.6 thumbnail pixels and a nucleus 0.06-0.22, so
    this is NOT nuclear counting — it is regional haematoxylin dominance. Measured
    Spearman against CLAM's attention: 0.51, against 0.53 for the pretrained U-Net.
    channel 2 (fixed matrix only) = DAB, the deconvolution residual on H&E (0.07).

    sigma smooths at roughly the tile scale, mirroring the blur the target receives.
    """
    rgb = thumb_tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    od = -np.log(np.clip(rgb, 1e-6, 1.0))
    HE = macenko_stains(rgb) if (macenko and channel < 2) else None
    if HE is not None:
        conc = np.linalg.lstsq(HE, od.reshape(-1, 3).T, rcond=None)[0].T
        m = np.maximum(conc[:, channel], 0).reshape(od.shape[:2])
    else:
        m = (od.reshape(-1, 3) @ _HED_FROM_RGB).reshape(od.shape)[:, :, channel]
    if sigma > 0:
        m = gaussian_filter(m.astype(np.float32), sigma=sigma, mode='reflect')
    d = torch.from_numpy(m)[None, None]
    if d.shape[-1] != out_size:
        d = torch.nn.functional.interpolate(d, size=(out_size, out_size),
                                            mode='bilinear', align_corners=False)
    return d[0, 0].numpy().astype(np.float64)


def weighted_sample_from_logw(log_w, k):
    """Sample k DISTINCT items with probability proportional to ``exp(log_w)``,
    WITHOUT replacement, via the Gumbel-top-k trick: perturb the log-weights with
    i.i.d. Gumbel noise and take the top-k keys. O(N), exact. Returns indices.
    Favours high-score items stochastically while always using the full budget.
    """
    lw = np.asarray(log_w, dtype=np.float64)
    k = min(int(k), lw.shape[0])
    if k <= 0:
        return np.array([], dtype=int)
    keys = lw + np.random.gumbel(size=lw.shape[0])
    return np.argpartition(keys, -k)[-k:]


def weighted_sample_no_replace(scores, k):
    """As above, for weights given on the probability scale."""
    w = np.clip(np.asarray(scores, dtype=np.float64), 1e-12, None)
    return weighted_sample_from_logw(np.log(w), k)


def grid_subsample(coords, k, rng=None):
    """Pick ~k tiles on a regular spatial lattice: one per cell, nearest the cell centre.

    The training-free counterpart to random sampling that removes its one obvious defect.
    Uniform sampling clusters — with k=128 out of 6,603 tiles some regions get several
    neighbours and others none — so a slide's coverage varies from draw to draw. A lattice
    guarantees spread, which isolates "is spatial coverage what matters?" from "is it the
    scores that matter?". If a plain grid matched the learned selector, the selector would
    be doing nothing a stride could not.

    The lattice origin is jittered per call so repeated draws give an error band, the same
    way the weighted strategies do; without it the strategy would be deterministic and
    plot with no band beside curves that have one.
    """
    rng = rng or np.random
    n = len(coords)
    if k >= n:
        return np.arange(n)
    xy = np.asarray(coords, dtype=np.float64)
    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1.0)
    # Cell counts proportional to the tissue's aspect ratio, so cells stay roughly square.
    ar = span[0] / span[1]
    ny = max(1, int(round(np.sqrt(k / max(ar, 1e-6)))))
    nx = max(1, int(round(k / ny)))
    off = rng.random_sample(2) if hasattr(rng, 'random_sample') else rng.random(2)
    ix = np.clip(((xy[:, 0] - lo[0]) / span[0] * nx + off[0]).astype(int), 0, nx)
    iy = np.clip(((xy[:, 1] - lo[1]) / span[1] * ny + off[1]).astype(int), 0, ny)
    cell = iy * (nx + 1) + ix
    # One tile per occupied cell: the one closest to that cell's centroid.
    picked = []
    for c in np.unique(cell):
        m = np.flatnonzero(cell == c)
        if len(m) == 1:
            picked.append(m[0]); continue
        d = xy[m] - xy[m].mean(0)
        picked.append(m[np.argmin((d * d).sum(1))])
    picked = np.array(picked)
    if len(picked) > k:                       # too many occupied cells: thin at random
        picked = rng.choice(picked, k, replace=False)
    elif len(picked) < k:                     # tissue sparser than the lattice: top up
        rest = np.setdiff1d(np.arange(n), picked)
        picked = np.concatenate([picked, rng.choice(rest, k - len(picked), replace=False)])
    return picked


def find_svs(h5_path, svs_dir=None):
    attempts = [
        h5_path.replace('.h5', '.svs'),
        h5_path.replace('/features/', '/data/').replace('.h5', '.svs'),
        h5_path.replace('/features/', '/data/').replace('.h5', '.ndpi'),
        h5_path.replace('/features/', '/raw_data/').replace('.h5', '.svs')
    ]
    for atm in attempts:
        if os.path.exists(atm): return atm

    # Fallback: the configured svs_dir
    if svs_dir:
        slide_id = os.path.splitext(os.path.basename(h5_path))[0]
        candidate = os.path.join(svs_dir, slide_id + '.svs')
        if os.path.exists(candidate):
            return candidate

    d = os.path.dirname(h5_path).replace('/features/', '/data/')
    b = os.path.basename(h5_path).split('.')[0]
    matched = glob.glob(f"{d}/*{b}*")
    for m in matched:
        if m.endswith(('.svs', '.ndpi', '.tiff')): return m
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/template.yaml')
    parser.add_argument('--domain', type=str, default='cohort',
                        help='Cohort tag: selects rows from the label CSV and names the outputs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write eval outputs; defaults to <data_root>/eval_results.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--proxy', type=str, default='all',
                        choices=['unet', 'attn_sample', 'unet_sample', 'hema', 'grid', 'all'],
                        help='Which proxy model(s) to use. "all" runs all available proxies in one pass.')
    parser.add_argument('--clam_weights', type=str, default=None, help='Path to CLAM weights to override config')
    parser.add_argument('--clam_split_file', type=str, default=None, help='Path to CLAM splits.csv (e.g. splits_0.csv) to filter slides by the chosen --split (e.g. test).')
    parser.add_argument('--clam_override_dir', type=str, default=None,
                        help='Directory with per-fold CLAM checkpoints (fold_0/checkpoint.pt, fold_1/checkpoint.pt, …). '
                             'Overrides the per-fold checkpoint when use_folds=True.')
    parser.add_argument('--fold', type=int, default=None, help='Run only this fold index (e.g. 0).')
    parser.add_argument('--n_resample', type=int, default=None,
                        help='Fallback for eval.n_resample if not set in the config YAML.')
    parser.add_argument('--budget_mode', type=str, default=None, choices=['pct', 'k'],
                        help="Budget axis: 'pct' = fraction of each slide's tiles (default), "
                             "'k' = same ABSOLUTE number of tiles for every slide.")
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated budget list overriding the defaults for the chosen mode.')
    parser.add_argument('--autoplot', action='store_true',
                        help='Also drop a degradation figure beside the CSVs. Off by default: with '
                             'folds running as separate jobs it would be rebuilt from whichever '
                             'subset has finished, and the early folds are the high-AUROC ones. '
                             'Build the comparison figures from the CSVs once all folds are in.')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Eval-time params live in the config's `eval:` section (preferred over CLI flags):
    #   eval.n_resample : resampling trials for the stochastic strategies (default 10)
    #   eval.budget_mode: 'pct' (default) or 'k'; eval.budgets: explicit budget list
    eval_cfg = cfg.get('eval', {}) or {}
    n_resample = int(eval_cfg.get('n_resample', args.n_resample if args.n_resample is not None else 10))
    budget_mode = str(args.budget_mode or eval_cfg.get('budget_mode', 'pct')).lower()
    # One-vs-rest scoring for multiclass cohorts; ignored when the task is binary.
    binarize_multiclass = bool(eval_cfg.get('binarize_multiclass', True))
    #   eval.score_transform : how the U-Net map becomes SAMPLING log-weights.
    #     'softmax' → the U-Net's own distribution, which is what ListNet calibrated.
    #                 Its log-weight is the logit itself: log softmax(x)_i = x_i - logsumexp(x),
    #                 and the shared constant cancels when sampling.
    #     'sigmoid' (default, legacy) → log(sigmoid(logits)).
    #   Top-k is unaffected either way: both transforms are monotone in the logit, so the
    #   ranking is identical; only the weighted sampling changes.
    score_transform = str(eval_cfg.get('score_transform', 'sigmoid')).lower()
    if score_transform not in ('sigmoid', 'softmax'):
        raise SystemExit(f"eval.score_transform must be 'softmax' or 'sigmoid', "
                         f"got {score_transform!r}")

    if args.output_dir is None:
        args.output_dir = paths.eval_results_dir(cfg)

    use_folds = cfg.get('experiment', {}).get('use_folds', False)
    use_cv = cfg.get('experiment', {}).get('use_cv', False)
    num_folds = 10 if (use_folds or use_cv) else 1
    
    # Fix seeds so random/attention baselines are identical across proxy runs
    seed = int(cfg.get('experiment', {}).get('seed', 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    n_classes = int(cfg['model']['n_classes'])
    feature_dim = int(cfg['model']['feature_dim'])
    
    svs_dir = cfg['data'].get('svs_dir', None)
    thumb_size = int(cfg['data'].get('thumb_size', 224))            # U-Net input resolution
    grid_size = int(cfg['data'].get('unet_grid_size', thumb_size))  # U-Net output / attention grid
    run_all = (args.proxy == 'all')

    # --- UNet architecture (weights loaded per-fold when use_folds) ---
    unet_model = None
    if args.proxy in ('unet', 'all'):
        unet_model = SimpleUNet(n_channels=3, n_classes=1,
                                out_size=(grid_size if grid_size != thumb_size else None))
        if not use_folds:
            unet_weights = cfg['model'].get('unet_path')
            if unet_weights and os.path.exists(unet_weights):
                unet_model.load_state_dict(torch.load(unet_weights, map_location=device))
                print(f"Loaded UNet model from {unet_weights}")
            else:
                print("WARNING: UNet weights not found!")
        unet_model.to(device).eval()

    # Budget axis. 'pct' keeps a per-slide FRACTION (historical behaviour); 'k' keeps the
    # same ABSOLUTE tile count for every slide, which is what a deployed system actually
    # does and removes the bag-size heterogeneity (real cohorts span two orders of
    # magnitude in tiles per slide).
    # In both modes the FULL-budget reference is the first entry: 1.0 for pct, 0 for k
    # (0 = sentinel meaning "all tiles"), so degradation is always measured against it.
    if budget_mode == 'k':
        budgets = [0, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
    else:
        budgets = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    if args.budgets:
        parsed = [float(x) for x in args.budgets.split(',') if x.strip()]
        budgets = [int(x) for x in parsed] if budget_mode == 'k' else parsed
    elif eval_cfg.get('budgets'):
        budgets = [int(x) for x in eval_cfg['budgets']] if budget_mode == 'k' else [float(x) for x in eval_cfg['budgets']]

    # Column name used in fold_*_metrics.csv for the budget axis.
    bcol = 'k' if budget_mode == 'k' else 'pct'

    def budget_to_k(budget, N):
        """Tiles to keep for this slide at this budget."""
        if budget_mode == 'k':
            return N if int(budget) <= 0 else min(int(budget), N)
        return max(1, int(N * budget))

    print(f"Budget axis: mode={budget_mode} ({bcol}), {len(budgets)} points: {budgets}")
    print(f"U-Net sampling log-weights: {score_transform}")

    strategies = ['random', 'grid', 'attn', 'unet']
    if args.proxy in ('hema', 'all'):
        strategies += ['hema', 'hema_sample']
    if args.proxy in ('attn_sample', 'all'):
        strategies += ['attn_sample']
    if args.proxy in ('unet_sample', 'all'):
        strategies += ['unet_sample']

    def budget_dir(domain, strat, budget):
        return os.path.join(args.output_dir, f"EVAL_{domain}_gradual_log_{strat}_{bcol}_{budget}")

    def update_and_plot(current_fold=None):
        """Write the per-fold metric CSVs; the figure is opt-in.

        Every eval used to drop a degradation_curves_<domain>_<split>_<proxy>.png next to
        the results, one per fold as they landed. With ten folds running as ten separate
        SLURM tasks that file is rewritten from whatever subset happens to be finished,
        so it shows a partial, shifting picture — and the folds that finish first are the
        high-AUROC ones, which makes any partial mean optimistic. The comparison figures
        are built afterwards from the CSVs, over all ten folds at once. Pass --autoplot
        to get the old behaviour back for a one-off run.
        """
        if not getattr(args, 'autoplot', False):
            return
        from tileselect.eval import plot as _plot
        domain_name = args.domain
        # Single-fold run → the cross-fold band is 0, so show the resampling band instead.
        band = 'resample' if args.fold is not None else 'fold'
        summary_df = _plot.build_summary(
            args.output_dir, strategies,
            max_fold=None if current_fold is None else current_fold, band=band)
        if summary_df.empty:
            return
        try:
            plot_path = _plot.plot_degradation(
                summary_df, domain_name, split=args.split,
                plots_dir=paths.plots_dir(cfg),
                proxy_tag=('combined' if run_all else args.proxy),
                current_fold=current_fold, band=band)
            if current_fold is None:
                print(f"Final evaluation complete. Plot saved to {plot_path}")
            else:
                print(f"[Fold {current_fold}] Partial plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

    for fold in range(num_folds):
        if args.fold is not None and fold != args.fold:
            continue
        domain_name = args.domain

        # Skip fold if all fold_{fold}.csv already exist for every strat/pct
        fold_metrics_path = os.path.join(args.output_dir, f"fold_{fold}_metrics.csv")
        current_strategies = strategies.copy()
        # which stain channel (if any) this run selects with; used by both stain blocks
        _stain = 'hema' if 'hema' in current_strategies else None
        if os.path.exists(fold_metrics_path):
            saved = pd.read_csv(fold_metrics_path)
            current_strategies = [s for s in strategies if s not in saved['strategy'].unique()]
            if not current_strategies:
                print(f"[Fold {fold}] Already complete, skipping.")
                continue
            print(f"[Fold {fold}] Missing strategies: {current_strategies}. Evaluating only these.")

        print(f"\n{'='*30}\nEvaluating Fold {fold}\n{'='*30}")

        clam_split_file = args.clam_split_file
        clam_weights = args.clam_weights

        if use_folds or use_cv:
            # Per-fold split and CLAM checkpoint come from the cross-validation
            # checkpoint directory (experiment.checkpoint_dir), which uses the
            # splits_{fold}.csv / s_{fold}_checkpoint.pt layout.
            # Explicit --clam_split_file / --clam_weights, if given, take precedence.
            ckpt_dir = cfg['experiment'].get('checkpoint_dir')
            if not clam_split_file and ckpt_dir:
                clam_split_file = os.path.join(ckpt_dir, f"splits_{fold}.csv")
            if not clam_weights:
                if args.clam_override_dir:
                    clam_weights = os.path.join(args.clam_override_dir, f"fold_{fold}", "checkpoint.pt")
                elif ckpt_dir:
                    clam_weights = os.path.join(ckpt_dir, f"s_{fold}_checkpoint.pt")
            print(f"[Fold {fold}] split={clam_split_file}  weights={clam_weights}")
        else:
            clam_weights = clam_weights if clam_weights else cfg['model'].get('clam_pretrained_path')

        # Load CLAM Model for this fold
        clam_model = CLAM_SB(n_classes=n_classes, embed_dim=feature_dim, dropout=cfg['model'].get('dropout', 0.25))
        if clam_weights:
            ckpt = torch.load(clam_weights, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            clam_model.load_state_dict(cleaned_state, strict=False)
            print(f"Loaded CLAM model from {clam_weights}")
        clam_model.to(device)
        clam_model.eval()

        needs_unet_model = any(s in current_strategies for s in ('unet', 'unet_sample'))

        # Load fold-specific UNet weights when running k-fold evaluation
        if (use_folds or use_cv) and unet_model is not None and needs_unet_model:
            base_unet_path = cfg['model'].get('unet_path', '')
            fold_unet_path = base_unet_path.replace('.pt', f'_fold_{fold}.pt')
            if os.path.exists(fold_unet_path):
                unet_model.load_state_dict(torch.load(fold_unet_path, map_location=device))
                print(f"Loaded fold-specific UNet from {fold_unet_path}")
            else:
                try:
                    unet_model.load_state_dict(torch.load(base_unet_path, map_location=device))
                    print(f"WARNING: fold UNet not found, loading fallback {base_unet_path}")
                except FileNotFoundError:
                    print(f"WARNING: No UNet weights found at {fold_unet_path} or {base_unet_path}")
            unet_model.eval()

        # Load Dataset for this fold
        if clam_split_file and os.path.exists(clam_split_file):
            clam_splits = pd.read_csv(clam_split_file)
            if args.split in clam_splits.columns:
                target_slides = [str(x).replace('.0', '') if str(x).endswith('.0') else str(x) for x in clam_splits[args.split].dropna().tolist()]
            else:
                print(f"WARNING: Split '{args.split}' not found in {clam_split_file}. Columns: {clam_splits.columns.tolist()}. Proceeding with all.")
                target_slides = None
                
            domain_csv = cfg['data'].get('csv_path', 'data/domain_labels.csv')
            df_domain = pd.read_csv(domain_csv)
            if target_slides:
                df_domain['slide_id'] = df_domain['slide_id'].astype(str)
                df_domain = df_domain[df_domain['slide_id'].isin(target_slides)]
            else:
                df_domain = df_domain[df_domain['domain'] == domain_name]
                
            temp_csv = f"/tmp/{domain_name}_{args.split}_split_{fold}_{os.getpid()}.csv"
            df_domain.to_csv(temp_csv, index=False)
            dataset = WsiTrainingDataset.from_directory(cfg['data']['dataset_dir'], temp_csv, h5_key=cfg['data']['h5_key'])
            print(f"Loaded {len(dataset)} slides from CLAM split file: {clam_split_file} (split: {args.split}).")
        elif args.split == 'all':
            domain_csv = cfg['data'].get('csv_path', 'data/domain_labels.csv')
            df_domain = pd.read_csv(domain_csv)
            df_domain = df_domain[df_domain['domain'] == domain_name]
            
            temp_csv = f"/tmp/{domain_name}_split_{fold}_{os.getpid()}.csv"
            df_domain.to_csv(temp_csv, index=False)
            dataset = WsiTrainingDataset.from_directory(cfg['data']['dataset_dir'], temp_csv, h5_key=cfg['data']['h5_key'])
            print(f"Loaded {len(dataset)} slides for domain {domain_name} evaluation.")
        else:
            split_dir = cfg['data'].get('split_dir', 'results/splits')
            split_csv = os.path.join(split_dir, f"{args.split}.csv")
            if not os.path.exists(split_csv):
                raise FileNotFoundError(f"Split file not found: {split_csv}")
            dataset = WsiTrainingDataset.from_directory(cfg['data']['dataset_dir'], split_csv, h5_key=cfg['data']['h5_key'])
            print(f"Loaded {len(dataset)} slides from global split {args.split}.")

        results = {strat: {b: {'probs': [], 'labels': [], 'slide_ids': []} for b in budgets} for strat in current_strategies}
        proxy_skipped = {s: 0 for s in current_strategies if s not in ('random', 'attn', 'attn_sample')}

        for i in tqdm(range(len(dataset)), desc=f"Evaluating slides (Fold {fold})"):
            h5_path, svs_path, label, num_patches = dataset[i]
            label = int(label)
            
            try:
                with h5py.File(h5_path, 'r') as f:
                    features = torch.tensor(f[cfg['data'].get('h5_key', 'features')][:]).float().to(device)
                    
                    slide_id_coord = os.path.basename(h5_path).split('.')[0]
                    if slide_id_coord == "original":
                        slide_id_coord = os.path.basename(os.path.dirname(h5_path))
                    
                    if cfg['data'].get('patches_dir'):
                        coord_h5 = os.path.join(cfg['data']['patches_dir'], f"{slide_id_coord}.h5")
                        with h5py.File(coord_h5, 'r') as f_c:
                            coords = np.array(f_c['coords']) if 'coords' in f_c else None
                    else:
                        coords = np.array(f['coords']) if 'coords' in f else None
            except Exception as e:
                print(f"Error loading {h5_path}: {e}")
                continue
                
            N = features.shape[0]
            if N == 0: continue
            
            if coords is not None and len(coords) != N:
                print(f"Skipping {h5_path}: Features length ({N}) != coords length ({len(coords)})")
                continue

            A_prob = None
            unet_scores = None
            unet_logw = None
            stain_scores = None
            need_h = ('attn' in current_strategies or 'attn_sample' in current_strategies)
            if need_h:
                with torch.no_grad():
                    A, _ = clam_model.attention_net(features)
                    if 'attn' in current_strategies or 'attn_sample' in current_strategies:
                        A_prob = torch.sigmoid(A.squeeze()).cpu().numpy()

            unet_scores         = None
            stain_scores          = None
            slide_id = os.path.basename(h5_path).split('.')[0]
            if slide_id == "original":
                slide_id = os.path.basename(os.path.dirname(h5_path))

            need_svs = (
                ('unet' in current_strategies and unet_model is not None) or
                ('unet_sample' in current_strategies and unet_model is not None) or
                any(s in current_strategies for s in ('hema', 'hema_sample'))
            )
            if need_svs:
                thumb_tensor = None
                w_slide, h_slide = None, None
                target_size = thumb_size          # thumbnail resolution (U-Net input)
                
                # Option A: SVS file
                svs_file = find_svs(h5_path, svs_dir=svs_dir)
                if svs_file:
                    try:
                        slide = openslide.OpenSlide(svs_file)
                        thumb = slide.get_thumbnail((target_size, target_size)).convert("RGB")
                        thumb = thumb.resize((target_size, target_size), Image.Resampling.LANCZOS)
                        thumb_tensor = transforms.ToTensor()(thumb).unsqueeze(0).to(device)
                        w_slide, h_slide = slide.dimensions
                        slide.close()
                    except Exception as e:
                        print(f"Error reading SVS {svs_file}: {e}")
                
                # Option B: Pre-extracted thumbnails and patches
                elif cfg['data'].get('thumbnail_dir'):
                    thumb_cand1 = os.path.join(cfg['data']['thumbnail_dir'], f"{slide_id}_thumbnail.png")
                    thumb_cand2 = os.path.join(cfg['data']['thumbnail_dir'], f"{slide_id}.png")
                    thumb_path = thumb_cand1 if os.path.exists(thumb_cand1) else thumb_cand2
                    
                    if os.path.exists(thumb_path):
                        try:
                            thumb_pil = Image.open(thumb_path).convert("RGB")
                            # We need original dimensions to map coordinates.
                            # Look at coords max values if dimensions are not known
                            if coords is not None and len(coords) > 0:
                                w_slide = int(np.max(coords[:, 0])) + 512  # approx
                                h_slide = int(np.max(coords[:, 1])) + 512
                            else:
                                w_slide, h_slide = thumb_pil.size
                            
                            thumb = thumb_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
                            thumb_tensor = transforms.ToTensor()(thumb).unsqueeze(0).to(device)
                        except Exception as e:
                            print(f"Error reading thumbnail {thumb_path}: {e}")
                
                if thumb_tensor is None:
                    print(f"[SKIP proxy] SVS/Thumb not found for {slide_id}")
                    for s in proxy_skipped:
                        proxy_skipped[s] += 1
                else:
                    try:
                        # Map each tile to its thumbnail cell once (for the
                        # U-Net heatmap lookups). Shared with data generation so the
                        # tile→cell mapping is identical in train and inference.
                        gx_arr, gy_arr = coords_to_grid(coords, (w_slide, h_slide), grid_size)

                        def scores_from_heatmap(heatmap_np):
                            return heatmap_np[gy_arr, gx_arr].astype(np.float64)

                        if any(s in current_strategies for s in ('hema', 'hema_sample')):
                            # Training-free selector: stain color only, no model involved.
                            stain_scores = scores_from_heatmap(
                                stain_map_from_thumb(thumb_tensor, grid_size, channel=0, sigma=8.0))

                        if unet_model is not None and any(s in current_strategies for s in
                                ('unet', 'unet_sample')):
                            with torch.no_grad():
                                raw_umap = unet_model(thumb_tensor)[0, 0]      # pre-sigmoid [H,W]
                                hm = torch.sigmoid(raw_umap).cpu().numpy()
                            unet_scores = scores_from_heatmap(hm)
                            # Log-weights for the weighted sampling. 'sigmoid' is coherent with
                            # a DiceBCE-trained U-Net, whose calibrated output IS the sigmoid.
                            # A ListNet-trained U-Net instead calibrates softmax(logits), so
                            # its log-weight is the logit itself. Selection then lives in the
                            # same space as the symmetric aggregation, which already softmaxes
                            # these logits. Top-k is unaffected either way (both monotone).
                            if score_transform == 'softmax':
                                unet_logw = raw_umap.cpu().numpy()[gy_arr, gx_arr].astype(np.float64)
                            else:
                                unet_logw = np.log(np.clip(unet_scores, 1e-12, None))

                    except Exception as e:
                        print(f"[SKIP proxy] Error computing scores for {slide_id}: {e}")
                        unet_scores = None
                        unet_logw = None
                        for s in proxy_skipped:
                            proxy_skipped[s] += 1

            for budget in budgets:
                k = budget_to_k(budget, N)

                # Random (n_resample trials)
                if 'random' in current_strategies:
                    if 'trials' not in results['random'][budget]:
                        results['random'][budget]['trials'] = {t: {'probs': []} for t in range(n_resample)}
                    
                    for trial in range(n_resample):
                        idx_random = np.random.permutation(len(features))[-k:]
                        prob_random, _ = evaluate_subset(clam_model, features[idx_random], label)
                        results['random'][budget]['trials'][trial]['probs'].append(prob_random)
                    
                    results['random'][budget]['probs'].append(results['random'][budget]['trials'][0]['probs'][-1])
                    results['random'][budget]['labels'].append(label)
                    results['random'][budget]['slide_ids'].append(slide_id)

                # Regular spatial lattice, jittered origin (n_resample trials)
                if 'grid' in current_strategies and coords is not None:
                    if 'trials' not in results['grid'][budget]:
                        results['grid'][budget]['trials'] = {t: {'probs': []} for t in range(n_resample)}
                    for trial in range(n_resample):
                        idx_g = grid_subsample(coords, k)
                        prob_g, _ = evaluate_subset(clam_model, features[idx_g], label)
                        results['grid'][budget]['trials'][trial]['probs'].append(prob_g)
                    results['grid'][budget]['probs'].append(results['grid'][budget]['trials'][0]['probs'][-1])
                    results['grid'][budget]['labels'].append(label)
                    results['grid'][budget]['slide_ids'].append(slide_id)

                # Attention-weighted sampling, no replacement (n_resample trials)
                if 'attn_sample' in current_strategies and A_prob is not None:
                    if 'trials' not in results['attn_sample'][budget]:
                        results['attn_sample'][budget]['trials'] = {t: {'probs': []} for t in range(n_resample)}
                    for trial in range(n_resample):
                        idx_as = weighted_sample_no_replace(A_prob, k)
                        prob_as, _ = evaluate_subset(clam_model, features[idx_as], label)
                        results['attn_sample'][budget]['trials'][trial]['probs'].append(prob_as)
                    results['attn_sample'][budget]['probs'].append(results['attn_sample'][budget]['trials'][0]['probs'][-1])
                    results['attn_sample'][budget]['labels'].append(label)
                    results['attn_sample'][budget]['slide_ids'].append(slide_id)

                # Stain-weighted sampling, no replacement (n_resample trials)
                _ss = f'{_stain}_sample' if _stain else None
                if _ss in current_strategies and stain_scores is not None:
                    if 'trials' not in results[_ss][budget]:
                        results[_ss][budget]['trials'] = {t: {'probs': []} for t in range(n_resample)}
                    # An optical density can be slightly negative on pale background;
                    # shift to a positive scale before using it as a sampling weight.
                    w_st = stain_scores - stain_scores.min() + 1e-6
                    for trial in range(n_resample):
                        idx_ss = weighted_sample_no_replace(w_st, k)
                        prob_ss, _ = evaluate_subset(clam_model, features[idx_ss], label)
                        results[_ss][budget]['trials'][trial]['probs'].append(prob_ss)
                    results[_ss][budget]['probs'].append(
                        results[_ss][budget]['trials'][0]['probs'][-1])
                    results[_ss][budget]['labels'].append(label)
                    results[_ss][budget]['slide_ids'].append(slide_id)

                # U-Net-weighted sampling, no replacement (n_resample trials)
                if 'unet_sample' in current_strategies and unet_scores is not None:
                    if 'trials' not in results['unet_sample'][budget]:
                        results['unet_sample'][budget]['trials'] = {t: {'probs': []} for t in range(n_resample)}
                    for trial in range(n_resample):
                        idx_usm = weighted_sample_from_logw(unet_logw, k)
                        prob_usm, _ = evaluate_subset(clam_model, features[idx_usm], label)
                        results['unet_sample'][budget]['trials'][trial]['probs'].append(prob_usm)
                    results['unet_sample'][budget]['probs'].append(results['unet_sample'][budget]['trials'][0]['probs'][-1])
                    results['unet_sample'][budget]['labels'].append(label)
                    results['unet_sample'][budget]['slide_ids'].append(slide_id)

                # Attention
                # Stain top-k: color alone picks the tiles; CLAM aggregates and classifies
                # exactly as in the random/attention baselines, so the only thing that
                # differs is the selection criterion.
                if _stain is not None and stain_scores is not None:
                    idx_st = np.argsort(stain_scores)[-k:]
                    prob_st, _ = evaluate_subset(clam_model, features[idx_st], label)
                    results[_stain][budget]['probs'].append(prob_st)
                    results[_stain][budget]['labels'].append(label)
                    results[_stain][budget]['slide_ids'].append(slide_id)

                if 'attn' in current_strategies and A_prob is not None:
                    idx_attn = np.argsort(A_prob)[-k:]
                    prob_attn, _ = evaluate_subset(clam_model, features[idx_attn], label)
                    results['attn'][budget]['probs'].append(prob_attn)
                    results['attn'][budget]['labels'].append(label)
                    results['attn'][budget]['slide_ids'].append(slide_id)

                # UNet proxy (solo si scores válidos)
                if 'unet' in current_strategies and unet_scores is not None:
                    idx_unet = np.argsort(unet_scores)[-k:]
                    prob_unet, _ = evaluate_subset(clam_model, features[idx_unet], label)
                    results['unet'][budget]['probs'].append(prob_unet)
                    results['unet'][budget]['labels'].append(label)
                    results['unet'][budget]['slide_ids'].append(slide_id)


        # Save fold results immediately — si el script se cae, no se pierde trabajo
        fold_metrics = []
        for strat in current_strategies:
            for budget in budgets:
                data = results[strat][budget]
                labels_arr = np.array(data['labels'])
                
                # Multiclass cohorts are scored one-vs-rest (class 0 against the rest)
                # so that a single AUROC is comparable across tasks. No-op when binary.
                if binarize_multiclass and len(labels_arr) > 0 and len(data['probs']) > 0:
                    probs_arr_tmp = np.array(data['probs'])
                    if probs_arr_tmp.ndim > 1 and probs_arr_tmp.shape[1] > 2:
                        labels_arr = (labels_arr != 0).astype(int)  # 0->0, others->1
                        
                        bin_probs = np.zeros((probs_arr_tmp.shape[0], 2))
                        bin_probs[:, 0] = probs_arr_tmp[:, 0]
                        bin_probs[:, 1] = 1.0 - probs_arr_tmp[:, 0]
                        data['probs'] = bin_probs.tolist()
                        data['labels'] = labels_arr.tolist()
                        
                        if 'trials' in data:
                            for t in data['trials']:
                                t_arr = np.array(data['trials'][t]['probs'])
                                if len(t_arr) > 0:
                                    t_bin = np.zeros((t_arr.shape[0], 2))
                                    t_bin[:, 0] = t_arr[:, 0]
                                    t_bin[:, 1] = 1.0 - t_arr[:, 0]
                                    data['trials'][t]['probs'] = t_bin.tolist()
                if not data['probs'] or len(np.unique(labels_arr)) < 2:
                    continue

                # Predictions (trial 0 para random) → fold_{fold}.csv
                d = budget_dir(domain_name, strat, budget)
                os.makedirs(d, exist_ok=True)
                probs_arr = np.array(data['probs'])
                is_multiclass = probs_arr.ndim > 1 and probs_arr.shape[1] > 2
                p_save = probs_arr.tolist() if is_multiclass else (probs_arr[:, 1] if probs_arr.ndim > 1 else probs_arr)
                out = {'Y': data['labels'], 'p_1': p_save}
                # slide_id is saved so a bootstrap can resample PATIENTS rather than
                # slides. Where a patient contributes several slides, treating slides as
                # independent narrows the interval spuriously.
                if len(data.get('slide_ids', [])) == len(data['labels']):
                    out['slide_id'] = data['slide_ids']
                pd.DataFrame(out).to_csv(os.path.join(d, f'fold_{fold}.csv'), index=False)

                def get_auc(y, p):
                    if is_multiclass:
                        from sklearn.metrics import roc_auc_score
                        return roc_auc_score(y, p, multi_class='ovr')
                    from sklearn.metrics import roc_auc_score
                    return roc_auc_score(y, p)
                
                def get_auprc(y, p):
                    if is_multiclass:
                        from sklearn.preprocessing import label_binarize
                        from sklearn.metrics import average_precision_score
                        y_bin = label_binarize(y, classes=np.arange(p.shape[1]))
                        return average_precision_score(y_bin, p, average='macro')
                    from sklearn.metrics import average_precision_score
                    return average_precision_score(y, p)

                # Métricas: stochastic strats average AUC over n_resample trials; auc_std over
                # those trials is the RESAMPLING spread (used as the band on a single fold).
                _weighted_strats = ['random', 'attn_sample', 'unet_sample']
                if strat in _weighted_strats and 'trials' in data:
                    trial_aucs, trial_auprcs = [], []
                    for trial in range(n_resample):
                        if trial not in data['trials']:
                            continue
                        t_p = np.array(data['trials'][trial]['probs'])
                        if len(t_p) == 0:
                            continue
                        t_pos = t_p if is_multiclass else (t_p[:, 1] if t_p.shape[1] == 2 else t_p[:, 0])
                        trial_aucs.append(get_auc(labels_arr, t_pos))
                        trial_auprcs.append(get_auprc(labels_arr, t_pos))
                    if trial_aucs:
                        fold_metrics.append({'strategy': strat, bcol: budget,
                                             'auc': np.mean(trial_aucs), 'auprc': np.mean(trial_auprcs),
                                             'auc_std': float(np.std(trial_aucs))})
                else:
                    p_pos = probs_arr if is_multiclass else (probs_arr[:, 1] if probs_arr.ndim > 1 and probs_arr.shape[1] == 2 else probs_arr)
                    fold_metrics.append({'strategy': strat, bcol: budget,
                                         'auc': get_auc(labels_arr, p_pos),
                                         'auprc': get_auprc(labels_arr, p_pos), 'auc_std': 0.0})

        new_metrics_df = pd.DataFrame(fold_metrics)
        if os.path.exists(fold_metrics_path):
            saved_df = pd.read_csv(fold_metrics_path)
            combined_df = pd.concat([saved_df, new_metrics_df]).drop_duplicates(subset=['strategy', bcol], keep='last')
            combined_df.to_csv(fold_metrics_path, index=False)
        else:
            new_metrics_df.to_csv(fold_metrics_path, index=False)
        n_slides = len(dataset)
        skip_summary = ', '.join(f"{s}: {c}" for s, c in proxy_skipped.items() if c > 0)
        print(f"[Fold {fold}] Results saved. {n_slides} slides total. Skipped (slide-budget pairs): {skip_summary or 'none'}")

        update_and_plot(current_fold=fold)

    update_and_plot(current_fold=None)

if __name__ == "__main__":
    main()
