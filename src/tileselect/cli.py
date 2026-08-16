"""Command-line entry point for the tile-selection pipeline.

Installed as the ``tileselect`` console script. Orchestrates the data and training
stages; the degradation evaluation has its own entry point
(``tileselect-eval`` / ``python -m tileselect.eval.degradation``).

Modules are imported lazily so a single stage does not pull in the whole stack.
"""
import argparse
import os
import sys

import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _discover_config(explicit):
    if explicit is not None:
        if not os.path.exists(explicit):
            sys.exit(f"Error: config file not found at {explicit}")
        return explicit
    config_dir = "configs"
    if not os.path.isdir(config_dir):
        sys.exit("Error: 'configs' directory not found; pass --config.")
    files = [f for f in os.listdir(config_dir) if f.endswith((".yaml", ".yml"))]
    if len(files) == 1:
        path = os.path.join(config_dir, files[0])
        print(f"Auto-detected config file: {path}")
        return path
    sys.exit(f"Error: {len(files)} configs in {config_dir}/; specify one with --config.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tileselect",
        description="Thumbnail-based tile selection for whole-slide image classification")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config")
    parser.add_argument("--create_splits", action="store_true", help="Generate train/val/test splits")
    parser.add_argument("--extract_thumbs", action="store_true", help="Extract thumbnail features (CONCH)")
    parser.add_argument("--generate_unet_data", action="store_true", help="Generate U-Net GT (CLAM attention heatmaps)")
    parser.add_argument("--train_unet", action="store_true", help="Train the U-Net tile selector")
    parser.add_argument("--train_clam", action="store_true", help="Train the CLAM classifier")
    parser.add_argument("--domain", type=str, default=None,
                        help="Cohort tag to select from the label CSV's optional domain column")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index (0-9) for --generate_unet_data / --train_unet; "
                             "omit to run all folds (when use_folds/use_cv).")
    args = parser.parse_args(argv)

    cfg = load_config(_discover_config(args.config))

    def folds():
        use_cv = cfg.get("experiment", {}).get("use_folds") or cfg.get("experiment", {}).get("use_cv")
        if args.fold is not None:
            return [args.fold]
        return range(10) if use_cv else [None]

    if args.create_splits:
        print("\n=== STAGE: Create Splits ===")
        from tileselect.data.splits import create_splits
        create_splits(cfg, args.domain)

    if args.extract_thumbs:
        print("\n=== STAGE: Extract Thumbs (CONCH) ===")
        from tileselect.data.extract_thumbs import run_extract_thumbs
        run_extract_thumbs(cfg)

    if args.generate_unet_data:
        from tileselect.data.generate_unet_data import run_generate_unet_data
        for fold in folds():
            print(f"\n=== STAGE: Generate U-Net Data ({'original' if fold is None else f'Fold {fold}'}) ===")
            run_generate_unet_data(cfg, fold=fold)

    if args.train_unet:
        from tileselect.train.train_unet import run_train_unet
        for fold in folds():
            print(f"\n=== STAGE: Train U-Net ({'original' if fold is None else f'Fold {fold}'}) ===")
            run_train_unet(cfg, fold=fold)

    if args.train_clam:
        print("\n=== STAGE: Train CLAM ===")
        from tileselect.train.train_clam import run_train_clam
        run_train_clam(cfg)

    if not any([args.create_splits, args.extract_thumbs, args.generate_unet_data,
                args.train_unet, args.train_clam]):
        print("No stage selected. Use --help to see available options.")
        parser.print_help()


if __name__ == "__main__":
    main()
