"""Resolution of external data/output locations.

Nothing produced by this library is written inside the repository. All datasets,
model checkpoints, generated tile data, evaluation outputs and plots live under a
single external root, taken from ``paths.data_root`` in the YAML config, or from the
``TILESELECT_DATA_ROOT`` environment variable, or ``~/tileselect-data`` as a last resort.
"""
import os

DEFAULT_DATA_ROOT = os.environ.get("TILESELECT_DATA_ROOT",
                                   os.path.expanduser("~/tileselect-data"))


def data_root(cfg):
    """External root for all data and outputs (never inside the repo)."""
    return cfg.get("paths", {}).get("data_root", DEFAULT_DATA_ROOT)


def results_dir(cfg):
    """Trained models / checkpoints (CLAM, U-Net)."""
    return os.path.join(data_root(cfg), "results")


def eval_results_dir(cfg):
    """Per-strategy degradation evaluation outputs."""
    return os.path.join(data_root(cfg), "eval_results")


def plots_dir(cfg):
    """Figures (training curves, degradation curves, comparisons)."""
    return os.path.join(data_root(cfg), "plots")


def data_dir(cfg):
    """Feature H5 files, generated U-Net data, label CSVs."""
    return os.path.join(data_root(cfg), "data")
