"""Shared tile ↔ thumbnail-grid mapping.

Single source of truth for projecting level-0 tile coordinates onto the cell of a
``size × size`` thumbnail heatmap. Used by BOTH the U-Net data generation (which
writes the attention targets onto that grid) and the degradation eval (which reads
the predicted heatmap back), so the two use the *exact* same mapping — no silent
train/test gap.
"""
import numpy as np


def coords_to_grid(coords, slide_dim, size):
    """Map each tile coordinate to its ``size × size`` grid cell.

    Args:
        coords: (N, 2) array of (x, y) level-0 coordinates.
        slide_dim: (w, h) slide dimensions at level 0.
        size: grid/thumbnail side length.

    Returns:
        (gx, gy): int arrays of shape (N,), each clamped to [0, size-1].
    """
    w, h = slide_dim
    gx = np.minimum(size - 1, (coords[:, 0] / w * size).astype(int))
    gy = np.minimum(size - 1, (coords[:, 1] / h * size).astype(int))
    return gx, gy
