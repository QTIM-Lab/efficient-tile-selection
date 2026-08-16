# Thumbnail-Based Tile Selection for Efficient Whole-Slide Image Classification

Code for the COMPAYL @ MICCAI 2026 paper *Thumbnail-Based Tile Selection for
Efficient Whole-Slide Image Classification in Resource-Limited Pathology*.

Attention-MIL slide classification is expensive because a foundation encoder has to
run on **every** tile before the attention layer can decide which tiles mattered. The
question here is how far the tile budget can be cut if a cheap selector picks the
tiles *first* — cheap enough that it reads only the slide thumbnail, never a tile.

The selector is a small U-Net trained to predict a frozen CLAM model's attention map
from the thumbnail alone. At inference it ranks tiles from a 224x224 heatmap, and only
the top-`k` tiles are ever encoded.

## What is in here

| Stage | Entry point | What it does |
|---|---|---|
| Splits | `tileselect --create_splits` | Patient-level train/val/test splits |
| Thumbnails | `tileselect --extract_thumbs` | Thumbnail extraction (+ CONCH features) |
| U-Net targets | `tileselect --generate_unet_data` | Renders frozen-CLAM attention onto the thumbnail grid |
| Train selector | `tileselect --train_unet` | Trains the thumbnail U-Net against those targets |
| Train classifier | `tileselect --train_clam` | Optional — see "Inherited checkpoints" below |
| Evaluation | `tileselect-eval` | The tile-budget degradation sweep |
| Figures | `tileselect-plot` | Re-plots curves from saved CSVs, no GPU needed |

## Install

Python >= 3.11. Dependencies are pinned in `uv.lock`.

```bash
uv sync                     # creates .venv with the locked versions
source .venv/bin/activate
```

or, without uv: `pip install -e .` (resolves fresh, does not honour the lock).

OpenSlide is pulled in through `openslide-bin`, so no system package is required.

## Data you need to supply

Nothing is written inside the repository. Every output goes under `paths.data_root`
(set it in the config, or export `TILESELECT_DATA_ROOT`).

The pipeline expects, per cohort:

- **Tile features** — one `.h5` per slide containing `features` `[N, D]` and `coords`
  `[N, 2]` (level-0 pixel coordinates, with a `patch_size_level0` attribute). `D` is
  768 for CONCH v1.5, 512 for CONCH v1.
- **A label CSV** — columns `slide_id` and `label`; an optional `features_path`
  column overrides the feature directory per slide.
- **Whole-slide images** — read only to get thumbnails (and to profile per-tile I/O).
- **CLAM checkpoints** — one per fold, `s_{fold}_checkpoint.pt`, alongside the
  matching `splits_{fold}.csv`.

`configs/template.yaml` documents every key the tool reads. Copy it and fill in your
own paths — the repository deliberately ships no data, no label files and no site
paths.

## Bring your own classifier

Nothing here trains a classifier for you. The tool consumes an existing CLAM
checkpoint and never updates it — that is the point, since the selector is distilled
from whatever attention that classifier already learned.

If you do retrain with `--train_clam`, the `model:` block has to mirror how your
checkpoint was originally trained, or the retrain will quietly produce a different
model. `inst_loss` is the usual trap: some CLAM training logs record `None`, which
CLAM resolves to `nn.CrossEntropyLoss` rather than the vendored SmoothTop1SVM. And
`bag_weight < 1` means the instance-level clustering loss also shaped
`attention_net` — which matters here, because `attention_net` is exactly what the
selector imitates, even though the instance head is never evaluated.

## Reproducing the experiments

```bash
cp configs/template.yaml configs/mycohort.yaml   # then fill in the paths
CFG=configs/mycohort.yaml

# 1. Splits, thumbnails, and the selector's targets (frozen CLAM attention)
tileselect --config $CFG --create_splits --domain mycohort
tileselect --config $CFG --extract_thumbs
tileselect --config $CFG --generate_unet_data     # all folds; --fold N for one

# 2. Train the thumbnail selector
tileselect --config $CFG --train_unet

# 3. Tile-budget degradation sweep
tileselect-eval --config $CFG --domain mycohort --proxy all --split test
```

`--domain` selects one cohort from the `domain` column of the label CSV; drop it if
the CSV holds a single cohort.

Results land in `<data_root>/eval_results/<run>/` as `fold_{i}_metrics.csv` (one row
per strategy and budget) plus the per-slide predictions each row was computed from.
`tileselect-plot --results_dir ... --domain mycohort` redraws the curves from those
CSVs alone, with no GPU and no torch.

### Selection strategies

`--proxy all` evaluates, at every budget:

| Strategy | Needs a model? | What it ranks by |
|---|---|---|
| `random` | no | uniform draw — the floor |
| `grid` | no | a regular spatial lattice, origin jittered per draw |
| `hema` | no | per-slide Macenko haematoxylin map from the thumbnail |
| `attn` | yes, all tiles | CLAM's own attention — needs every tile encoded |
| `unet` | yes, thumbnail only | the predicted attention heatmap |
| `attn_sample` / `unet_sample` | as above | weighted sampling without replacement instead of top-`k` |

`grid` and `hema` are the training-free controls: they separate "does spatial coverage
explain it?" and "does stain intensity explain it?" from "does the *ranking* matter?".
`attn` is not a fair baseline for cost — it is the thing being avoided, since computing
attention already requires encoding the whole bag.

### Budget axis

`--budget_mode pct` (default) sweeps fractions of each slide's tiles; `--budget_mode k`
sweeps absolute tile counts, which is the fairer axis when slides differ in size.
Override the grid with `--budgets 4096,1024,256,64,16,8,4`.

Stochastic strategies are repeated `eval.n_resample` times (10 by default) so their
curves carry a spread; deterministic top-`k` strategies are drawn without a band
because repeating them changes nothing.

## Layout

```
src/tileselect/
  paths.py            resolves every output location from data_root
  cli.py              the `tileselect` pipeline entry point
  models/             clam.py (CLAM_SB), unet.py (SimpleUNet)
  data/               splits, thumbnails, U-Net target generation, dataset
  train/              train_unet.py, train_clam.py
  eval/               degradation.py (the sweep), plot.py (figures)
  utils/              tile<->grid mapping, CONCH loader, TIFF warning suppression
  vendored/topk/      SmoothTop1SVM, vendored; only reached when inst_loss: svm
configs/              template.yaml -- every key, documented
scripts/slurm/        a generic SLURM template
```

`utils/tiles.py` holds the single tile-to-thumbnail-cell mapping used both when
writing the U-Net's targets and when reading its predictions back, so there is no
silent train/inference mismatch.

## Citation

```bibtex
@inproceedings{pulidoarias2026thumbnail,
  title     = {Thumbnail-Based Tile Selection for Efficient Whole-Slide Image
               Classification in Resource-Limited Pathology},
  author    = {Pulido Arias, Dagoberto and Cleveland, Mason and
               Mindroc-Filimon, Diana and Kim, Albert and Bridge, Christopher P.},
  booktitle = {COMPAYL, MICCAI Workshops},
  year      = {2026}
}
```

## License

See `LICENSE`.
