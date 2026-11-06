# Supplementary experiments: Tables 10, 11, 21 and 31

These are reconstructed implementations of the supplementary evaluation protocols,
released on 2026-11-07. They compute predictions and metrics from data and model
weights; they do not read the paper's aggregate tables as experimental outputs.
The checked-in configuration is an explicit **regeneration profile**, not a claim
that every parameter matches the original run. Generated results must be compared
with the paper before being described as a numerical reproduction.

## Install and prepare inputs

Run commands from the repository root with Python 3.10 or newer:

```sh
python -m pip install -r requirements-supplementary.txt
python -m experiments.supplementary --help
```

Obtain the original datasets from [DCRNN](https://github.com/liyaguang/DCRNN)
(METR-LA and PEMS-BAY) and [ASTGNN](https://github.com/guoshnBJTU/ASTGNN)
(PEMS04), subject to their distribution terms. Expected paths:

| Dataset | Traffic matrix | Adjacency |
|---|---|---|
| METR-LA | `data/metr-la/METR-LA.csv` or `metr_la.h5` | `data/metr-la/adj_mx.pkl` |
| PEMS-BAY | `data/pems-bay/PEMS-BAY.csv` or `pems_bay.h5` | `data/pems-bay/adj_mx_bay.pkl` or `adj_mx.pkl` |
| PEMS04 | `data/pems04/pems04.npz`, `data` array, channel 0 (flow) | `data/pems04/adj_mx.pkl` |

CSV files have a timestamp index followed by sensor columns. Adjacencies must use
the same sensor ordering. A plain `adj_mx.npy` is also accepted. Only load trusted
pickle adjacency files and checkpoint files. Data may be placed elsewhere with
`--data-root /path/to/data`.

The checkpoint map is in
[`configs/supplementary_experiments.json`](../../configs/supplementary_experiments.json).
It reuses the published bank where available. GTS/DGCRN native-source weights and
full-network PEMS-BAY GWNet/MTGNN weights have separate paths under
`baselines/supplementary/`; they are created by the training command. The existing
207-node PEMS-BAY transfer checkpoints are not substitutes for 325-node native-source
checkpoints.

```sh
# Reports missing data/checkpoints and exits nonzero if any are missing.
python -m experiments.supplementary check

# Train only missing checkpoints; existing frozen weights are never overwritten.
python -m experiments.supplementary train-missing --device cuda

# Or train one model/source/seed into a new checkpoint file.
python -m experiments.supplementary train --model gts --dataset metr-la --seed 0 --device cuda
```

All six adaptive models have executable training and inference adapters. Training
selects weights using the validation split, never test error. GTS and DGCRN model
implementations are vendored at pinned upstream commits with licenses; see
[`third_party/SUPPLEMENTARY_SOURCES.md`](../../third_party/SUPPLEMENTARY_SOURCES.md).
GTS checkpoint files include the training-only graph features and materialize all
lazy recurrent parameters before optimization or loading. No extra upstream clone
or TensorFlow installation is required.

## Run the four experiments

```sh
python experiments/table10_severity.py --device cuda --output results/regenerated/table10
python experiments/table11_corruptions.py --device cuda --output results/regenerated/table11
python experiments/table21_fewshot.py --device cuda --output results/regenerated/table21
python experiments/table31_routing.py --device cuda --output results/regenerated/table31
```

Use `--device cpu` without a GPU. The equivalent module interface is
`python -m experiments.supplementary table10` (or `table11`, `table21`, `table31`).
Each command accepts `--config`, `--data-root` and `--output`. A diagnostic
`--max-windows` cap is recorded in results; omit it for full evaluation. Existing
`results.json` files are not overwritten. A failed run can leave `partial.json`
marked `complete: false`; it does not count as a completed experiment.

| Table | Coverage | Main output |
|---|---|---|
| 10 | DCRNN, MTGNN-Light, GWNet, GTS, DGCRN; 10/20/30/40/50% affected sensors | Healthy-sensor masked mph MAE increase; failed/whole-network metrics also saved |
| 11 | DCRNN, GRU, STGCN, MTGNN-Light, GWNet, GWNet[Strong]; five corruptions at all five severities | Three sensor splits; select `ratio=0.3` and `masked_physical_mae` for the paper's table format |
| 21 | DCRNN, MTGNN-Light, GWNet, GRU; k=0/50/200/800; 30 optimizer steps for k>0 | Target-test masked mph MAE, including the unadapted k=0 checkpoint |
| 31 | AGCRN, MegaCRN, MTGNN-Light, GWNet, GTS, DGCRN on each of METR-LA/PEMS-BAY/PEMS04 | Six-architecture Pearson, two-sided t test, all 720 exact permutations, leave-one-out and Spearman |

`results.json` contains per-training-seed/per-mask measurements, failed-sensor
indices, configuration, input/checkpoint hashes, and completion status.
`summary.csv` aggregates masks within a training seed and then training seeds,
with population standard deviations. Table 31 additionally writes
`architecture_measurements.csv`; its correlations use six architecture means,
not individual masks as independent observations. Every delta uses clean and
corrupted predictions on the same windows, targets and sensor split.

## Protocol choices

The paper specifies the architectures, severity fractions, target-data budgets,
30 adaptation steps and native-source comparisons. The profile also makes the
remaining choices explicit so a new run is fully specified:

- Chronological 70/10/20 splits, 12-step history and horizon, training seeds
  0/1/2, mask seeds 42--46, batch size 64. Scaling is fitted only on training data;
  saved checkpoint scalers are used where present. Windows do not cross splits.
- Failure membership comes only from sampled sensor indices. Genuine zero readings
  on healthy sensors remain healthy. Normalized MAE includes every target;
  masked physical-unit MAE includes targets greater than zero. PEMS04 reports flow,
  not mph.
- The default zero fill is **zero in normalized input space**, matching the
  existing bank evaluation scripts. Physical zero fill is a separate configurable
  choice: set `protocol.corruption.fill_space` to `physical`. This changes the
  intervention and must not be mixed into one correlation or table.
- Stale hold repeats the first observed value within each input window. Noise is
  independent Gaussian noise with standard deviation 0.1 in normalized units.
  Bias adds +1 normalized unit. Intermittent failure zero-fills each selected
  sensor/time independently with probability 0.5. These amplitudes and the hold
  convention are regeneration choices, configurable in the JSON profile.
  Traffic values alone are corrupted; calendar covariates remain intact.
- Table 21 uses one sorted target subset (seed 42), target-training statistics for
  both normalization and denormalization, and target-subset adjacency by default
  (`table21.adjacency: target`; `source` is also supported). For each source
  checkpoint and k, weights and the Adam optimizer are reset. Nested target-training
  windows are selected with seed 42; 30 Huber-loss updates use learning rate 0.001
  and gradient clipping at 5. No target-test windows enter adaptation. The subset
  indices, training-window indices and losses are saved.
- The original four-model routing proxies retain the repository's definitions:
  GWNet softmax(ReLU(E1 E2^T)), AGCRN softmax(ReLU(E E^T)), MTGNN sparse cosine
  top-k, and the MegaCRN We1/We2 memory proxy. These are diagnostic proxies, which
  need not equal every propagation operator used by the model. GTS uses
  row-normalized link probabilities with diagonal removed; DGCRN averages row
  maxima of its normalized forward/reverse dynamic graphs across clean encoder
  and decoder steps, windows and nodes. The latter two aggregation choices are
  explicit definitions for this reconstruction. DGCRN routing is observed during
  clean inference, without modifying graph computation.
- GTS graph samples and DGCRN initialization randomness are paired between clean
  and corrupted forward passes. Decoder inference uses calendar covariates but no
  target traffic labels. Changing batch size can change model stochastic draws;
  random corruption fields themselves are independent of batching.
- New native checkpoints use the profile's Adam/Huber training setup (100 epochs,
  patience 15, learning rate 0.001). GTS graph features use the first 2016 training
  timestamps, with a 10-neighbour feature-similarity graph and a 0.1 BCE graph-loss
  weight. These are new-training settings, not recovered historical hyperparameters.

## Verify the implementation

```sh
python -m unittest discover -s tests -p test_supplementary.py -v
```

The tests check failure-mask semantics, target masking, both zero-fill definitions,
batch-weighted metrics, train-only normalization, split boundaries, all ten model
adapters with batch size one, 30-step adaptation and exact permutation statistics.
An end-to-end test trains tiny synthetic checkpoints and runs all four commands,
including all six routing models on all three dataset layouts. Synthetic outputs
are temporary test fixtures; they are not paper results or numerical reproductions.
