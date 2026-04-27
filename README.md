# How Adaptive Traffic Models Redistribute Fragility
## A Multi-Shift Diagnostic of Spatiotemporal Architectures

*Submitted to NeurIPS 2026 Evaluations & Datasets Track*

---

## Overview

We conduct a controlled diagnostic of eight spatiotemporal graph neural network (STGNN) architectures under three deployment-realistic distribution shifts: **city transfer**, **sensor dropout**, and **temporal span**. All evaluations use frozen checkpoints trained once on METR-LA — no target-domain retraining, no cherry-picking.

Key findings:
- Adaptive models (AGCRN, MegaCRN, STAEformer) show lower city-transfer error than graph-free baselines but amplify sensor-dropout fragility by 2–4×.
- A pre-registered routing-density hypothesis is confirmed across both METR-LA (ρ = +0.979) and PEMS04 (ρ = +0.910): denser learned adjacency correlates with higher sensor-shift sensitivity.
- Four of five pre-registered hypotheses were refuted; the failures are documented explicitly as part of the evaluation contribution.

**Paper**: [`paper/main.pdf`](paper/main.pdf)

---

## Repository Structure

```
configs/                     # frozen training hyperparameters
  stgnn_unified.yaml
experiments/                 # evaluation + training scripts
  stgnn_backbone_bank.py     # all 8 backbone model definitions
  stgnn_shift_eval_v2.py     # main shift evaluation (city / sensor / temporal)
  pems04_routing_gate_native.py
  mask_aware_dropout_sweep.py
  cross_city_no_pad_sensitivity.py
  ...
baselines/
  stgnn_protocol_registry.json   # checkpoint manifest + test metrics
  stgnn/                         # METR-LA frozen checkpoints (3 seeds each)
  stgnn_bay207/                  # PEMS-BAY source-polish checkpoints
  stgnn_official_fix/            # AGCRN / MegaCRN / STAEformer checkpoints
  stgnn_pems04_extra/            # PEMS04 extra seeds
results/
  official_adaptive_relation/
    UPDATED_MERGED_COMPARISON_TABLE_2026-04-16.csv   # Table 1 source
    UPDATED_MERGED_COMPARISON_TABLE_2026-04-16.json
  polish_v15/
    pems04_native_routing_gate.json
    mask_aware_ratio_sweep.json
    cross_city_no_pad_sensitivity.json
paper/
  main.pdf      # compiled paper (double-blind)
  main.tex      # LaTeX source
  sections/     # per-section .tex files
  figures/      # all figures + generation scripts
```

---

## Reproducing Results

### 1. Environment setup

```bash
conda create -n stgnn python=3.10
conda activate stgnn
pip install -r requirements.txt
```

### 2. Data

Download and place datasets under `data/`:

| Dataset | Source | Path |
|---------|--------|------|
| METR-LA | [DCRNN repo](https://github.com/liyaguang/DCRNN) | `data/metr-la/` |
| PEMS-BAY | [DCRNN repo](https://github.com/liyaguang/DCRNN) | `data/pems-bay/` |
| PEMS04 | [ASTGNN repo](https://github.com/guoshnBJTU/ASTGNN) | `data/pems04/` |

Expected files: `metr-la.h5`, `adj_mx.pkl` for METR-LA; `pems-bay.h5`, `adj_mx_bay.pkl` for PEMS-BAY.

### 3. Main shift evaluation (Table 1)

```bash
# All 8 backbones × 3 shifts — reads frozen checkpoints from baselines/
python experiments/stgnn_shift_eval_v2.py \
    --registry baselines/stgnn_protocol_registry.json \
    --output_dir results/my_run/
```

### 4. Routing-density analysis (§5.2)

```bash
python experiments/routing_sparsity_spectrum.py \
    --registry baselines/stgnn_protocol_registry.json \
    --output results/my_run/routing_density.json
```

### 5. PEMS04 native replication (Appendix)

```bash
python experiments/pems04_routing_gate_native.py \
    --registry baselines/stgnn_protocol_registry.json \
    --output results/my_run/pems04_routing.json
```

### 6. Mask-aware dropout sweep (Appendix)

```bash
python experiments/mask_aware_dropout_sweep.py \
    --registry baselines/stgnn_protocol_registry.json \
    --output results/my_run/mask_sweep.json
```

---

## Checkpoint Manifest

`baselines/stgnn_protocol_registry.json` lists every frozen checkpoint with its backbone, dataset, seed, and in-distribution test metrics. All evaluation scripts load checkpoints via this registry.

Training hyperparameters are frozen in `configs/stgnn_unified.yaml` and were held constant across all eight backbones and all seeds.

---

## Figures

All figures are reproducible from the canonical results CSV. Generation scripts are in `paper/figures/`:

```bash
python paper/figures/gen_shift_profiles_v15.py \
    --table results/official_adaptive_relation/UPDATED_MERGED_COMPARISON_TABLE_2026-04-16.csv \
    --out paper/figures/shift_profiles.pdf
```

---

## License

Code released under MIT License. Model checkpoints are derived from publicly available STGNN implementations; see individual backbone attributions in `experiments/stgnn_backbone_bank.py`.
