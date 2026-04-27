#!/usr/bin/env bash
# Polish v11 — PEMS04 source training for 5 missing models × 3 seeds.
# Models already on PEMS04 (in baselines/stgnn/*__pems04__seed[0-2].pt):
#   DCRNN, GRU, GWNet, GWNet_strong.
# Missing: STGCN, MTGNN, AGCRN, MegaCRN, STAEformer → train with seeds 0,1,2.
# Output dirs:
#   - STGCN/MTGNN → baselines/stgnn_pems04_extra/
#   - AGCRN/MegaCRN/STAEformer → baselines/stgnn/
# Prereqs:
#   - experiments/official_adaptive_common.py now handles pems04 via npz loader.
#   - third_party/official_staeformer/model/STAEformer.yaml has a PEMS04 entry.
#   - data/pems04/pems04.npz and data/pems04/adj_mx.pkl must exist.
set -euo pipefail
cd "$(dirname "$0")/.." 
source /opt/conda/bin/activate && conda activate pytorch
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

mkdir -p logs/pems04_source_polish
mkdir -p baselines/stgnn_pems04_extra

START=$(date -Iseconds)
echo "[START] $START"

# ── STGCN & MTGNN on PEMS04 (seeds 0,1,2) ────────────────────────────────────
for BB in stgcn mtgnn; do
  echo "[PEMS04] $BB seeds=0,1,2 starting at $(date -Iseconds)"
  python experiments/stgnn_backbone_bank.py \
    --backbones "$BB" --seeds 0 1 2 \
    --dataset pems04 \
    --output_dir baselines/stgnn_pems04_extra/ \
    --registry baselines/stgnn_protocol_registry.json \
    2>&1 | tee -a "logs/pems04_source_polish/${BB}_pems04.log"
done

# ── AGCRN / STAEformer / MegaCRN on PEMS04 (seeds 0,1,2) ─────────────────────
for MODEL in agcrn staeformer megacrn; do
  for SEED in 0 1 2; do
    echo "[PEMS04] $MODEL seed=$SEED starting at $(date -Iseconds)"
    python experiments/train_official_adaptive_baseline.py \
      --model "$MODEL" --dataset pems04 --seed "$SEED" \
      --output_dir baselines/stgnn \
      --registry baselines/stgnn_protocol_registry.json \
      2>&1 | tee -a "logs/pems04_source_polish/${MODEL}_pems04_seed${SEED}.log"
  done
done

echo "[DONE] finished at $(date -Iseconds); started $START"
