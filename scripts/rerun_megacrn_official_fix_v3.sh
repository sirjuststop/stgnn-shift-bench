#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." 

python3 scripts/build_official_adaptive_registry.py

seeds=("$@")
if [ ${#seeds[@]} -eq 0 ]; then
  seeds=(0 1 2)
fi

for seed in "${seeds[@]}"; do
  log_path="results/official_adaptive_relation/logs/train_megacrn_seed${seed}_fix_v3.log"
  printf '[start %s] model=megacrn seed=%s\n' "$(date -u +%FT%TZ)" "$seed" | tee "$log_path"
  set +e
  python3 experiments/train_official_adaptive_baseline.py \
    --model megacrn \
    --dataset metr-la \
    --seed "$seed" \
    --device cuda \
    --output_dir baselines/stgnn_official_fix \
    --registry results/official_adaptive_relation/official_adaptive_unified_registry.json \
    2>&1 | tee -a "$log_path"
  status=${PIPESTATUS[0]}
  set -e
  printf '[end %s] model=megacrn seed=%s exit=%s\n' "$(date -u +%FT%TZ)" "$seed" "$status" | tee -a "$log_path"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done
