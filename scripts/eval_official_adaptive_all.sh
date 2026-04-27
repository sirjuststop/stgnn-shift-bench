#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." 

python3 scripts/build_official_adaptive_registry.py
python3 experiments/eval_official_adaptive_baselines.py \
  --models agcrn megacrn staeformer \
  --registry results/official_adaptive_relation/official_adaptive_unified_registry.json \
  --source_dataset metr-la \
  --target_dataset pems-bay \
  --output_dir results/official_adaptive_relation \
  "$@"
