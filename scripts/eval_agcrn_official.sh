#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." 
python3 experiments/eval_official_adaptive_baselines.py --models agcrn "$@"
