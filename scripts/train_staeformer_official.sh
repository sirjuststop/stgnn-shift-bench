#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." 
python3 experiments/train_official_adaptive_baseline.py --model staeformer "$@"
