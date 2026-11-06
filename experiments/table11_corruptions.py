"""Table11 executable entry point; run with --help for options."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.supplementary.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["table11", *sys.argv[1:]]))
