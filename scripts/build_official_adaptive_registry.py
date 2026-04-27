#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "official_adaptive_relation" / "official_adaptive_unified_registry.json"
SOURCE_SPECS = [
    (
        ROOT / "baselines" / "stgnn_protocol_registry.json",
        "agcrn__metr-la__",
    ),
    (
        ROOT / "results" / "official_adaptive_relation" / "official_fix_registry_staeformer.json",
        "staeformer__metr-la__",
    ),
    (
        ROOT / "results" / "official_adaptive_relation" / "official_fix_registry_megacrn.json",
        "megacrn__metr-la__",
    ),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Overwrite existing unified-registry entries when the same key is present in a source registry.",
    )
    return ap.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    output_path = Path(args.output)
    existing = load_json(output_path)
    merged = dict(existing)
    added = []
    refreshed = []

    for source_path, prefix in SOURCE_SPECS:
        source = load_json(source_path)
        for key in sorted(k for k in source if k.startswith(prefix)):
            if key in merged and not args.refresh_existing:
                continue
            merged[key] = source[key]
            if key in existing:
                refreshed.append(key)
            else:
                added.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    summary = {
        "output": str(output_path.relative_to(ROOT)),
        "num_entries": len(merged),
        "added": added,
        "refreshed": refreshed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
