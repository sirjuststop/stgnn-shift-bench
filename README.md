# Traffic STGNNs across Sensor, City, and Time Shifts

## Routing Concentration Tracks Sensitivity to Sensor Dropout

**Xile Wang and Mingqi Yang**
South China University of Technology

NeurIPS 2026 Evaluations & Datasets camera-ready artifact.

Revision v5 (2026-11-04) corrects protocol comparisons, transfer rankings, and statistical reporting, and simplifies the manuscript and accompanying documentation.

[Paper PDF](paper/main.pdf) | [Paper source and build instructions](paper/README.md) | [Revision notes](paper/REVISION_NOTES.md)

We evaluate eight frozen traffic forecasting architectures under sensor dropout, cross-city transfer, and weekday/weekend subgroup differences. Sensor-dropout sensitivity tracks routing concentration, while cross-city and temporal rankings depend on the shift and reporting convention. The expanded routing comparison includes six graph-based adaptive architectures. A mask-aware GWNet intervention recovers about 75% of its multi-mask dropout gap without retraining.

The final paper distinguishes normalized/plain temporal error from masked physical-unit error and includes the supplementary routing, corruption, and transfer analyses.

## Rebuild the paper and revised assets

The source and compiled PDF are in `paper/`. A LaTeX distribution can compile the manuscript without datasets or checkpoints:

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

From the repository root, regenerate Figure 1 and the revised tables:

```sh
python -m pip install numpy matplotlib
python paper/figures/build_camera_ready_assets.py --from-packaged-inputs
```

The numerical inputs are included in `paper/inputs/asset_sources.json`. This command rebuilds the presentation of saved results, without running model evaluation.

## Experiment environment and data

```sh
conda create -n stgnn python=3.10
conda activate stgnn
pip install -r requirements.txt
```

Download the original datasets under their distribution terms:

| Dataset | Source | Local directory |
|---------|--------|-----------------|
| METR-LA | [DCRNN](https://github.com/liyaguang/DCRNN) | `data/metr-la/` |
| PEMS-BAY | [DCRNN](https://github.com/liyaguang/DCRNN) | `data/pems-bay/` |
| PEMS04 | [ASTGNN](https://github.com/guoshnBJTU/ASTGNN) | `data/pems04/` |

The loaders accept `METR-LA.csv` / `PEMS-BAY.csv`, or HDF5 files as described in `experiments/stgnn_backbone_bank.py`. Adjacency files are `data/metr-la/adj_mx.pkl` and `data/pems-bay/adj_mx_bay.pkl`. Raw datasets are not redistributed here.

`baselines/stgnn_protocol_registry.json` identifies saved checkpoints and their configurations. The original frozen checkpoint files remain in `baselines/`. Canonical-bank evaluation and the three official adaptive models have separate entry points:

```sh
python experiments/stgnn_shift_eval_v2.py --help
python experiments/eval_official_adaptive_baselines.py --help
```

For example, to evaluate the canonical bank with source-only structural access and write into a new directory:

```sh
python experiments/stgnn_shift_eval_v2.py \
  --registry baselines/stgnn_protocol_registry.json \
  --backbones dcrnn gwnet gru stgcn mtgnn \
  --source_only_adj_control \
  --output_dir results/my_run/canonical

python experiments/eval_official_adaptive_baselines.py \
  --models agcrn megacrn staeformer \
  --registry baselines/stgnn_protocol_registry.json \
  --output_dir results/my_run/adaptive
```

Model definitions used by the official adapters are included in `third_party/` and `external_repos/AGCRN/`. The AGCRN files are pinned to the commit recorded by the existing model adapter; upstream attribution and license are preserved.

## Recommendation artifact

Run from the repository root:

```python
from stgnn_stress import recommend_architecture

rec = recommend_architecture(
    sensor_churn=True, city_transfer=False, weekend_heavy=False
)
print(rec.architecture, rec.runner_up, rec.margin)
```

The frozen recommendation table is `stgnn_stress/data/decision_tree.json`. Its temporal axis uses the original normalized error profile, as explained in the paper; it does not rank masked physical-unit weekend errors. The package's shift harness is a scaffold; use the experiment entry points above for evaluation.

## Result and source map

| Location | Contents |
|----------|----------|
| `configs/stgnn_unified.yaml` | Canonical-bank training configuration |
| `experiments/` | Training and evaluation entry points |
| `baselines/` | Frozen checkpoints and registry |
| `results/official_adaptive_relation/` | Saved main comparison and adaptive evaluation results |
| `results/polish_v15/` | Native PEMS04 routing, mask-aware sweep, and city-transfer checks |
| `paper/generated/metrics.json` | Reaggregated camera-ready statistics and source hashes |
| `paper/generated/rebuttal_reported.json` | Separately identified author-reported aggregate results |
| `paper/lay_summary.txt` | General-audience summary for the submission system |

## Funding

This work was supported by the National Natural Science Foundation of China (Grant No. 62506134), the Guangdong-Hong Kong-Macao Applied Mathematics Center Project (Grant No. 2026A1515060008), and the Natural Science Foundation of Guangdong Province (Grant No. 2026A1515010171). The authors declare no competing interests.

## License

Project code follows the repository's MIT license declaration. Third-party model code retains its upstream attribution and terms; see the relevant directories and model-source references in `experiments/official_adaptive_models.py`.
