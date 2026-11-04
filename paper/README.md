# NeurIPS 2026 camera-ready paper

**Traffic STGNNs across Sensor, City, and Time Shifts: Routing Concentration Tracks Sensitivity to Sensor Dropout**

Xile Wang and Mingqi Yang, South China University of Technology.

This directory contains the final-format manuscript, references, appendix, checklist, and the funding statement supplied by the authors. The compiled PDF has 29 pages, with the main text ending on page 9. The original submission and earlier local revisions are preserved separately.

Revision v5 (2026-11-04) incorporates corrections to protocol comparisons,
transfer rankings, statistical reporting, and manuscript wording.

## Build the PDF

Run from this directory with a LaTeX distribution:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The source package includes `main.bbl`, the style file, and every required figure and table. Building the PDF does not require Python, datasets, or checkpoints.

## Regenerate the revised assets

```sh
python -m pip install numpy matplotlib
python figures/build_camera_ready_assets.py --from-packaged-inputs
```

`inputs/asset_sources.json` contains the saved numerical inputs needed to rebuild Figure 1 and the revised tables. No private review text or original submission archive is needed. `generated/metrics.json` records reaggregated statistics; `generated/rebuttal_reported.json` separately records the additional aggregate results supplied in the authors' rebuttal. This command regenerates presentation assets; it does not rerun experiments.

In the complete research workspace, the generator can also read its original source files by omitting `--from-packaged-inputs`.

`lay_summary.txt` is a paragraph for the submission system's general-audience summary field. The repository is https://github.com/sirjuststop/stgnn-shift-bench.
