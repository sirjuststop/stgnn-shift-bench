# Sources for supplementary model adapters

| Local directory | Upstream | Commit | License |
|---|---|---|---|
| `supplementary_gts` | [chaoshangcs/GTS](https://github.com/chaoshangcs/GTS) | `8ed45ff1476639f78c382ff09ecca8e60523e7ce` | Apache-2.0, included as `LICENSE` |
| `supplementary_dgcrn` | [LakeChuan/DGCRN](https://github.com/LakeChuan/DGCRN) | `b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228` | MIT, included as `LICENSE` |

GTS source files: `model/pytorch/model.py` and `model/pytorch/cell.py`.
DGCRN source files: `methods/DGCRN/net.py` and `methods/DGCRN/layer.py`.

Local modifications (2026-11-07):

- Use package-relative imports; remove GTS's unused `lib.utils` import, avoiding
  an unnecessary TensorFlow dependency.
- Allocate GTS noise, hidden states, masks and lazy parameters on the input/module
  device. Register relation encodings as nonpersistent buffers so `.to(device)`
  moves them without changing the checkpoint parameter names.
- Preserve DGCRN's batch dimension for batch size one, use the current
  `orthogonal_` initializer spelling and the upstream GPU initialization convention
  on both CPU and CUDA, and raise a descriptive exception instead
  of terminating successfully on incompatible decoder covariates.

The training, data-loading, corruption, few-shot and reporting code in
`experiments/supplementary/` is specific to this repository. It is a reconstruction
with the settings documented in that directory, not the upstream training
pipeline or a recovered copy of the original supplementary scripts.
