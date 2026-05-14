# fmridataset compatibility facade

`fmridataset` is no longer a feature-bearing package in this repository. It is
a thin compatibility facade over `fmrimod.dataset` so existing imports can move
gradually while the implementation has one source of truth.

Use `fmrimod.dataset` or the root `fmrimod` namespace in new code:

```python
from fmrimod.dataset import FmriDataset, matrix_dataset, fmri_dataset
```

The facade re-exports only canonical objects that have already landed in
`fmrimod.dataset`. Names that are not yet ported are intentionally absent
rather than provided as placeholders.

The canonical dataset overview lives in `docs/tutorials/datasets.qmd`.
