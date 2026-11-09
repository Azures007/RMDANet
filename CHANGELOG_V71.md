# Changes from v7.0

- Added `core/v71_controls.py` for prior-debiased class/sample CNT transport, target perturbation stability, label-free maturity gating, and confidence-masked LMMD.
- Extended `core/semantic_safety.py` with an independent-perturbation target feedback accumulator; target feedback attenuation may now reach 30% rather than 10%, but never increases the augmentation scale.
- `core/trainer.py` now strictly rejects any target GT usage or in-training target OA, performs per-epoch target-maturity gating and active-epoch alignment warmup, computes LMMD only for reliably selected target samples, and optionally trains a second head only when active.
- `core/data.py` exposes `budgeted_source_split()`; `train+val` counts against the specified source-label budget. Remaining GT pixels are masked out of both loaders.
- `core/model.py`: secondary classifier head is conditional, absent by default.
- Source sampler supports `--min-steps-per-epoch` and `--steps-per-epoch`. Target loader supports small unlabeled pools.
- New scripts `run_v71.py`, `run_three_v71.sh`, `eval_three_v71.sh`, and unit tests `tests/test_v71.py`.

**Not implemented**: a theorem guaranteeing semantic invariance; independent target annotation or second sensor oracle; verified label-shift identification; spectral-axis FFT replacement of PRAT; a semantic-neighbor pooling redesign; spatially disjoint source validation; full real-data performance validation.
