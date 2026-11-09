# v7.0 precise change log

Base: original uploaded `RMDANet_v6.0.tar.gz`.

**Existing file altered**: `core/trainer.py` (CNT/SSF parameter declarations, source-basis initialization, augmentation safety gates, new CNT loss, source-prototype update gating, class feedback, diagnostics, new opt-in mode identifiers).

**New core files**: `core/semantic_safety.py`, `core/class_null_transport.py`.

**New supporting files**: `scripts/run_v7.py`, `scripts/run_three_v7.sh`, `scripts/eval_three_v7.sh`, `scripts/final_eval_locked.py`, `tests/test_ssf.py`, `README_V7.md`, `CHANGELOG_V7.md`.

**Original unchanged files** include *every* YAML, `core/model.py`, `core/prat.py`, `core/memory.py`, `core/data.py`, original dataset training entrypoints, existing checkpoint scoring, loss weights, sampling scheduler, original documentation, and all other bundled files.

**Not imported**: v2.4 CEMPA prototype attraction/confusion, spatial loss, entropy loss, v2.4 selection-score reweighting, v2.4 data alignment memory changes, v2.4 source sampler with min-steps-per-epoch.

**C0 default**: intended as a no-new-method baseline. The synthetic deterministic one-epoch parity test exactly matched unmodified v6.0 checkpoint tensors and selection dictionary.

The safe v7.0 script loads unchanged per-dataset YAML before applying explicit launcher flags; this addresses v6.0's YAML-overrides-CLI behavior **in the new launcher only**. Old entrypoints are intentionally unmodified.
