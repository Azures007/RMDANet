# RMDANet v7.1: 25-point implementation review

**Base:** RMDANet v7.0 (`C0` to `C4`). The experimental model backbone, PRAT Fourier algorithm, memory system and source classification loss are retained except for conditional removal of the untrained secondary classifier. The adaptation behavior and source-label budget **are intentionally changed** from v7.0. These changes do not guarantee better target OA.

## Important research-protocol change

`--source-train-ratio 0.05` now means **at most 5% of labeled source pixels per class IN TOTAL** (train + validation), not 5% train and 95% validation. By default approximately 4% source labels are used in training and 1% in validation. If `source_samples_per_class=200`, there are at most 200 accessible labels per class in train+validation combined, divided 160/40. Remaining source GT pixels are excluded from both data loaders and checkpoint selection. Input ground-truth files are read to randomly select a budget, but excluded labels are never used for supervision or validation. Ratios are per-class rounded to the nearest pixel.

Do **not** compare v7.1 target OA to v7.0 OA without correcting for this changed label-access protocol.

Target ground truth is unavailable in training. The training code refuses `use_gt_target_mask=True`, nonzero `target_gt_mix_ratio`, target OA diagnostics, and in-training final evaluation. It refuses `dual_head_filter=True` unless classifier2 has positive source supervised loss.

## v7.1 training mechanisms

1. **Class-conditional CNT**: estimates soft target-class spectral centers from sufficiently confident EMA teacher probabilities; derives different class directions from source-only budgeted class means; debiases a global mean estimate for predicted class prior differences; optionally adjusts direction for sample residuals. Unsupported classes receive **zero shift**. A bounded teacher-margin backtracking test can attenuate candidate perturbations. None of these proves semantic invariance; soft class membership can itself be wrong.
2. **Reliability-conditioned aligners**: CCA, CORAL and LMMD are gated by a label-free maturity estimator derived from stored unlabeled target predictions, history stability, flip rate, class coverage and reliability. With `--align-maturity-gate`, alignment's effective start time is event-driven; CCA/LMMD warmup uses the number of active epochs, not an arbitrary fixed start epoch. During unconfident target warmup the aligner may remain off for the *entire* run.
3. **LMMD**: uses only stable pseudo-label predictions weighted by reliability, with a minimum sample/effective support requirement per class. No fallback to unfiltered hard pseudo labels. If eligible target samples are absent, LMMD is exactly zero.
4. **Target consistency**: a new spectral-response nuisance perturbation of the target input is evaluated by EMA teacher, without reusing `stable_mask` as the evidence; fragile target pseudo supervision is downweighted, and SSF's target feedback can use this signal. It is not an independent teacher oracle and does not prove correctness.
5. **Dual head**: classifier2 is absent unless a positive `lambda_dual_source` and/or explicitly configured dual-head supervision warrants it. `--dual-head-filter` without a positive source supervised second-head loss fails before training.
6. **Low label budget safety**: WeightedRandomSampler repeats *the same budgeted source training labels* enough times for at least five optimizer steps per epoch. `--steps-per-epoch N` enforces exactly N updates for any dataset and is useful for short fair ablations.

## Running on AutoDL

Upload `RMDANet_v7.1.tar.gz` to `/root/autodl-tmp/newv1`:

```bash
cd /root/autodl-tmp/newv1
tar -xzf RMDANet_v7.1.tar.gz
cd RMDANet_v7.1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2
```

One dataset, one mode, one seed:

```bash
python -u scripts/run_v71.py pavia C3 1237 \
  --source-train-ratio 0.05 --patch-size 7 --batch-size 24 \
  --epochs 100 --steps-per-epoch 8 --target-train-max-samples 4000
```

Replace `pavia` with `houston` or `hyrank` for the other datasets; `C0` is an ablation **within v7.1**, not exactly the old v7.0 C0. Available modes: C0/C1/C2/C3/C4. `--dry-run` prints the effective configuration without data loading or training.

Run all three sequentially:

```bash
bash scripts/run_three_v71.sh C3 1237 --source-train-ratio 0.05 \
  --patch-size 7 --batch-size 24 --epochs 100 \
  --steps-per-epoch 8 --target-train-max-samples 4000
```

After **locking the checkpoint**, one-time evaluation of the same 100-epoch configuration:

```bash
bash scripts/eval_three_v71.sh C3 1237 100
```

Training outputs: `results_v71/<dataset>/<MODE>_epochs<EPOCHS>_seed<SEED>/selected_model.pt`. The independent evaluator creates `target_final_once.json` and refuses to overwrite it.

## Known limitations

- A soft target class prediction is not target ground truth, so class-prior debiasing and class-conditional CNT are not identifiable without additional assumptions.
- Neither classifier-margin checks nor class-mean-null projections mathematically guarantee source semantic invariance.
- Student, EMA teacher, target prototype and perturbation robustness are still correlated signals. Wrong but highly confident/stable pseudo labels can survive.
- Target maturity is a heuristic computed from the same system and may remain zero or be confidently wrong. No target GT is used to tune it at train time.
- The source train/validation coordinates may be spatial neighbors; this version does not implement a spatially disjoint split.
- No claim of state-of-the-art OA or novelty is made: no real-data training was used to validate the modifications here.

See `AUDIT_25_ITEMS_V71.md` for the full 25-row ledger in its original order.

## Quick test checklist

```bash
python -m compileall -q core scripts
python -m pytest -q tests/test_ssf.py tests/test_v71.py
python scripts/run_v71.py pavia C3 1237 --source-train-ratio 0.05 \
  --patch-size 7 --batch-size 24 --epochs 100 --steps-per-epoch 8 --dry-run
```

The exact per-dataset effective parameters, including original v6 dataset-specific
loss weights, are printed by `--dry-run`. In a full run, use `training_config.json`
as the authoritative set of effective parameters.
