# RMDANet v7.0 — v6.0 + CNT / SSF ONLY

**Base:** The original user-supplied `RMDANet_v6.0.tar.gz` (not the later `UniformAudit_FIXED` fork).
**Added algorithm:** CNT and SSF from the user-supplied `RMDANet_v2_4_SSF.tar.gz`.
**Mode C0 is default and should take the identical v6.0 training path.** The prior v6.0 `datasets/*/config.yaml` are unchanged.

## Core compatibility and scope

Only the existing `core/trainer.py` was changed, and the following core files were added:

- `core/semantic_safety.py` (copied CNT projection, bounded shift, teacher margin gate and epoch-lagged feedback from v2.4).
- `core/class_null_transport.py` (v2.4 import shim).

All original v6.0 `core/model.py`, `core/prat.py`, `core/memory.py`, `core/data.py`, dataset-specific configs and original losses are **byte-identical**. In particular **do not import v2.4's CEMPA, entropy, or spatial losses, PRAT style maxima, reliability thresholds, selection-score coefficients, or source sampler change**.

Only new loss term is `lambda_cnt * [weighted source cross entropy + 0.25 * source-margin squared penalty]` when CNT is enabled, with CNT warmup. Only SSF-gated branches may modify the existing PRAT loss/feature-pair and source prototype updates.

`C0`: original v6.0, no CNT, no safety gate, no feedback.
`C1`: v6.0 + CNT, no safety gate or feedback.
`C2`: v6.0 + CNT, source-teacher semantic safety gates CNT/PRAT supervised paths, no feedback.
`C3`: v6.0 + CNT + safety gates + previous-epoch PRAT/CNT classwise feedback and safe augmented feature prototype updates.
`C4`: v6.0 + PRAT safety and previous-epoch PRAT feedback, **without CNT**.

SSF starts at epoch 15 by default. CNT weight is 0.12 with 20-epoch warmup and max shift ratio 0.12. Teacher safety uses source labels only (never target ground-truth); target pseudo-label support is used only as a mild attenuation factor in C3/C4.

## Safe, recommended AutoDL workflow

Unpack under `/root/autodl-tmp/newv1/` and `cd RMDANet_v7.0`.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2

# Check the original per-dataset parameters without training:
python scripts/run_v7.py pavia C3 1237 --dry-run

# Compare C0 (v6.0) to C3 under the exact same original-v6 data/optimization protocol:
bash scripts/run_three_v7.sh C0 1237
bash scripts/run_three_v7.sh C3 1237

# Evaluate only once after all checkpoints are locked (loads target GT only here):
bash scripts/eval_three_v7.sh C0 1237
bash scripts/eval_three_v7.sh C3 1237
```

The original v6.0 **defaults differ between datasets**:

| Dataset | Source samples | Patch | Batch | Epochs |
|---|---|---:|---:|---:|
| PaviaU → PaviaC | 10% | 7 | 128 | 110 |
| Houston13 → Houston18 | 20% | 17 | 96 | 280 |
| Dioni → Loukia | 200/class | 7 | 128 | 120 |

The launcher only overrides `num_workers=0` (AutoDL memory stability), the data path, C-mode, seed, output path, and enables strict separation of target GT from training and checkpoint selection. The original YAML files remain unmodified.

### Optional low-label, unified screening (not v6 defaults)

```bash
COMMON='--source-train-ratio 0.05 --patch-size 7 --batch-size 24 --epochs 100 --target-train-max-samples 4000'
bash scripts/run_three_v7.sh C0 1237 $COMMON
bash scripts/run_three_v7.sh C3 1237 $COMMON
bash scripts/eval_three_v7.sh C0 1237 100
bash scripts/eval_three_v7.sh C3 1237 100
```

**Important:** The v6.0 source sampler is *not* changed, so this protocol does **not** enforce identical parameter updates per epoch among datasets; it only uses the same epoch count and batch size. This was intentional per request to keep all non-CNT/SSF code unchanged. With too-large batch sizes under 5% Houston training can have extremely few or zero full batches; using batch 24 mitigates it. The results cannot be directly compared to original v6.0 historical OA under different data budgets.

### Dataset-specific runs

```bash
python scripts/run_v7.py pavia C3 1237
python scripts/run_v7.py houston C3 1237
python scripts/run_v7.py hyrank C3 1237
```

Run names: `results_v7/{dataset}/{mode}_epochs{epochs}_seed{seed}/selected_model.pt`. Do not overwrite existing runs. If only one dataset needs evaluation:

```bash
python scripts/final_eval_locked.py \
  --checkpoint results_v7/pavia/C3_epochs110_seed1237/selected_model.pt \
  --data-path /root/autodl-tmp/Datasets/Pavia \
  --output results_v7/pavia/C3_epochs110_seed1237/target_final_once.json \
  --confirm-final-eval
```

This eval requires `--confirm-final-eval` and refuses to overwrite an existing report. During training, only SOURCE ground truth and unlabeled TARGET image are loaded. Do not use original v6.0 per-dataset entrypoints for strict UDA: some older configurable target-GT paths still exist there; use the new launcher.

## Checks and validation done before packaging

- All Python files compiled and shell scripts passed syntax checks.
- CNT/SSF tests: 7 passed.
- Synthetic SOURCE GT + TARGET image **with no target-GT file**: C0, C1, C2, C3, C4 trained and saved checkpoints.
- After training, target GT was introduced solely for locked final evaluation; evaluator completed successfully.
- One-epoch deterministic C0 equivalence test against an untouched v6.0 source checkout: **all floating model weights had max abs difference 0.0; checkpoint selection dict was exactly identical**. This is a synthetic parity test, **not** proof of bit-identical full training on GPU or of real-dataset accuracy gains.

No real Pavia/Houston/HyRANK target OA claims are made for v7.0 before actual training.
