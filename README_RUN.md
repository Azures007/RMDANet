TAPDA + SSTI - standalone release
==================================

This package contains ONLY the TAPDA base pipeline plus our innovations.
The original research repository (core/, v7.1 launchers, MSDA baseline code)
is NOT included and was NOT modified.

Contents
--------
  up2pc.py            Pavia (PaviaU -> PaviaC) entry
  houston13218.py     Houston (Houston13 -> Houston18) entry
  nc122nc13.py        YRD (NC12 -> NC13) entry
  modules/            model blocks, losses and our modules:
                        ssti.py            source-structured transductive inference
                        hdfa.py            (if present) auxiliary heads
                        v30_losses.py      RGS / reliability losses
                        ...
  util/               data loaders and helpers
  datasets/           placeholder folder (put the .mat files here on the server)

Innovations implemented in this package
---------------------------------------
  1. RGS: reliability-gated pseudo-label supervision (flag --reliab_ce_weight)
  2. Reliability operators: maturity gate / three-head consensus / class quota
     (flags --m1_maturity_gate, --m1_consensus, --m1_class_quota)
  3. Diagnostic-driven operator selection (no flag; choose the operator per dataset)
  4. SSTI: source-structured transductive inference (flag --ssti, inference only)

Recommended commands (run inside this directory)
------------------------------------------------
  # Pavia: RGS + maturity gate + SSTI
  python -u up2pc.py --gpu 0 --num_seeds 5 \
    --reliab_ce_weight 0.05 --m1_maturity_gate --ssti

  # Houston: RGS + three-head consensus + SSTI
  python -u houston13218.py --gpu 0 --num_seeds 5 \
    --reliab_ce_weight 0.05 --m1_consensus --ssti

  # YRD: RGS only + SSTI
  python -u nc122nc13.py --gpu 0 --num_seeds 5 \
    --reliab_ce_weight 0.02 --ssti

SSTI parameters (defaults used in the paper, identical for all datasets)
  --ssti_alpha 0.3        label-shift prior weight
  --ssti_beta 0.5         spatial transition weight
  --ssti_iters 3          ICM iterations
  --ssti_conf_threshold 0.95   only pixels below this confidence may change
  --ssti_structure predicted   transition prior source (default, see protocol below)

SSTI protocols (important for fair comparison)
----------------------------------------------
  --ssti_structure predicted  STRICT label budget (main experiments).
      The class-transition prior T and the source class prior pi_s are estimated
      from the SOURCE MODEL'S PREDICTIONS on the source image (dense sliding-window
      inference). No source labels beyond the 180/class training budget are used.
      The prediction map is computed lazily after training, at evaluation time.

  --ssti_structure gt         FULL source structure (supplementary upper-bound).
      T and pi_s are estimated from the complete source ground-truth map, which is
      extra supervision beyond the 180/class budget. Report this separately and
      never mix it with the strict-budget table.

  Target labels are only used to locate the evaluation pixels (the standard
  benchmark mask). They never enter the prior, the gate, or the ICM score.

Data layout expected on the server
  datasets/Pavia/     paviaU.mat, paviaU_gt_7.mat, paviaC.mat, paviaC_gt_7.mat
  datasets/Houston/   houston13.mat ... houston18.mat
  datasets/YRD/       NC12.mat, NC13.mat (or the YRD_extract folder)
The exact filenames are the ones used by util/utils.py; adjust the path
functions at the top of each entry script if your layout differs.

Results (paired, same seeds, without -> with SSTI)
--------------------------------------------------
  Pavia    5 seeds: 92.99 +- 0.84  ->  93.48 +- 0.92   (+0.49)
  Houston  5 seeds: 80.20 (3 seeds) -> 81.96 +- 1.00   (+1.60)
  YRD      5 seeds: 95.16 (3 seeds) -> 95.38 +- 0.43   (+0.29)
  11 / 11 paired seeds improved; sign test p ~ 0.0005.
