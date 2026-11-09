# v7.1: 25-point review in the same order as the v7.0 audit

| # | Prior issue | v7.1 status | What changed / what remains |
|---:|---|---|---|
| 1 | CNT global mean confuses class prior and domain shift | **已改进，未根除** | Source-prior correction + confident target soft class estimates; identifiability still absent when teacher estimates wrong. |
| 2 | Class-mean null space does not prove semantic invariance | **已增强保护，理论未根除** | Bounded teacher-margin source backtracking; does not prove class or feature invariance. |
| 3 | One CNT direction for all classes | **已修改** | Target class-conditioned transport directions, not only class scalars. |
| 4 | No instance-dependent domain direction | **已修改（启发式）** | Local residual matching provides optional sample corrections, gated by class target support. |
| 5 | SSF target feedback self-repeats stable_mask | **已修改（证据仍相关）** | Spectral input perturbation on all target samples, Teacher JS-consistency independent of `stable_mask` selection. Same Teacher model remains. |
| 6 | SSF cannot guarantee semantic invariance | **已增强保护，理论未根除** | Empirical Teacher correctness/margin checks remain; no mathematical guarantee. |
| 7 | Target pseudo-label error accumulation | **已增强保护，不能保证消除** | Perturbation-derived soft reliability attenuation; wrong consistent predictions may persist. |
| 8 | Reliability evidence not independent | **已改进，未完全独立** | Independent perturbation rather than an extra check on selected `stable_mask`; predictions remain from the same EMA Teacher. |
| 9 | HyRANK LMMD uses all pseudo labels | **已修改** | Stable mask and reliability-weighted target class means; no unfiltered fallback. |
| 10 | LMMD doesn't use reliability weights | **已修改** | Class conditional target means are reliability weighted; excludes classes with low support. |
| 11 | CCA starts before target maturity | **已修改（启发式门控）** | Requires unlabeled target maturity and valid source prototypes when maturity gate enabled. |
| 12 | Alignment uses fixed epoch instead of target maturity | **已修改** | CCA/LMMD/CORAL adaptively gated by target maturity; warmup counts active epochs. |
| 13 | LMMD warmup argument unused | **已修改** | `lmmd_warm_epochs` participates in LMMD effective weight. |
| 14 | Secondary head dormant | **已修改** | No second classifier module in default model. |
| 15 | Untrained second head could filter pseudo labels | **已修改** | Enabling the filter without positive second-head source supervision raises an error. |
| 16 | 5% source train + 95% source GT validation | **已修改** | All source GT used for train **plus validation** is contained in the specified 5% source-label budget. |
| 17 | v7.0 default is uniformly 5% source | **没问题／原描述不适用** | Defaults still retain individual original source budgets; 5% is an explicit CLI option. |
| 18 | Target GT used in official training | **已修改（硬拒绝）** | Any target-GT mask, diagnostic target OA or inline final eval is rejected in `train()`; target data loader has no label-mask path. |
| 19 | PRAT FFT is spatial, not wavelength-axis FFT | **原状保留／方法局限** | PRAT unchanged; still 2D spatial FFT per spectral band, not a wavelength-response transport model. |
| 20 | PRAT novelty resembles prior frequency UDA | **未修改／需文献验证** | No claim about novelty can be settled by code changes. |
| 21 | PRAT should be replaced by spectral transport | **未修改／待实验** | Preserved PRAT as requested. |
| 22 | Center pooling is not explicitly semantic-aware | **未修改／可选研究** | No evidence it is a bug; backbone pooling retained. |
| 23 | No true class-prior shift solution | **已增强，仍未彻底解决** | CNT prior deconfounding and existing prior EMA, but true target priors unidentifiable without assumptions. |
| 24 | CCA/LMMD conflict with pseudo reliability | **已缓解，不能证明无冲突** | LMMD masks pseudo classes by reliability; CCA gated on maturity; different objectives may still conflict. |
| 25 | SSF handles magnitude not class/sample direction | **已修改** | CNT class-conditioned + local sample residual directions, SSF still controls classwise strength. |

**Status discipline:** An implemented safeguard is NOT a correctness proof. Full real-dataset and multi-seed comparisons are still required before claiming a performance improvement.
