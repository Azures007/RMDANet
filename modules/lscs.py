"""Label-Shift-aware Corrective Supervision (LSCS).

Motivation (measured on Houston, see docs/evidence/TAPDA_PROBLEM_EVIDENCE.md):
  * the target teacher is over-confident: confidence 0.945 vs accuracy 79.25%;
  * source-domain calibration does not transfer (source 93.2% -> target 79.25%,
    naive risk certification underestimates the error by ~30x);
  * supervising samples that the student already classifies correctly adds no
    information (all-sample CE is neutral);
  * every previous variant hurt the minority class (class 0 drops by 15-35 pts).

LSCS therefore
  1. estimates the target class prior with BBSE (source confusion matrix +
     target prediction histogram) and re-weights the source calibrator by
     pi_t(y)/pi_s(y) -- a label-shift-aware calibration;
  2. supervises ONLY corrective samples: teacher == F2 != F1 (a 2-vs-1
     disagreement, i.e. places where the current classifier is wrong);
  3. certifies the selected set with a risk budget delta and a per-class quota.
"""

from __future__ import annotations

import numpy as np

from modules.rcs import rcs_signals

_STATE = {
    "num_classes": 0,
    "feat": [],
    "correct": [],
    "cls": [],
    "src_conf": None,
    "src_counts": None,
    "tgt_hist": None,
    "prior_t": None,
    "prior_s": None,
    "model": None,
}


def lscs_reset(num_classes=None):
    _STATE["feat"].clear()
    _STATE["correct"].clear()
    _STATE["cls"].clear()
    _STATE["src_conf"] = None
    _STATE["src_counts"] = None
    _STATE["tgt_hist"] = None
    _STATE["prior_t"] = None
    _STATE["prior_s"] = None
    _STATE["model"] = None
    if num_classes:
        _STATE["num_classes"] = int(num_classes)


def _ensure(num_classes):
    c = int(num_classes)
    if _STATE["src_conf"] is None or _STATE["src_conf"].shape[0] != c:
        _STATE["src_conf"] = np.zeros((c, c), dtype=np.float64)   # rows: true, cols: pred
        _STATE["src_counts"] = np.zeros(c, dtype=np.float64)
        _STATE["tgt_hist"] = np.zeros(c, dtype=np.float64)


def lscs_observe_source(teacher_prob, logits1, logits2, labels):
    """Accumulate source calibration samples and the source confusion matrix."""
    import torch

    c = int(teacher_prob.shape[1])
    _ensure(c)
    feats = rcs_signals(teacher_prob, logits1, logits2)
    with torch.no_grad():
        pred = teacher_prob.detach().argmax(dim=1).cpu().numpy().astype(np.int64)
        true = labels.detach().cpu().numpy().astype(np.int64)
    _STATE["feat"].append(feats)
    _STATE["correct"].append((pred == true).astype(np.int64))
    _STATE["cls"].append(true)
    for t, p in zip(true, pred):
        _STATE["src_conf"][t, p] += 1.0
        _STATE["src_counts"][t] += 1.0
    return feats


def lscs_update_target_hist(teacher_pred, momentum=0.9):
    """EMA of the target prediction histogram (no labels used)."""
    import torch

    with torch.no_grad():
        pred = teacher_pred.detach().cpu().numpy().astype(np.int64)
    c = _STATE["num_classes"]
    _ensure(c)
    hist = np.bincount(pred, minlength=c).astype(np.float64)
    _STATE["tgt_hist"] = momentum * _STATE["tgt_hist"] + (1.0 - momentum) * hist \
        if _STATE["tgt_hist"].sum() > 0 else hist


def lscs_estimate_prior():
    """BBSE: pi_t = argmin ||C pi - mu||^2 with simplex projection."""
    c = _STATE["num_classes"]
    if _STATE["src_conf"] is None or _STATE["src_counts"] is None:
        return None
    if np.sum(_STATE["src_conf"]) < 500:
        return None
    C = _STATE["src_conf"] / np.maximum(_STATE["src_conf"].sum(axis=1, keepdims=True), 1.0)
    mu = _STATE["tgt_hist"] / max(1.0, _STATE["tgt_hist"].sum())
    if mu.sum() <= 0:
        return None
    try:
        pi = np.linalg.lstsq(C, mu, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s <= 0:
        return None
    pi = pi / s
    _STATE["prior_t"] = pi
    _STATE["prior_s"] = _STATE["src_counts"] / max(1.0, _STATE["src_counts"].sum())
    return pi


def lscs_fit(min_samples=500):
    """Fit the label-shift-weighted calibration model (once)."""
    if _STATE["model"] is not None:
        return True
    if sum(len(x) for x in _STATE["feat"]) < min_samples:
        return False
    if _STATE["prior_t"] is None:
        if lscs_estimate_prior() is None:
            # fall back: no label-shift correction (weight = 1)
            _STATE["prior_t"] = np.ones(_STATE["num_classes"], dtype=np.float64) / _STATE["num_classes"]
            _STATE["prior_s"] = np.ones(_STATE["num_classes"], dtype=np.float64) / _STATE["num_classes"]
            print("[lscs] prior estimation failed, using uniform weights")
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return False
    X = np.concatenate(_STATE["feat"], axis=0)
    y = np.concatenate(_STATE["correct"], axis=0)
    cls = np.concatenate(_STATE["cls"], axis=0)
    if len(np.unique(y)) < 2:
        return False
    w = _STATE["prior_t"][cls] / np.maximum(_STATE["prior_s"][cls], 1e-6)
    w = np.clip(w, 0.05, 20.0)
    model = LogisticRegression(max_iter=300, C=1.0)
    model.fit(X, y, sample_weight=w)
    _STATE["model"] = model
    print(f"[lscs] calibrator fitted on {len(y)} source samples with label-shift weights "
          f"(prior_t={np.round(_STATE['prior_t'], 3).tolist()})")
    return True


def lscs_risk_select(q, delta=0.10, min_frac=0.02, max_frac=0.60):
    q = np.asarray(q, dtype=np.float64)
    n = len(q)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-q)
    qs = q[order]
    risks = np.cumsum(1.0 - qs) / np.arange(1, n + 1)
    ok = np.nonzero(risks <= float(delta))[0]
    kmax = int(ok[-1] + 1) if len(ok) else 0
    kmin = max(1, int(np.ceil(float(min_frac) * n)))
    kmax = max(kmax, kmin, 1)
    k = min(kmax, max(1, int(np.floor(float(max_frac) * n))))
    return order[:k].astype(np.int64)


def lscs_select(teacher_prob, logits1, logits2, delta=0.10, class_cap=8,
                min_frac=0.02, max_frac=0.60, num_classes=None):
    """Corrective + risk-certified + class-capped selection.

    Candidate set: teacher == F2 and teacher != F1 (2-vs-1 correction).
    Returns indices into the current target batch.
    """
    import torch

    c = int(teacher_prob.shape[1])
    _ensure(c)
    with torch.no_grad():
        pred_t = teacher_prob.detach().argmax(dim=1)
        pred1 = logits1.detach().argmax(dim=1)
        pred2 = logits2.detach().argmax(dim=1)
        cand = (pred_t == pred2) & (pred_t != pred1)
        feats = rcs_signals(teacher_prob, logits1, logits2)
    idx = cand.nonzero(as_tuple=True)[0].cpu().numpy()
    if len(idx) == 0:
        return idx.astype(np.int64)
    if _STATE["model"] is None:
        return np.zeros(0, dtype=np.int64)
    q = _STATE["model"].predict_proba(feats[idx])[:, 1]
    order = np.argsort(-q)
    idx, q = idx[order], q[order]
    # risk-certified prefix
    risks = np.cumsum(1.0 - q) / np.arange(1, len(q) + 1)
    ok = np.nonzero(risks <= float(delta))[0]
    k = int(ok[-1] + 1) if len(ok) else 0
    k = max(k, max(1, int(np.ceil(float(min_frac) * len(q)))))
    k = min(k, max(1, int(np.floor(float(max_frac) * len(q)))))
    idx, q = idx[:k], q[:k]
    # per-class quota (prevent minority collapse)
    pred_np = pred_t.cpu().numpy()
    keep = []
    counts = {}
    for i in idx.tolist():
        cls = int(pred_np[i])
        if counts.get(cls, 0) >= int(class_cap):
            continue
        counts[cls] = counts.get(cls, 0) + 1
        keep.append(i)
    out = np.asarray(keep, dtype=np.int64)
    print(f"[lscs] candidates={len(cand.nonzero(as_tuple=True)[0])} "
          f"certified={k} kept={len(out)} risk={risks[k-1]:.4f}<= {delta} "
          f"classes={ {k: v for k, v in counts.items()} }")
    return out
