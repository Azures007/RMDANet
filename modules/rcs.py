"""Risk-Certified Supplemental Supervision (RCS).

Idea
----
Whether a target pseudo-label may be used for discriminative supervision is
decided by a *risk certificate* instead of a hand-tuned confidence threshold:

1. On the SOURCE domain (labels available) we observe four label-free signals
     s = [teacher confidence, top1-top2 margin,
          F1/F2 agreement, teacher-student agreement]
   together with the indicator "teacher prediction is correct", and fit a
   logistic calibrator q(s) = P(pseudo-label correct | s).
2. On the target domain the same q(s) is evaluated.  Sorting samples by
   decreasing q, the largest prefix whose estimated risk
       R(k) = (1/k) * sum_{i<=k} (1 - q_i)
   stays below delta is certified for supervision.

Under the standard calibration-transfer assumption (the mapping s -> P(correct)
does not depend on the domain), R(k) is an unbiased estimate of the pseudo-label
error rate of the supervised set, so the selected set keeps risk <= delta.
"""

from __future__ import annotations

import numpy as np

_STATE = {
    "features": [],
    "labels": [],
    "model": None,
}


def rcs_reset():
    _STATE["features"].clear()
    _STATE["labels"].clear()
    _STATE["model"] = None


def rcs_signals(teacher_prob, logits1, logits2):
    """Four label-free signals, shape (N, 4), numpy float32."""
    import torch

    with torch.no_grad():
        p = teacher_prob.detach().float()
        conf, pred_t = p.max(dim=1)
        top2 = p.topk(min(2, p.shape[1]), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0]))
        pred1 = logits1.detach().argmax(dim=1)
        pred2 = logits2.detach().argmax(dim=1)
        agree12 = (pred1 == pred2).float()
        agree_ts = (pred1 == pred_t).float()
        feats = torch.stack([conf, margin, agree12, agree_ts], dim=1)
        return feats.cpu().numpy().astype(np.float32)


def rcs_observe(features, correct):
    """Accumulate source-domain calibration samples."""
    _STATE["features"].append(np.asarray(features, dtype=np.float32))
    _STATE["labels"].append(np.asarray(correct, dtype=np.int64))


def rcs_num_samples():
    return int(sum(len(x) for x in _STATE["features"]))


def rcs_fit(min_samples=200):
    """Fit the logistic calibrator on the accumulated source samples (once)."""
    if _STATE["model"] is not None:
        return True
    if rcs_num_samples() < min_samples:
        return False
    X = np.concatenate(_STATE["features"], axis=0)
    y = np.concatenate(_STATE["labels"], axis=0)
    if len(np.unique(y)) < 2:
        return False
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return False
    model = LogisticRegression(max_iter=200, C=1.0)
    model.fit(X, y)
    _STATE["model"] = model
    print(f"[rcs] calibrator fitted on {len(y)} source samples "
          f"(positive rate {y.mean():.3f})")
    return True


def rcs_predict(features):
    """Pseudo-label correctness probability for target samples."""
    model = _STATE["model"]
    if model is None:
        return None
    return model.predict_proba(np.asarray(features, dtype=np.float32))[:, 1]


def rcs_risk_select(q, delta=0.10, min_frac=0.05, max_frac=0.60):
    """Return indices of the largest prefix with estimated risk <= delta."""
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
    kmax = max(kmax, kmin)
    limit = max(1, int(np.floor(float(max_frac) * n)))
    k = min(kmax, limit)
    keep = order[:k]
    print(f"[rcs] certified {k}/{n} samples (estimated risk {risks[k-1]:.4f} <= {delta})")
    return keep.astype(np.int64)
