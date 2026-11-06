"""Multi-View Latent Confusion Recovery (MV-LCR).

Views
-----
1. TAPDA classifier F1 probabilities on the target scene
2. TAPDA classifier F2 probabilities
3. A shallow source-trained spectral classifier (logistic regression on the
   centre-pixel spectra)

Model
-----
Dawid-Skene style latent class model

    P(y_hat^1, ..., y_hat^V) = sum_z pi_z prod_v C_v(y_hat^v | z)

fitted with EM on the *unlabeled* target predictions.  The view-specific
confusion matrices C_v let a systematically biased view be down-weighted, so
this is not a plain ensemble average and not a label-shift prior correction.
No target label and no training gradient is used.
"""

from __future__ import annotations

import numpy as np


def fit_mvlcr(view_probs, num_classes=None, iters=30, eps=1e-3,
              prior=None, fix_prior=False):
    """EM for the multi-view latent class model.

    view_probs: list of (N, C) probability matrices (one per view).
    Returns posterior (N, C).
    """
    probs = [np.asarray(p, dtype=np.float64) for p in view_probs]
    V = len(probs)
    N, C = probs[0].shape
    if num_classes is not None:
        C = int(num_classes)

    hard = np.stack([p.argmax(axis=1) for p in probs], axis=0)  # (V, N)

    # initialise from the most reliable view instead of the mean of all views
    post = probs[0].copy()
    post = post / np.maximum(post.sum(axis=1, keepdims=True), 1e-12)
    fixed_pi = None
    if fix_prior:
        fixed_pi = (np.asarray(prior, dtype=np.float64)
                    if prior is not None else np.full(C, 1.0 / C))
        fixed_pi = np.maximum(fixed_pi, eps)
        fixed_pi = fixed_pi / fixed_pi.sum()

    for it in range(max(1, int(iters))):
        if fixed_pi is not None:
            pi = fixed_pi
        else:
            pi = post.mean(axis=0)
            pi = np.maximum(pi, eps)
            pi = pi / pi.sum()
        confusion = np.full((V, C, C), eps)  # [v, predicted, true]
        for v in range(V):
            for c in range(C):
                m = hard[v] == c
                if np.any(m):
                    confusion[v, c] += post[m].sum(axis=0)
            confusion[v] /= confusion[v].sum(axis=1, keepdims=True)

        logp = np.tile(np.log(pi), (N, 1))
        for v in range(V):
            logp += np.log(confusion[v][hard[v]] + 1e-12)
        logp -= logp.max(axis=1, keepdims=True)
        new_post = np.exp(logp)
        new_post /= np.maximum(new_post.sum(axis=1, keepdims=True), 1e-12)
        if fixed_pi is not None:
            # IPF projection: force the marginal class distribution back to the
            # source prior, so minority classes cannot be absorbed by the EM.
            for _ in range(5):
                col = new_post.sum(axis=0)
                new_post *= (fixed_pi * len(new_post)) / np.maximum(col, 1e-12)
                new_post /= np.maximum(new_post.sum(axis=1, keepdims=True), 1e-12)
        delta = np.abs(new_post - post).mean()
        post = new_post
        if delta < 1e-6:
            break

    print(f"[mvlcr] fitted {V} views on {N} target pixels, iters={it}, "
          f"prior={np.round(pi, 3).tolist()}")
    return post


def spectral_view(train_x, train_y, test_x, num_classes):
    """Shallow source-trained spectral classifier as an extra view."""
    from sklearn.linear_model import LogisticRegression

    Xs = np.asarray(train_x, dtype=np.float64)
    ys = np.asarray(train_y).astype(np.int64).ravel()
    Xt = np.asarray(test_x, dtype=np.float64)
    if Xs.ndim > 2:
        Xs = Xs.reshape(Xs.shape[0], -1)
    if Xt.ndim > 2:
        Xt = Xt.reshape(Xt.shape[0], -1)
    mu, sd = Xs.mean(axis=0, keepdims=True), Xs.std(axis=0, keepdims=True) + 1e-6
    Xs = (Xs - mu) / sd
    Xt = (Xt - mu) / sd

    classes = np.unique(ys)
    # normalise to 0-based contiguous labels for sklearn output mapping
    if classes.min() >= 1 and classes.max() == len(classes):
        ys_fit = ys - 1
    else:
        ys_fit = ys - classes.min()
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(Xs, ys_fit)
    raw = clf.predict_proba(Xt)  # columns follow clf.classes_ (0..K-1)
    probs = np.full((len(Xt), int(num_classes)), 1e-6)
    probs[:, :raw.shape[1]] = raw
    probs /= probs.sum(axis=1, keepdims=True)
    print(f"[mvlcr] spectral view trained on {len(Xs)} source samples, "
          f"acc unknown (no target labels used)")
    return probs


def spatial_view(probs, coords, shape, passes=1):
    """Neighbourhood-averaged probability view (label-free, different error profile)."""
    import numpy as np
    from scipy.ndimage import uniform_filter

    P = np.asarray(probs, dtype=np.float32)
    rows, cols = np.asarray(coords[0], dtype=np.int64), np.asarray(coords[1], dtype=np.int64)
    h, w = int(shape[0]), int(shape[1])
    C = P.shape[1]
    grid = np.zeros((h, w, C), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    grid[rows, cols] = P
    mask[rows, cols] = 1.0

    cur = grid
    for _ in range(max(1, int(passes))):
        den = uniform_filter(mask, size=3, mode="constant")
        num = np.stack([uniform_filter(cur[..., c], size=3, mode="constant")
                        for c in range(C)], axis=-1)
        cur = num / np.maximum(den[..., None], 1e-6)
    out = cur[rows, cols]
    out = np.maximum(out, 1e-6)
    out /= out.sum(axis=1, keepdims=True)
    print(f"[mvlcr] spatial view built (passes={passes})")
    return out
