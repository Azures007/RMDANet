"""Synthetic CPU unit tests for modules/rgs_selector.py (no dataset required)."""

import torch

from modules.rgs_selector import select_rgs_samples


class Args:
    def __init__(self, mode="legacy", thr=0.95, weight="teacher_conf", cap=0):
        self.rgs_selector = mode
        self.rgs_conf_threshold = thr
        self.rgs_weight_mode = weight
        self.rgs_class_cap = cap


def _toy(n=8, c=3, seed=0):
    torch.manual_seed(seed)
    probs = torch.softmax(torch.randn(n, c), dim=1)
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values
    f1 = torch.randn(n, c)
    f2 = torch.randn(n, c)
    proj = torch.randn(n, 16)
    proto = torch.randn(c, 16)
    seen = torch.ones(c, dtype=torch.long)
    return pred, conf, f1, f2, proj, {"prototypes": proto, "seen": seen}


def test_legacy_returns_none():
    pred, conf, f1, f2, proj, st = _toy()
    idx, labels, w, stats = select_rgs_samples(pred, conf, f1, f2, proj, st, 3, 1, Args("legacy"))
    assert idx is None and labels is None and w is None
    assert stats["mode"] == "legacy"


def test_independent_conf_threshold():
    pred, conf, f1, f2, proj, st = _toy()
    idx, labels, w, stats = select_rgs_samples(pred, conf, f1, f2, proj, st, 3, 1, Args("independent_conf", thr=0.9))
    assert idx is not None
    assert idx.numel() == int((conf >= 0.9).sum())
    assert torch.all(conf[idx] >= 0.9)
    assert torch.equal(labels, pred[idx])


def test_independent_agree_uses_f2():
    pred, conf, f1, f2, proj, st = _toy()
    args = Args("independent_agree", thr=0.0)
    idx, labels, w, stats = select_rgs_samples(pred, conf, f1, f2, proj, st, 3, 1, args)
    f2p = f2.argmax(dim=1)
    expect = (pred == f2p).sum().item()
    assert idx.numel() == expect
    assert torch.all(labels == pred[idx])
    assert 0.0 <= stats["agree_rate"] <= 1.0


def test_independent_proto_masking():
    pred, conf, f1, f2, proj, st = _toy()
    st["seen"] = torch.tensor([1, 0, 1])
    args = Args("independent_proto", thr=0.0)
    idx, labels, w, stats = select_rgs_samples(pred, conf, f1, f2, proj, st, 3, 1, args)
    assert idx is not None
    # unseen-class samples must never be selected by proto support
    assert torch.all(st["seen"][labels] > 0)


def test_empty_selection_is_safe():
    pred, conf, f1, f2, proj, st = _toy()
    args = Args("independent_conf", thr=1.1)
    idx, labels, w, stats = select_rgs_samples(pred, conf, f1, f2, proj, st, 3, 1, args)
    assert idx.numel() == 0 and labels.numel() == 0 and w.numel() == 0
    assert stats["empty"] is True


def test_index_label_alignment_and_unique():
    pred, conf, f1, f2, proj, st = _toy(n=64)
    args = Args("independent_conf", thr=0.0)
    idx, labels, w, stats = select_rgs_samples(pred, conf, f1, f2, proj, st, 3, 1, args)
    assert idx.unique().numel() == idx.numel()
    assert torch.equal(labels, pred[idx])
    assert w.shape == idx.shape
    assert not w.requires_grad


def test_no_grad_leak():
    pred, conf, f1, f2, proj, st = _toy()
    proj2 = proj.clone().requires_grad_(True)
    idx, labels, w, stats = select_rgs_samples(pred, conf, f1, f2, proj2, st, 3, 1, Args("independent_conf", thr=0.0))
    assert not idx.requires_grad and not labels.requires_grad and not w.requires_grad


def test_uniform_vs_teacher_conf_weight():
    pred, conf, f1, f2, proj, st = _toy()
    i1, _, w1, _ = select_rgs_samples(pred, conf, f1, f2, proj, st, 3, 1, Args("independent_conf", 0.0, "uniform"))
    i2, l2, w2, _ = select_rgs_samples(pred, conf, f1, f2, proj, st, 3, 1, Args("independent_conf", 0.0, "teacher_conf"))
    assert torch.equal(i1, i2)
    assert torch.allclose(w1, torch.ones_like(w1))
    assert torch.allclose(w2, conf[i2])


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {exc}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
