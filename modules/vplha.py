"""VPL-HA: Verified Pseudo-Label Head Adaptation (closed-loop).

TAPDA 训练完成后主干预(G)冻结，用「推理侧验证过」的目标域伪标签训练一个
轻量线性头，并把新的头预测重新送入 RATC/SSTI 的验证—精修环节，迭代 2 轮，
形成 "诊断(RATC) → 精修/验证(SSTI) → 目标头(RPL-HA) → 再诊断" 的闭环。
"""

from __future__ import annotations

import numpy as np


def _torch():
    import torch
    return torch


def _verified_mask(pseudo_labels, conf, f1_pred, f2_pred, args):
    tau = float(getattr(args, 'vplha_conf', 0.9))
    return ((f1_pred == pseudo_labels) & (f2_pred == pseudo_labels)
            & (conf >= tau))


def train_verified_head(feats, pseudo_labels, num_classes, conf, f1_pred, f2_pred, args,
                        mask=None):
    torch = _torch()
    import torch.nn as nn

    if mask is None:
        mask = _verified_mask(pseudo_labels, conf, f1_pred, f2_pred, args)
    n_ok = int(mask.sum())
    frac = n_ok / max(1, mask.shape[0])
    min_n = int(getattr(args, 'vplha_min_samples', 100))
    if n_ok < min_n:
        return None, frac

    X = torch.from_numpy(np.ascontiguousarray(feats[mask], dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(pseudo_labels[mask])).long()
    counts = torch.bincount(y, minlength=int(num_classes)).float().clamp_min(1.0)
    w = (counts.sum() / counts)
    w = w / w.mean()

    head = nn.Linear(X.shape[1], int(num_classes))
    opt = torch.optim.Adam(head.parameters(),
                           lr=float(getattr(args, 'vplha_lr', 0.01)),
                           weight_decay=float(getattr(args, 'vplha_wd', 1e-4)))
    lossf = nn.CrossEntropyLoss(weight=w,
                                label_smoothing=float(getattr(args, 'vplha_smooth', 0.1)))
    epochs = int(getattr(args, 'vplha_epochs', 200))
    bs = int(getattr(args, 'vplha_batch', 4096))
    n = X.shape[0]
    # test_model 外层是 torch.no_grad()，这里必须显式打开梯度
    with torch.enable_grad():
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                loss = lossf(head(X[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
    head.eval()
    return head, frac


def predict_head(head, feats, num_classes, batch=4096):
    torch = _torch()
    outs = []
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch):
            x = torch.from_numpy(np.ascontiguousarray(feats[i:i + batch], dtype=np.float32))
            outs.append(torch.softmax(head(x), dim=1).numpy())
    return np.concatenate(outs, 0)


def vplha_adapt(feats, base_p, f1_pred, f2_pred, labels_gt, num_classes,
                ssti_call, args):
    """Returns (final_labels, last_round_acc, info_dict)."""
    base_p = np.clip(np.asarray(base_p, dtype=np.float64), 1e-8, None)
    base_p = base_p / base_p.sum(axis=1, keepdims=True)

    if ssti_call is not None:
        pl = ssti_call(np.log(base_p))
    else:
        pl = base_p.argmax(axis=1)
    conf = base_p.max(axis=1)

    rounds = max(1, int(getattr(args, 'vplha_rounds', 2)))
    alpha = float(getattr(args, 'vplha_alpha', 0.3))
    pl_final = pl
    last_acc = float((pl_final == labels_gt).mean() * 100.0)
    info = {}

    # 验证掩码只计算一次（基于原始 base 预测），后续轮复用，
    # 否则 SSTI 改写伪标签后掩码会塌缩到接近 0。
    base_lab = base_p.argmax(axis=1)
    static_mask = _verified_mask(base_lab, base_p.max(axis=1), f1_pred, f2_pred, args)
    min_frac = float(getattr(args, 'vplha_min_frac', 0.05))
    if float(static_mask.mean()) < min_frac:
        print(f'[vplha] verified fraction {float(static_mask.mean())*100:.1f}% '
              f'< {min_frac*100:.0f}%, skip adaptation')
        info['verified_frac'] = float(static_mask.mean())
        info['final_acc'] = last_acc
        return pl_final, last_acc, info

    for r in range(rounds):
        head, frac = train_verified_head(feats, pl, num_classes, conf,
                                         f1_pred, f2_pred, args, mask=static_mask)
        info['verified_frac'] = frac
        if head is None:
            print(f'[vplha] round {r}: verified samples too few '
                  f'({frac * 100:.1f}%), stop')
            break
        hp = predict_head(head, feats, num_classes)
        ens = (1.0 - alpha) * base_p + alpha * hp
        ens = np.clip(ens, 1e-8, None)
        ens = ens / ens.sum(axis=1, keepdims=True)
        if ssti_call is not None:
            pl_new = ssti_call(np.log(ens))
        else:
            pl_new = ens.argmax(axis=1)
        acc = float((pl_new == labels_gt).mean() * 100.0)
        chg = float((pl_new != pl).mean() * 100.0)
        print(f'[vplha] round {r}: verified={frac * 100:.1f}% '
              f'head_alpha={alpha:.2f} change={chg:.2f}% OA={acc:.2f}%')
        pl_final = pl_new
        last_acc = acc
        pl = pl_new
        conf = ens.max(axis=1)

    info['final_acc'] = last_acc
    return pl_final, last_acc, info
