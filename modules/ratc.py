"""RATC: Reliability-Aware Test-Time Correction.

The framework diagnoses whether the target scene has drifted the BatchNorm
statistics (a label-free signal of domain/statistics mismatch) and then selects
one of two correction mechanisms:

    high BN shift   -> TENT   (BN-affine entropy minimisation, recalibrates the model)
    low  BN shift   -> MV-LCR (multi-view latent confusion recovery, corrects predictions)

Both mechanisms are inference-only and never use target labels.  The diagnosis
is computed from forward hooks: for every BatchNorm layer we compare the target
batch statistics (input to the BN layer) with the source running statistics:

    shift = mean_{layers,channels} |mu_t - mu_s| / sqrt(var_s)

A high shift means the target inputs feed the BN layers with statistics that the
source training never saw; recalibrating BN (TENT) is then the right move.
"""

from __future__ import annotations


def bn_shift(G, F1, F2, loader, device, n_batches=6):
    """Label-free BatchNorm statistic shift between target data and source stats."""
    import torch

    records = []
    hooks = []

    def make_hook(module):
        def hook(_m, inputs, _out):
            x = inputs[0]
            with torch.no_grad():
                dims = [0] + list(range(2, x.dim()))
                mu = x.mean(dim=dims).detach().cpu()
                var = x.var(dim=dims, unbiased=False).detach().cpu()
                records.append((mu, var,
                                module.running_mean.detach().cpu(),
                                module.running_var.detach().cpu()))
        return hook

    for net in (G, F1, F2):
        for module in net.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and \
                    module.running_mean is not None:
                hooks.append(module.register_forward_hook(make_hook(module)))

    for net in (G, F1, F2):
        net.eval()

    seen = 0
    with torch.no_grad():
        for data, _ in loader:
            _ = G(data.to(device))
            seen += 1
            if seen >= n_batches:
                break
    for h in hooks:
        h.remove()

    scores = []
    for mu_t, _var_t, mu_s, var_s in records:
        scores.append(float(((mu_t - mu_s).abs() / (var_s.sqrt() + 1e-6)).mean()))
    shift = sum(scores) / max(1, len(scores))
    return shift


def diagnose(G, F1, F2, loader, device, threshold=0.35, n_batches=6):
    shift = bn_shift(G, F1, F2, loader, device, n_batches=n_batches)
    choice = "tent" if shift > float(threshold) else "mvlcr"
    print(f"[ratc] BN shift = {shift:.4f} (threshold {threshold}) -> chose {choice}")
    return shift, choice
