#!/usr/bin/env python3
"""Verify that the residual RCGP formula is implemented in core/model.py.

Checks:
  1. CenterGuidedPooling creates a learnable `_alpha_logit` parameter when
     residual=True;
  2. forward() equals  gap + alpha * (guided - gap),  alpha = sigmoid(_alpha_logit).

Run from the repository root:
    python tests/test_drse_residual.py
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from model import CenterGuidedPooling  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    module = CenterGuidedPooling(
        8,
        center_size=5,
        attention_temperature=0.30,
        mode="guided",
        gate_init=(0.6, 0.3, 0.1),
        residual=True,
        alpha_logit=-4.0,
    )
    assert hasattr(module, "_alpha_logit"), "missing _alpha_logit parameter"
    assert isinstance(module._alpha_logit, torch.nn.Parameter), "_alpha_logit must be learnable"

    guided_only = copy.deepcopy(module)
    guided_only.residual = False
    guided_only._alpha_logit = None

    x = torch.randn(3, 8, 7, 7)
    pooled, _, _ = module(x)
    guided, _, _ = guided_only(x)
    gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
    alpha = torch.sigmoid(module._alpha_logit)
    expected = gap + alpha * (guided - gap)

    max_err = float((pooled - expected).abs().max())
    assert max_err < 1e-6, f"residual formula mismatch, max |err| = {max_err}"

    print(f"[OK] residual RCGP formula verified: alpha_init={alpha.item():.4f}, max_err={max_err:.2e}")


if __name__ == "__main__":
    main()
