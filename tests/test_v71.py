"""v7.1 unit-level checks with no access to target labels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
import numpy as np
import torch
import pytest
from data import budgeted_source_split
from v71_controls import (reliability_lmmd, maturity_snapshot,
                          classwise_cnt_shift, independent_spectral_view, js_agreement)
from model import DualReliabilityEncoder


def test_exact_source_budget_train_and_val_only():
    gt = np.zeros((10, 100), dtype=np.int64)
    gt[:5, :] = 1
    gt[5:, :] = 2
    train, val = budgeted_source_split(gt, total_budget_ratio=0.1, seed=1237, train_fraction=.8)
    assert (train > 0).sum() == 80
    assert (val > 0).sum() == 20
    assert (train * val == 0).all()
    assert not ((train > 0) | (val > 0)).all()
    assert sorted(np.unique(train).tolist()) == [0,1,2]


def test_lmmd_drops_unreliable_target_classes():
    sf = torch.randn(8, 5, requires_grad=True)
    tf = torch.randn(8, 5, requires_grad=True)
    labels = torch.tensor([0,0,0,0,1,1,1,1])
    zero = reliability_lmmd(sf, labels, tf, labels, torch.zeros(8,dtype=torch.bool),
                            torch.ones(8),2)
    assert zero.item() == 0
    zero.backward()
    assert sf.grad.abs().sum() == 0
    sf.grad.zero_()
    positive = reliability_lmmd(sf, labels, tf, labels,
                                torch.ones(8,dtype=torch.bool),torch.ones(8),2)
    assert positive.item() > 0
    positive.backward()
    assert sf.grad.abs().sum() > 0


def test_classwise_cnt_has_nonidentical_directions_without_target_gt():
    source=torch.ones(8,3,7,7)
    y=torch.tensor([0,0,0,0,1,1,1,1])
    source[y==1,1] = 2
    target=torch.ones(10,3,7,7)
    target[:5,0] = 1.5
    target[5:,1] = 2.5
    posterior=torch.tensor([[.99,.01]]*5+[[.01,.99]]*5)
    means=torch.stack([source[y==i,:,3,3].mean(0) for i in range(2)])
    mapped, strength, directions=classwise_cnt_shift(source,target,y,posterior,means,
                                                      torch.zeros(3,0),torch.ones(2),
                                                      max_ratio=.5,min_support=2)
    assert strength>0
    assert not torch.allclose(directions[0],directions[1])
    assert torch.isfinite(mapped).all()
    none, amount, _=classwise_cnt_shift(source,target,y,torch.full((10,2),.5),means,
                                         torch.zeros(3,0),torch.ones(2))
    assert amount==0
    assert torch.equal(none,source)


def test_maturity_is_zero_without_stable_evidence():
    class Mock:
        num_classes=3;min_age=2;min_streak=2;reliability_threshold=.5
        valid=torch.ones(10,dtype=torch.bool)
        age=torch.ones(10,dtype=torch.long)
        streak=torch.ones(10,dtype=torch.long)
        reliability_memory=torch.full((10,),.1)
        label_flip_count=torch.zeros(10,dtype=torch.long)
        prob_memory=torch.full((10,3),1/3)
    assert maturity_snapshot(Mock())['maturity']==0


def test_independent_view_and_js_agreement():
    x=torch.ones(4,15,7,7)
    x2=independent_spectral_view(x)
    assert x2.shape==x.shape
    p=torch.tensor([[.95,.05],[.2,.8]])
    assert torch.allclose(js_agreement(p,p),torch.ones(2),atol=1e-6)


def test_secondary_head_is_dormant_only_when_explicit():
    base=DualReliabilityEncoder(bands=24,num_classes=3,projection_dim=32)
    assert base.classifier2 is None
    with_head=DualReliabilityEncoder(bands=24,num_classes=3,projection_dim=32,enable_dual_head=True)
    assert with_head.classifier2 is not None
