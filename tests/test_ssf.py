import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
import torch
from semantic_safety import (true_margin,judge_augmentation,weighted_ce,
                             class_basis_from_centers,transport,SafetyFeedback)


def test_true_margin():
    l=torch.tensor([[3.,1.,0.],[0.,4.,3.]])
    y=torch.tensor([0,1])
    assert torch.allclose(true_margin(l,y),torch.tensor([2.,1.]))


def test_safety_rejects_semantic_flip():
    y=torch.tensor([0,1,0])
    clean=torch.tensor([[5.,1.],[1.,5.],[4.,1.]])
    aug=torch.tensor([[4.,1.],[5.,1.],[2.,1.]])
    safe,d=judge_augmentation(clean,aug,y,tolerance=0.30)
    assert safe.tolist()==[True,False,False]
    assert d.shape == (3,)


def test_weighted_ce_zero_safe():
    x=torch.randn(4,3,requires_grad=True)
    y=torch.tensor([0,1,2,1])
    z=weighted_ce(x,y,torch.zeros(4))
    z.backward()
    assert torch.all(x.grad==0)


def test_basis_orthogonal_to_null_shift():
    x=torch.tensor([[1.,0.,.1],[1.1,0.,.2],[0.,1.,.4],[0.,1.2,.3]])
    y=torch.tensor([0,0,1,1])
    basis=class_basis_from_centers(x,y,2)
    assert basis.shape==(3,1)
    source=torch.randn(3,3,7,7)
    target=torch.randn(3,3,7,7)
    mapped,_=transport(source,target,basis,torch.ones(2),torch.tensor([0,1,0]))
    diff=(mapped-source).mean(dim=(0,2,3))
    assert torch.abs(diff @ basis[:,0])<1e-5


def test_transport_mask_and_bound_and_per_class():
    source=torch.ones(4,6,7,7)
    source[0,:,0,0]=0
    target=torch.full_like(source,3)
    q=torch.zeros(6,0)
    scale=torch.tensor([1.,.5])
    mapped,shift=transport(source,target,q,scale,torch.tensor([0,0,1,1]))
    assert torch.all(mapped[0,:,0,0]==0)
    assert torch.allclose(mapped[0,:,3,3]-source[0,:,3,3],
                          2*(mapped[2,:,3,3]-source[2,:,3,3]),atol=1e-5)
    assert shift <= .12+1e-5


def test_feedback_is_epoch_lagged_and_uses_no_target_gt():
    ctl=SafetyFeedback(2,'cpu',min_source=2,target_support=2,ema_decay=0.,min_scale=.5)
    assert ctl.scale_cnt.tolist()==[1,1]
    ctl.observe_source(torch.tensor([0,0,1,1]),torch.tensor([True,False,True,True]),branch='cnt')
    ctl.observe_source(torch.tensor([0,0,1,1]),torch.tensor([True,True,False,False]),branch='prat')
    ctl.observe_target(torch.tensor([0,0,1,1]),torch.tensor([True,True,False,False]),
                       torch.tensor([False,True,True,True]))
    assert ctl.scale_cnt.tolist()==[1,1]
    result=ctl.end_epoch()
    assert ctl.scale_cnt[0]<ctl.scale_cnt[1]
    assert ctl.scale_prat[1]<ctl.scale_prat[0]
    assert result['target_supported_classes']==1
    assert torch.all(ctl.target==0)


def test_no_target_GT_import_in_safety_module():
    source=(Path(__file__).resolve().parents[1]/'core/semantic_safety.py').read_text()
    assert 'load_target_ground_truth' not in source
