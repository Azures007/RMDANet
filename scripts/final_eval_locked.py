"""Single-use, post-training target GT evaluation. Never called by training scripts."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
from data import load_source_scene,load_target_image_only,align_target_bands
from trainer import (DualReliabilityEncoder,build_parser,build_target_evaluation_loader,
                     evaluate_target,set_seed)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-path',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',required=True)
    p.add_argument('--confirm-final-eval',action='store_true')
    args=p.parse_args()
    if not args.confirm_final_eval: raise PermissionError('Explicit --confirm-final-eval required')
    path=Path(args.checkpoint);out=Path(args.output)
    if not path.is_file(): raise FileNotFoundError(path)
    if out.exists(): raise FileExistsError(f'Cannot overwrite final evaluation: {out}')
    try: checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: checkpoint=torch.load(path,map_location='cpu')
    if not all(k in checkpoint for k in ('args','model','selection')):
        raise RuntimeError('Invalid checkpoint: requires model, args, selection')
    train_config=checkpoint['args']
    if not bool(train_config.get('skip_final_evaluation',False)):
        raise RuntimeError('Checkpoint was not produced with strict training isolation')
    print('[FINAL EVAL] Checkpoint locked:',path,'selection:',checkpoint['selection'])
    ap=build_parser()
    ap.set_defaults(**train_config)
    t=ap.parse_args([])
    t.data_path=args.data_path
    t.num_workers=0
    device=torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    mode=str(getattr(t,'input_normalization','') or ('median' if t.amplitude_channel else 'l2'))
    amplitude_input=bool(getattr(t,'amplitude_input',False))
    # Source GT is permissible for source-only input preparation; target GT still unopened.
    src=load_source_scene(t.data_path,t.source_name,t.num_classes,normalize_mode=mode)
    target=load_target_image_only(t.data_path,t.target_name,normalize_mode=mode)
    aligned,_=align_target_bands(src.image,target.image,strength=t.band_alignment_strength,
                                  renormalize=mode=='l2')
    if amplitude_input:
        if target.amplitude is None:
            raise RuntimeError('amplitude_input requires a target amplitude map')
        aligned=np.concatenate([aligned,target.amplitude[...,None]],axis=-1)
    model=DualReliabilityEncoder(bands=aligned.shape[-1],num_classes=t.num_classes,
       projection_dim=t.projection_dim,projection_norm=t.projection_norm,
       aspp_mode=t.aspp_mode,pooling_mode=t.pooling_mode,
       spectral_fusion_mode=t.spectral_fusion_mode,amplitude_channel=t.amplitude_channel,
       amplitude_input=amplitude_input,
       backbone=str(getattr(t,'backbone','ours')),
       enable_dual_head=(t.dual_head_filter or t.lambda_dual_source > 0)).to(device)
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    print('[FINAL EVAL] Loading TARGET GT NOW for reporting ONLY')
    loader=build_target_evaluation_loader(t,aligned)
    metrics=evaluate_target(model,loader,device,num_classes=t.num_classes,use_tta=t.final_tta)
    payload={'checkpoint':str(path),'selection':checkpoint['selection'],
             'ssf_mode':train_config.get('ssf_mode'),'metrics':metrics}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    print(f"[FINAL EVAL] OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} Kappa={metrics['Kappa']:.4f}")
    print('[FINAL EVAL] per-class:',metrics['class_accuracy'])
    print('[FINAL EVAL] saved:',out)

if __name__=='__main__':main()
