"""Single-use, post-training target GT evaluation. Never called by training scripts."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
def apply_temperature(probabilities, temperature):
    if abs(float(temperature) - 1.0) < 1e-6:
        return probabilities
    logits = np.log(np.clip(probabilities, 1e-12, None)) / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
from data import (PatchDataset,align_target_bands,coordinates_from_mask,
                  labels_at_coordinates,load_source_scene,load_target_ground_truth_only,
                  load_target_image_only)
from trainer import (DualReliabilityEncoder,build_parser,build_target_evaluation_loader,
                     evaluate_target,set_seed,predict_probabilities)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-path',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',required=True)
    p.add_argument('--confirm-final-eval',action='store_true')
    p.add_argument('--prior-calib',action='store_true',dest='prior_calib')
    p.add_argument('--adabn',action='store_true',dest='adabn')
    p.add_argument('--sliding',action='store_true',dest='sliding')
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
    # Infer the true input channel count from the checkpoint weights (the
    # args["bands"] entry may be missing when the front-end changes channels).
    try:
        inferred_bands=int(checkpoint['model']['spectral_encoder.raw_projection.1.weight'].shape[1])
        t.bands=inferred_bands
        print(f'[FINAL EVAL] inferred bands={inferred_bands} from checkpoint')
    except Exception:
        pass
    device=torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    mode='median' if t.amplitude_channel else 'l2'
    # Source GT is permissible for source-only input preparation; target GT still unopened.
    src=load_source_scene(t.data_path,t.source_name,t.num_classes,normalize_mode=mode)
    target=load_target_image_only(t.data_path,t.target_name,normalize_mode=mode)
    aligned,_=align_target_bands(src.image,target.image,strength=t.band_alignment_strength,
                                  renormalize=mode!='median')
    # Match the checkpoint structure: older checkpoints (pre-residual) have no
    # `_alpha_logit`, newer (residual RCGP) ones do.  Single-branch checkpoints
    # (DRSE disabled) have no spectral branch / fusion keys.
    has_residual_alpha = any(str(k).endswith("_alpha_logit") for k in checkpoint["model"].keys())
    has_spectral_branch = any(
        str(k).startswith(("spectral_encoder.", "spectral_fusion.")) for k in checkpoint["model"].keys()
    )
    model=DualReliabilityEncoder(bands=int(getattr(t,'bands',src.image.shape[-1])),num_classes=t.num_classes,
       projection_dim=t.projection_dim,projection_norm=t.projection_norm,
       aspp_mode=t.aspp_mode,pooling_mode=t.pooling_mode,
       spectral_fusion_mode=t.spectral_fusion_mode,amplitude_channel=t.amplitude_channel,
       use_spectral_branch=has_spectral_branch,
       pool_center_size=getattr(t,"pool_center_size",5),
       pool_temperature=getattr(t,"pool_temperature",0.30),
       pool_gate_init=tuple(float(v) for v in str(getattr(t,"pool_gate_init","0.6,0.3,0.1")).split(",")),
       pool_residual=has_residual_alpha,
       pool_alpha_logit=getattr(t,"pool_alpha_logit",-4.0),
       fusion_alpha=getattr(t,"fusion_alpha",1.0),
       fusion_delta=getattr(t,"fusion_delta",-0.2),
       fusion_tau_a=getattr(t,"fusion_tau_a",0.20),
       fusion_t_a=getattr(t,"fusion_t_a",0.10),
       fusion_t_w=getattr(t,"fusion_t_w",0.15),
       fusion_w_min=getattr(t,"fusion_w_min",0.10),
       fusion_w_max=getattr(t,"fusion_w_max",0.90)).to(device)
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    print('[FINAL EVAL] Loading TARGET GT NOW for reporting ONLY')
    loader=build_target_evaluation_loader(t,aligned)
    if args.adabn:
        # P7: AdaBN — recompute BatchNorm statistics on unlabeled target data.
        bn_modules=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
        print(f'[FINAL EVAL] AdaBN on {len(bn_modules)} BN layer(s)')
        for m in bn_modules:m.train()
        with torch.no_grad():
            for _ in range(3):
                for batch in loader:
                    model(batch[0].to(device,non_blocking=True))
        model.eval()
    _tent_env = os.environ.get("TENT", "").strip().lower()
    if _tent_env == "":
        tent_enabled = str(getattr(t, "dataset", "")).lower() == "hyrank"
    else:
        tent_enabled = _tent_env in ("1", "true", "yes")
    if tent_enabled:
        # Test-time entropy minimization: adapt only BatchNorm affine params/stats
        # on unlabeled target data (no target GT used).
        bn_params = []
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()
                if module.weight is not None:
                    bn_params.append(module.weight)
                if module.bias is not None:
                    bn_params.append(module.bias)
        if not bn_params:
            print("[TENT] no BatchNorm affine parameters found; skipped")
        else:
            for module in model.modules():
                if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            optimizer = torch.optim.Adam(bn_params, lr=float(os.environ.get("TENT_LR", "0.001")))
            steps = int(os.environ.get("TENT_STEPS", "50"))
            iterator = iter(loader)
            done = 0
            while done < steps:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                x = batch[0].to(device, non_blocking=True)
                logits = model(x).logits.float()
                prob = torch.softmax(logits, dim=1)
                loss = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                done += 1
            model.eval()
            print(f"[TENT] adapted BN affine ({len(bn_params)} tensors) in {steps} steps")
    if args.sliding:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from eval_prior_calib import em_target_prior,metrics_from_confusion
        gt=load_target_ground_truth_only(t.data_path,t.target_name,t.num_classes)
        valid=(gt>0)&(np.linalg.norm(aligned,axis=-1)>0)
        coordinates=coordinates_from_mask(valid)
        labels_np=labels_at_coordinates(gt,coordinates)-1
        slab=dict(pavia=(2,1.0),houston=(2,1.5),hyrank=(1,1.5))
        _key=str(getattr(t,'dataset','') or '').lower()
        _d_radius,_d_temp=slab.get(_key,(1,1.0))
        sl_radius=int(os.environ.get('SLIDING_RADIUS',str(_d_radius)))
        offsets=[(dy,dx) for dy in range(-sl_radius,sl_radius+1) for dx in range(-sl_radius,sl_radius+1)]
        height,width=aligned.shape[:2]
        prob_sum=None
        weight_sum=0.0
        sigma=float(os.environ.get('SLIDING_SIGMA','0'))
        for dy,dx in offsets:
            shifted=coordinates+np.array([dy,dx])
            shifted[:,0]=np.clip(shifted[:,0],0,height-1)
            shifted[:,1]=np.clip(shifted[:,1],0,width-1)
            dataset=PatchDataset(aligned,shifted,patch_size=t.patch_size,labels=labels_np,augment=False)
            loader_p10=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False)
            offset_probs=[]
            with torch.no_grad():
                for batch in loader_p10:
                    x=batch[0].to(device,non_blocking=True)
                    offset_probs.append(torch.softmax(model(x).logits.float(),dim=1).cpu())
            offset_mean=torch.cat(offset_probs)
            weight = 1.0 if sigma <= 0.0 else float(np.exp(-(dy * dy + dx * dx) / (2.0 * sigma * sigma)))
            weight_sum += weight
            weighted = offset_mean * weight
            prob_sum = weighted if prob_sum is None else prob_sum + weighted
        probabilities=(prob_sum/weight_sum).numpy().astype('float64')
        _smooth_env = os.environ.get('SPATIAL_SMOOTH', '')
        if _smooth_env == '':
            smooth_sigma = 4.0 if str(getattr(t, 'dataset', '')).lower() == 'pavia' else 0.0
        else:
            smooth_sigma = float(_smooth_env)
        if smooth_sigma>0:
            from scipy.ndimage import gaussian_filter
            prob_map=np.zeros((height,width,t.num_classes),dtype='float32')
            prob_map[coordinates[:,0],coordinates[:,1]]=probabilities.astype('float32')
            for cidx in range(t.num_classes):
                prob_map[:,:,cidx]=gaussian_filter(prob_map[:,:,cidx],sigma=smooth_sigma,mode='nearest')
            probabilities=prob_map[coordinates[:,0],coordinates[:,1]].astype('float64')
            probabilities=probabilities/np.clip(probabilities.sum(axis=1,keepdims=True),1e-12,None)
        probabilities=apply_temperature(probabilities, float(os.environ.get('EVAL_TEMPERATURE',str(_d_temp))))
        source_prior=np.full(t.num_classes,1.0/t.num_classes)
        p2_iters=int(os.environ.get('P2_EM_ITERS','30'))
        p2_smooth=float(os.environ.get('P2_SMOOTH','0.5'))
        p2_floor=float(os.environ.get('P2_FLOOR','0.02'))
        p2_alpha=float(os.environ.get('P2_ALPHA','0.5'))
        target_prior=em_target_prior(probabilities,source_prior,iterations=p2_iters)
        target_prior=(1.0-p2_smooth)*target_prior+p2_smooth*source_prior
        target_prior=np.clip(target_prior,p2_floor,None);target_prior=target_prior/target_prior.sum()
        probabilities=np.asarray(probabilities).reshape(-1,t.num_classes)
        target_prior=np.asarray(target_prior).reshape(-1)
        source_prior=np.asarray(source_prior).reshape(-1)
        if os.environ.get("CLASSLWISE_ALPHA","").strip()=="1":
            weak_mask=target_prior < (1.0/t.num_classes)*0.7
            alpha_vec=np.where(weak_mask,0.8,0.3).reshape(-1)
        else:
            alpha_vec=np.full(t.num_classes,p2_alpha)
        adjusted=probabilities*np.exp(alpha_vec[None,:]*np.log(target_prior/source_prior)[None,:])
        prediction=adjusted.argmax(axis=1)
        confusion=np.zeros((t.num_classes,t.num_classes),dtype=np.int64)
        for truth,pred in zip(labels_np,prediction):confusion[truth,pred]+=1
        metrics=metrics_from_confusion(confusion)
        metrics['inference']={'sliding_offsets':len(offsets),'prior_calib':True}
    elif args.prior_calib:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from eval_prior_calib import em_target_prior,metrics_from_confusion
        probabilities_list=[];label_list=[]
        with torch.no_grad():
            for batch in loader:
                x=batch[0].to(device,non_blocking=True)
                probabilities_list.append(predict_probabilities(model,x,use_tta=t.final_tta).float().cpu())
                label_list.append(batch[1].cpu())
        probabilities=torch.cat(probabilities_list).numpy().astype('float64')
        _slab2=dict(pavia=(2,1.0),houston=(2,1.5),hyrank=(1,1.5))
        _k2=str(getattr(t,'dataset','') or '').lower()
        _d_temp2=_slab2.get(_k2,(1,1.0))[1]
        probabilities=apply_temperature(probabilities, float(os.environ.get('EVAL_TEMPERATURE',str(_d_temp2))))
        labels=torch.cat(label_list).numpy().astype('int64')
        num_classes=t.num_classes
        source_prior=np.full(num_classes,1.0/num_classes,dtype='float64')
        p2_iters=int(os.environ.get('P2_EM_ITERS','30'))
        p2_smooth=float(os.environ.get('P2_SMOOTH','0.5'))
        p2_floor=float(os.environ.get('P2_FLOOR','0.02'))
        p2_alpha=float(os.environ.get('P2_ALPHA','0.5'))
        target_prior=em_target_prior(probabilities,source_prior,iterations=p2_iters)
        target_prior=(1.0-p2_smooth)*target_prior+p2_smooth*source_prior
        target_prior=np.clip(target_prior,p2_floor,None);target_prior=target_prior/target_prior.sum()
        probabilities=np.asarray(probabilities).reshape(-1,num_classes)
        target_prior=np.asarray(target_prior).reshape(-1)
        source_prior=np.asarray(source_prior).reshape(-1)
        if os.environ.get("CLASSLWISE_ALPHA","").strip()=="1":
            weak_mask=target_prior < (1.0/num_classes)*0.7
            alpha_vec=np.where(weak_mask,0.8,0.3).reshape(-1)
        else:
            alpha_vec=np.full(num_classes,p2_alpha)
        adjusted=probabilities*np.exp(alpha_vec[None,:]*np.log(target_prior/source_prior)[None,:])
        prediction=adjusted.argmax(axis=1)
        confusion=np.zeros((num_classes,num_classes),dtype='int64')
        for truth,pred in zip(labels,prediction):confusion[truth,pred]+=1
        metrics=metrics_from_confusion(confusion)
        metrics['prior_calib']={'alpha':0.5,'smooth':0.5,'floor':0.02,'target_prior':target_prior.tolist()}
    else:
        metrics=evaluate_target(model,loader,device,num_classes=t.num_classes,use_tta=t.final_tta)
    payload={'checkpoint':str(path),'selection':checkpoint['selection'],'metrics':metrics}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    print(f"[FINAL EVAL] OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} Kappa={metrics['Kappa']:.4f}")
    print('[FINAL EVAL] per-class:',metrics['class_accuracy'])
    print('[FINAL EVAL] saved:',out)

if __name__=='__main__':main()
