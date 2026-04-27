import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.official_adaptive_common import (
    build_agcrn_arrays,
    build_megacrn_arrays,
    build_source_dataset,
    build_staeformer_arrays,
    compute_masked_metrics,
    compute_metrics,
    ensure_parent,
    make_torch_loader,
    scaler_mode_for_model,
    set_seed,
    update_registry,
)
from experiments.official_adaptive_models import (
    MODEL_SOURCE_INFO,
    build_model,
    resolve_train_config,
)
from third_party.official_megacrn.utils import masked_mae_loss as megacrn_masked_mae_loss
from third_party.official_staeformer.lib.utils import MaskedMAELoss as STAEformerMaskedMAELoss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['agcrn', 'megacrn', 'staeformer'], required=True)
    ap.add_argument('--dataset', default='metr-la', choices=['metr-la', 'pems-bay', 'pems04'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output_dir', default='baselines/stgnn')
    ap.add_argument('--registry', default='baselines/stgnn_protocol_registry.json')
    ap.add_argument('--log_interval', type=int, default=0)
    return ap.parse_args()


def resolve_runtime_config(model_name, dataset_key, args):
    cfg = resolve_train_config(model_name, dataset_key)
    for key in ['epochs', 'patience', 'batch_size', 'lr', 'weight_decay']:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return cfg


def make_loaders(model_name, dataset_key, batch_size):
    ds = build_source_dataset(dataset_key, scaler_mode=scaler_mode_for_model(model_name))
    scaler = ds['scaler']
    if model_name == 'agcrn':
        builder = build_agcrn_arrays
    elif model_name == 'megacrn':
        builder = build_megacrn_arrays
    elif model_name == 'staeformer':
        builder = build_staeformer_arrays
    else:
        raise KeyError(model_name)
    x_train, y_train = builder(ds['train'], scaler)
    x_val, y_val = builder(ds['val'], scaler)
    x_test, y_test = builder(ds['test'], scaler)
    loaders = {
        'train': make_torch_loader(x_train, y_train, batch_size, shuffle=True, drop_last=False),
        'val': make_torch_loader(x_val, y_val, batch_size, shuffle=False, drop_last=False),
        'test': make_torch_loader(x_test, y_test, batch_size, shuffle=False, drop_last=False),
    }
    arrays = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
    }
    return ds, scaler, loaders, arrays


def forward_and_loss(model_name, model, batch_x, batch_y, scaler, criterion=None, train=False, batches_seen=0, cfg=None):
    if model_name == 'agcrn':
        pred_norm = model(batch_x, None)
        target_norm = batch_y
        pred_raw = scaler.inverse_transform_torch(pred_norm)
        target_raw = scaler.inverse_transform_torch(target_norm)
        loss = nn.L1Loss()(pred_raw, target_raw)
        return pred_norm, target_norm, pred_raw, target_raw, loss, batches_seen

    if model_name == 'megacrn':
        x_val = batch_x[..., :1]
        y_norm = batch_y[..., :1]
        y_cov = batch_y[..., 1:]
        labels = y_norm if train else None
        pred_norm, _, query, pos, neg = model(x_val, y_cov, labels, batches_seen if train else None)
        pred_raw = scaler.inverse_transform_torch(pred_norm)
        target_raw = scaler.inverse_transform_torch(y_norm)
        loss1 = megacrn_masked_mae_loss(pred_raw, target_raw)
        separate_loss = nn.TripletMarginLoss(margin=1.0)(query, pos.detach(), neg.detach())
        compact_loss = nn.MSELoss()(query, pos.detach())
        loss = loss1 + cfg['lamb'] * separate_loss + cfg['lamb1'] * compact_loss
        next_batches_seen = batches_seen + 1 if train else batches_seen
        return pred_norm, y_norm, pred_raw, target_raw, loss, next_batches_seen

    if model_name == 'staeformer':
        pred_norm = model(batch_x)
        target_raw = batch_y
        target_norm = scaler.transform_torch(target_raw)
        pred_raw = scaler.inverse_transform_torch(pred_norm)
        loss = criterion(pred_raw, target_raw)
        return pred_norm, target_norm, pred_raw, target_raw, loss, batches_seen

    raise KeyError(model_name)


@torch.no_grad()
def evaluate(model_name, model, loader, scaler, device, cfg=None):
    criterion = STAEformerMaskedMAELoss() if model_name == 'staeformer' else None
    model.eval()
    losses = []
    pred_raws, target_raws = [], []
    pred_norms, target_norms = [], []
    batches_seen = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred_norm, target_norm, pred_raw, target_raw, loss, batches_seen = forward_and_loss(
            model_name,
            model,
            batch_x,
            batch_y,
            scaler,
            criterion=criterion,
            train=False,
            batches_seen=batches_seen,
            cfg=cfg,
        )
        losses.append(loss.item())
        pred_raws.append(pred_raw.cpu())
        target_raws.append(target_raw.cpu())
        pred_norms.append(pred_norm.cpu())
        target_norms.append(target_norm.cpu())

    pred_raw = torch.cat(pred_raws, dim=0)
    target_raw = torch.cat(target_raws, dim=0)
    pred_norm = torch.cat(pred_norms, dim=0)
    target_norm = torch.cat(target_norms, dim=0)
    return {
        'loss': float(np.mean(losses)),
        'pred_raw': pred_raw,
        'target_raw': target_raw,
        'pred_norm': pred_norm,
        'target_norm': target_norm,
        'metrics_raw': compute_metrics(pred_raw, target_raw),
        'metrics_raw_masked': compute_masked_metrics(pred_raw, target_raw),
        'metrics_norm': compute_metrics(pred_norm, target_norm),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    runtime_cfg = resolve_runtime_config(args.model, args.dataset, args)

    ds, scaler, loaders, _ = make_loaders(args.model, args.dataset, runtime_cfg['batch_size'])
    model, model_cfg = build_model(args.model, args.dataset, device)
    total_train_batches = len(loaders['train'])
    log_interval = args.log_interval
    if log_interval <= 0:
        if args.model == 'megacrn':
            log_interval = 50
        elif args.model == 'staeformer':
            log_interval = 100
        else:
            log_interval = 0

    optimizer_kwargs = {'lr': runtime_cfg['lr'], 'weight_decay': runtime_cfg['weight_decay']}
    if args.model == 'megacrn':
        optimizer_kwargs['eps'] = runtime_cfg.get('epsilon', 1e-8)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    scheduler = None
    if args.model == 'megacrn':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=runtime_cfg['steps'], gamma=runtime_cfg['lr_decay_ratio']
        )
    elif args.model == 'staeformer':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=runtime_cfg['milestones'], gamma=runtime_cfg['lr_decay_ratio']
        )

    stae_criterion = STAEformerMaskedMAELoss()
    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    batches_seen = 0

    for epoch in range(runtime_cfg['epochs']):
        model.train()
        train_losses = []
        for batch_idx, (batch_x, batch_y) in enumerate(loaders['train'], start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred_norm, target_norm, pred_raw, target_raw, loss, batches_seen = forward_and_loss(
                args.model,
                model,
                batch_x,
                batch_y,
                scaler,
                criterion=stae_criterion,
                train=True,
                batches_seen=batches_seen,
                cfg=runtime_cfg,
            )
            loss.backward()
            grad_clip = runtime_cfg.get('grad_clip', 0.0) or 0.0
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

            if log_interval and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_train_batches):
                print(
                    f"[{args.model}] epoch={epoch+1} batch={batch_idx}/{total_train_batches} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        if scheduler is not None:
            scheduler.step()

        val_res = evaluate(args.model, model, loaders['val'], scaler, device, cfg=runtime_cfg)
        if val_res['loss'] < best_val:
            best_val = val_res['loss']
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= runtime_cfg['patience']:
                break

        print(
            f"[{args.model}] epoch={epoch+1} train_loss={np.mean(train_losses):.4f} "
            f"val_loss={val_res['loss']:.4f} val_MAE={val_res['metrics_raw']['MAE']:.4f} "
            f"val_MAE_masked={val_res['metrics_raw_masked']['MAE']:.4f}",
            flush=True,
        )

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = max(best_epoch, epoch + 1 if 'epoch' in locals() else 0)

    model.load_state_dict(best_state)
    test_res = evaluate(args.model, model, loaders['test'], scaler, device, cfg=runtime_cfg)

    ckpt = {
        'model_name': args.model,
        'dataset': args.dataset,
        'seed': args.seed,
        'model_config': model_cfg,
        'train_config': runtime_cfg,
        'scaler_mean': scaler.mean.tolist(),
        'scaler_std': scaler.std.tolist(),
        'state_dict': model.state_dict(),
        'best_epoch': best_epoch + 1,
        'source': MODEL_SOURCE_INFO[args.model],
        'test_metrics_raw': test_res['metrics_raw'],
        'test_metrics_raw_masked': test_res['metrics_raw_masked'],
        'test_metrics_norm': test_res['metrics_norm'],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{args.model}__{args.dataset}__seed{args.seed}.pt"
    ckpt_path = out_dir / ckpt_name
    torch.save(ckpt, ckpt_path)

    entry = {
        'key': ckpt_name[:-3],
        'backbone': args.model,
        'dataset': args.dataset,
        'seed': args.seed,
        'checkpoint': str(ckpt_path),
        'test_metrics': test_res['metrics_raw'],
        'test_metrics_masked': test_res['metrics_raw_masked'],
        'status': 'frozen',
        'source': MODEL_SOURCE_INFO[args.model],
    }
    update_registry(args.registry, ckpt_name[:-3], entry)

    summary = {
        'checkpoint': str(ckpt_path),
        'best_epoch': best_epoch + 1,
        'test_metrics_raw': test_res['metrics_raw'],
        'test_metrics_raw_masked': test_res['metrics_raw_masked'],
        'test_metrics_norm': test_res['metrics_norm'],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
