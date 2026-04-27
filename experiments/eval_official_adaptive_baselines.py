import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.official_adaptive_common import (
    StandardScaler,
    build_agcrn_arrays,
    build_megacrn_arrays,
    build_source_dataset,
    build_staeformer_arrays,
    compute_masked_metrics,
    fit_train_scaler,
    load_registry,
    make_torch_loader,
    sample_target_subsets,
    scaler_mode_for_model,
)
from experiments.official_adaptive_models import build_model_from_config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['agcrn', 'megacrn', 'staeformer'])
    ap.add_argument('--registry', default='baselines/stgnn_protocol_registry.json')
    ap.add_argument('--source_dataset', default='metr-la', choices=['metr-la'])
    ap.add_argument('--target_dataset', default='pems-bay', choices=['pems-bay'])
    ap.add_argument('--output_dir', default='results/official_adaptive_relation')
    ap.add_argument('--n_subsets', type=int, default=5)
    ap.add_argument('--mask_seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--dropout_ratio', type=float, default=0.30)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()


def build_arrays(model_name, frame, scaler):
    if model_name == 'agcrn':
        return build_agcrn_arrays(frame, scaler)
    if model_name == 'megacrn':
        return build_megacrn_arrays(frame, scaler)
    if model_name == 'staeformer':
        return build_staeformer_arrays(frame, scaler)
    raise KeyError(model_name)


def forward_model(model_name, model, batch_x, batch_y):
    if model_name == 'agcrn':
        return model(batch_x, None)
    if model_name == 'megacrn':
        pred_norm, _, _, _, _ = model(batch_x[..., :1], batch_y[..., 1:], None, None)
        return pred_norm
    if model_name == 'staeformer':
        return model(batch_x)
    raise KeyError(model_name)


@torch.no_grad()
def collect_predictions(model_name, model, loader, scaler, device):
    model.eval()
    pred_norms, target_norms, pred_raws, target_raws = [], [], [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred_norm = forward_model(model_name, model, batch_x, batch_y)
        if model_name == 'staeformer':
            target_raw = batch_y
            target_norm = scaler.transform_torch(target_raw)
        elif model_name == 'megacrn':
            target_norm = batch_y[..., :1]
            target_raw = scaler.inverse_transform_torch(target_norm)
        else:
            target_norm = batch_y
            target_raw = scaler.inverse_transform_torch(target_norm)
        pred_raw = scaler.inverse_transform_torch(pred_norm)
        pred_norms.append(pred_norm.cpu())
        target_norms.append(target_norm.cpu())
        pred_raws.append(pred_raw.cpu())
        target_raws.append(target_raw.cpu())
    return {
        'pred_norm': torch.cat(pred_norms, dim=0),
        'target_norm': torch.cat(target_norms, dim=0),
        'pred_raw': torch.cat(pred_raws, dim=0),
        'target_raw': torch.cat(target_raws, dim=0),
    }


def mae_tensor(pred, target):
    return (pred - target).abs().mean().item()


def evaluate_sensor_dropout(model_name, model, x_test, y_test, scaler, device, dropout_ratio, mask_seed):
    rng = np.random.RandomState(mask_seed)
    n_nodes = x_test.shape[2]
    n_drop = int(n_nodes * dropout_ratio)
    drop_idx = sorted(rng.choice(n_nodes, n_drop, replace=False).tolist())
    keep_idx = [i for i in range(n_nodes) if i not in drop_idx]
    x_masked = x_test.copy()
    x_masked[:, :, drop_idx, 0] = 0.0
    clean_loader = make_torch_loader(x_test, y_test, batch_size=64, shuffle=False)
    masked_loader = make_torch_loader(x_masked, y_test, batch_size=64, shuffle=False)
    clean = collect_predictions(model_name, model, clean_loader, scaler, device)
    masked = collect_predictions(model_name, model, masked_loader, scaler, device)
    clean_mae = mae_tensor(clean['pred_norm'][:, :, keep_idx, :], clean['target_norm'][:, :, keep_idx, :])
    masked_mae = mae_tensor(masked['pred_norm'][:, :, keep_idx, :], masked['target_norm'][:, :, keep_idx, :])
    return {
        'mask_seed': mask_seed,
        'clean_MAE_kept': clean_mae,
        'masked_MAE_kept': masked_mae,
        'delta_sensor': masked_mae - clean_mae,
        'kept_nodes': len(keep_idx),
        'dropped_nodes': n_drop,
    }


def evaluate_cross_city(model_name, model, source_scaler, subsets, device):
    src_scaler_global = StandardScaler(
        np.full((1, subsets[0]['train'].shape[1]), source_scaler.global_mean, dtype=np.float32),
        np.full((1, subsets[0]['train'].shape[1]), source_scaler.global_std, dtype=np.float32),
    )
    src_maes = []
    tgt_maes = []
    tgt_maes_masked = []
    for subset in subsets:
        target_scaler = fit_train_scaler(
            subset['train'], mode=scaler_mode_for_model(model_name)
        )
        x_src, y_src = build_arrays(model_name, subset['test'], src_scaler_global)
        loader_src = make_torch_loader(x_src, y_src, batch_size=64, shuffle=False)
        preds_src = collect_predictions(model_name, model, loader_src, src_scaler_global, device)
        src_maes.append(mae_tensor(preds_src['pred_norm'], preds_src['target_norm']))

        x_tgt, y_tgt = build_arrays(model_name, subset['test'], target_scaler)
        loader_tgt = make_torch_loader(x_tgt, y_tgt, batch_size=64, shuffle=False)
        preds_tgt = collect_predictions(model_name, model, loader_tgt, target_scaler, device)
        tgt_maes.append(mae_tensor(preds_tgt['pred_raw'], preds_tgt['target_raw']))
        tgt_maes_masked.append(compute_masked_metrics(preds_tgt['pred_raw'], preds_tgt['target_raw'])['MAE'])
    return {
        'city_src_MAE_mean': float(np.mean(src_maes)),
        'city_src_MAE_std': float(np.std(src_maes)),
        'city_tgt_MAE_mean': float(np.mean(tgt_maes)),
        'city_tgt_MAE_std': float(np.std(tgt_maes)),
        'city_tgt_MAE_masked_mean': float(np.mean(tgt_maes_masked)),
        'city_tgt_MAE_masked_std': float(np.std(tgt_maes_masked)),
        'n_subsets': len(subsets),
    }


def summarize_results(results):
    rows = []
    for model_name, seeds in results.items():
        clean = [seed['clean']['clean_denorm_MAE'] for seed in seeds.values()]
        clean_masked = [seed['clean'].get('clean_denorm_MAE_masked', seed['clean']['clean_denorm_MAE']) for seed in seeds.values()]
        sensor = [seed['sensor_multimask']['delta_sensor_mean'] for seed in seeds.values()]
        frag = [seed['sensor_multimask']['relative_fragility'] for seed in seeds.values()]
        frag_masked = [seed['sensor_multimask'].get('relative_fragility_masked', seed['sensor_multimask']['relative_fragility']) for seed in seeds.values()]
        city = [seed['city']['city_src_MAE_mean'] for seed in seeds.values()]
        city_tgt_masked = [seed['city'].get('city_tgt_MAE_masked_mean', seed['city']['city_tgt_MAE_mean']) for seed in seeds.values()]
        rows.append({
            'model': model_name,
            'clean_denorm_MAE_mean': float(np.mean(clean)),
            'clean_denorm_MAE_std': float(np.std(clean)),
            'clean_denorm_MAE_masked_mean': float(np.mean(clean_masked)),
            'clean_denorm_MAE_masked_std': float(np.std(clean_masked)),
            'delta_sensor_mean': float(np.mean(sensor)),
            'delta_sensor_std': float(np.std(sensor)),
            'relative_fragility_mean': float(np.mean(frag)),
            'relative_fragility_std': float(np.std(frag)),
            'relative_fragility_masked_mean': float(np.mean(frag_masked)),
            'relative_fragility_masked_std': float(np.std(frag_masked)),
            'city_src_MAE_mean': float(np.mean(city)),
            'city_src_MAE_std': float(np.std(city)),
            'city_tgt_MAE_masked_mean': float(np.mean(city_tgt_masked)),
        })
    return rows


def main():
    args = parse_args()
    registry = load_registry(args.registry)
    device = torch.device(args.device)
    source = build_source_dataset(args.source_dataset)
    subsets = sample_target_subsets(args.target_dataset, source['train'].shape[1], n_subsets=args.n_subsets)

    results = {}
    for model_name in args.models:
        keys = sorted(k for k in registry if k.startswith(f'{model_name}__{args.source_dataset}__'))
        model_results = {}
        for key in keys:
            entry = registry[key]
            ckpt = torch.load(entry['checkpoint'], map_location=device, weights_only=False)
            model = build_model_from_config(model_name, ckpt['model_config'], device)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            scaler = StandardScaler(np.array(ckpt['scaler_mean'], dtype=np.float32), np.array(ckpt['scaler_std'], dtype=np.float32))

            x_test, y_test = build_arrays(model_name, source['test'], scaler)
            clean_loader = make_torch_loader(x_test, y_test, batch_size=64, shuffle=False)
            clean_preds = collect_predictions(model_name, model, clean_loader, scaler, device)
            clean_norm_mae = mae_tensor(clean_preds['pred_norm'], clean_preds['target_norm'])
            clean_denorm_mae = mae_tensor(clean_preds['pred_raw'], clean_preds['target_raw'])
            clean_denorm_metrics = compute_masked_metrics(clean_preds['pred_raw'], clean_preds['target_raw'])

            mask_results = [
                evaluate_sensor_dropout(model_name, model, x_test, y_test, scaler, device, args.dropout_ratio, seed)
                for seed in args.mask_seeds
            ]
            delta_list = [item['delta_sensor'] for item in mask_results]
            sensor_res = {
                'mask_seeds': args.mask_seeds,
                'per_mask': mask_results,
                'delta_sensor_mean': float(np.mean(delta_list)),
                'delta_sensor_std': float(np.std(delta_list)),
                'relative_fragility': float(np.mean(delta_list) / max(clean_denorm_mae, 1e-8)),
                'relative_fragility_masked': float(np.mean(delta_list) / max(clean_denorm_metrics['MAE'], 1e-8)),
            }

            city_res = evaluate_cross_city(model_name, model, scaler, subsets, device)
            city_res['delta_city_src'] = city_res['city_src_MAE_mean'] - clean_norm_mae

            model_results[key] = {
                'backbone': model_name,
                'seed': entry['seed'],
                'clean': {
                    'clean_norm_MAE': clean_norm_mae,
                    'clean_denorm_MAE': clean_denorm_mae,
                    'clean_denorm_MAE_masked': clean_denorm_metrics['MAE'],
                    'clean_denorm_RMSE_masked': clean_denorm_metrics['RMSE'],
                    'clean_denorm_MAPE_masked': clean_denorm_metrics['MAPE'],
                },
                'sensor_multimask': sensor_res,
                'city': city_res,
            }
        results[model_name] = model_results

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'official_adaptive_shift_eval.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    summary_rows = summarize_results(results)
    summary_path = output_dir / 'official_adaptive_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_rows, f, indent=2)

    print(json.dumps(summary_rows, indent=2))


if __name__ == '__main__':
    main()
