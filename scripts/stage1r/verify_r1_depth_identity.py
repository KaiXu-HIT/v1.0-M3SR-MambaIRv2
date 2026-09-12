#!/usr/bin/env python3
"""Stage1R R1-A: verify gate-zero depth adapter identity on metrics and tensors."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.archs.mambairv2_arch import MambaIRv2  # noqa: E402
from basicsr.archs.rgb_depth_residual_mambairv2_arch import (  # noqa: E402
    RGBDepthResidualMambaIRv2,
)
from basicsr.data.rgb_depth_paired_image_dataset import (  # noqa: E402
    RGBDepthPairedImageDataset,
)
from verify_r0_baseline import (  # noqa: E402
    REFERENCE,
    parse_metrics,
    resolve_log,
    sha256_file,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / 'options/test/mambairv2/test_S1R_R1_DepthIdentity_x4.yml')
DEFAULT_CHECKPOINT = Path(
    '/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/'
    'experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth')


def load_checkpoint(path, param_key):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    if param_key is not None:
        if param_key not in checkpoint:
            raise KeyError(
                f'Checkpoint has no parameter key {param_key!r}: {path}')
        checkpoint = checkpoint[param_key]
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint parameters must be a state-dict mapping.')
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in checkpoint.items()
    }


def build_models(network_options, state, device):
    identity_options = dict(network_options)
    identity_options.pop('type')
    baseline_options = dict(identity_options)
    baseline_options.pop('depth_in_chans')

    baseline = MambaIRv2(**baseline_options)
    identity = RGBDepthResidualMambaIRv2(**identity_options)
    baseline.load_state_dict(state, strict=True)
    incompatible = identity.load_state_dict(state, strict=False)
    expected_missing = identity.stage1r_expected_missing_keys()
    actual_missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if actual_missing != expected_missing or unexpected:
        raise RuntimeError(
            'Warm-start key audit failed. '
            f'Expected missing: {sorted(expected_missing)}; '
            f'actual missing: {sorted(actual_missing)}; '
            f'unexpected: {sorted(unexpected)}.')

    baseline_state = baseline.state_dict()
    identity_state = identity.state_dict()
    unequal_base_keys = [
        key for key, value in baseline_state.items()
        if key not in identity_state or not torch.equal(value, identity_state[key])]
    if unequal_base_keys:
        raise RuntimeError(
            'Loaded R1-A base tensors differ from v1.0: '
            f'{unequal_base_keys[:20]}')
    if identity.alpha_d.detach().item() != 0.0:
        raise RuntimeError(
            f'alpha_d must initialize to zero, got {identity.alpha_d.item()}.')

    baseline = baseline.to(device).eval()
    identity = identity.to(device).eval()
    return baseline, identity, sorted(actual_missing)


def aligned_tiles(rgb, depth):
    _, _, height, width = rgb.shape
    rows = height // 200 + 1
    cols = width // 200 + 1
    pad_h = (rows - height % rows) % rows
    pad_w = (cols - width % cols) % cols
    rgb = F.pad(rgb, (0, pad_w, 0, pad_h), 'reflect')
    depth = F.pad(depth, (0, pad_w, 0, pad_h), 'reflect')
    split_h = rgb.shape[-2] // rows
    split_w = rgb.shape[-1] // cols
    shave_h = split_h // 10
    shave_w = split_w // 10
    for row in range(rows):
        for col in range(cols):
            top_start = row * split_h if row == 0 else row * split_h - shave_h
            top_end = (
                (row + 1) * split_h if row == rows - 1
                else (row + 1) * split_h + shave_h)
            left_start = col * split_w if col == 0 else col * split_w - shave_w
            left_end = (
                (col + 1) * split_w if col == cols - 1
                else (col + 1) * split_w + shave_w)
            top = slice(top_start, top_end)
            left = slice(left_start, left_end)
            yield rgb[..., top, left], depth[..., top, left]


def build_sample_plan(config, samples_per_dataset, seed):
    plan = []
    for dataset_number, (_, options) in enumerate(
            sorted(config['datasets'].items())):
        dataset_options = copy.deepcopy(options)
        dataset_options['phase'] = 'test'
        dataset_options['scale'] = config['scale']
        dataset = RGBDepthPairedImageDataset(dataset_options)
        if len(dataset) < samples_per_dataset:
            raise ValueError(
                f'{dataset_options["name"]} has only {len(dataset)} samples; '
                f'cannot select {samples_per_dataset}.')
        indices = random.Random(seed + dataset_number).sample(
            range(len(dataset)), samples_per_dataset)
        plan.extend((dataset_options['name'], dataset, index) for index in indices)
    return plan


def compare_raw_outputs(baseline, identity, plan, device, raw_tolerance):
    records = []
    total_abs = 0.0
    total_values = 0
    overall_max = 0.0
    all_exact = True
    with torch.inference_mode():
        for dataset_name, dataset, index in plan:
            sample = dataset[index]
            rgb = sample['lq'].unsqueeze(0).to(device)
            depth = sample['depth'].unsqueeze(0).to(device)
            sample_abs = 0.0
            sample_values = 0
            sample_max = 0.0
            sample_exact = True
            tile_count = 0
            for rgb_tile, depth_tile in aligned_tiles(rgb, depth):
                baseline_output = baseline(rgb_tile)
                identity_output = identity(rgb_tile, depth_tile)
                difference = (identity_output - baseline_output).abs()
                sample_abs += difference.double().sum().item()
                sample_values += difference.numel()
                sample_max = max(sample_max, difference.max().item())
                sample_exact = sample_exact and torch.equal(
                    identity_output, baseline_output)
                tile_count += 1
            sample_mean = sample_abs / sample_values
            passed = sample_max <= raw_tolerance
            record = {
                'dataset': dataset_name,
                'index': index,
                'lq_path': sample['lq_path'],
                'depth_path': sample['depth_path'],
                'tiles': tile_count,
                'mean_abs_difference': sample_mean,
                'max_abs_difference': sample_max,
                'bitwise_equal': sample_exact,
                'passed': passed,
            }
            records.append(record)
            total_abs += sample_abs
            total_values += sample_values
            overall_max = max(overall_max, sample_max)
            all_exact = all_exact and sample_exact
            print(
                f'{dataset_name:<10} {Path(sample["lq_path"]).name:<24} '
                f'mean={sample_mean:.3e} max={sample_max:.3e} '
                f'exact={sample_exact} {"PASS" if passed else "FAIL"}')
    return {
        'samples': records,
        'sample_count': len(records),
        'mean_abs_difference': total_abs / total_values,
        'max_abs_difference': overall_max,
        'all_bitwise_equal': all_exact,
        'passed': overall_max <= raw_tolerance,
    }


def audit_metrics(log_path, psnr_tolerance, ssim_tolerance):
    observed = parse_metrics(log_path)
    report = {}
    all_passed = True
    print('Dataset    PSNR(ref/obs/delta)          SSIM(ref/obs/delta)        Result')
    print('-' * 88)
    for dataset, reference in REFERENCE.items():
        result = observed[dataset]
        psnr_delta = result['psnr'] - reference['psnr']
        ssim_delta = result['ssim'] - reference['ssim']
        passed = (
            abs(psnr_delta) < psnr_tolerance
            and abs(ssim_delta) < ssim_tolerance)
        all_passed = all_passed and passed
        report[dataset] = {
            'reference': reference,
            'observed': result,
            'delta': {'psnr': psnr_delta, 'ssim': ssim_delta},
            'passed': passed,
        }
        print(
            f'{dataset:<10} '
            f'{reference["psnr"]:7.4f}/{result["psnr"]:7.4f}/{psnr_delta:+8.4f}    '
            f'{reference["ssim"]:.4f}/{result["ssim"]:.4f}/{ssim_delta:+9.5f}    '
            f'{"PASS" if passed else "FAIL"}')
    return report, all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Verify Stage1R R1-A metric and raw-tensor identity.')
    parser.add_argument(
        'log_or_results_dir', type=Path,
        help='R1-A BasicSR test log or result directory.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--samples-per-dataset', type=int, default=2)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--raw-tol', type=float, default=1e-6)
    parser.add_argument('--psnr-tol', type=float, default=0.005)
    parser.add_argument('--ssim-tol', type=float, default=0.0002)
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()

    if args.samples_per_dataset < 2:
        parser.error('Use at least two images per dataset (10 images total).')
    if min(args.raw_tol, args.psnr_tol, args.ssim_tol) <= 0:
        parser.error('All tolerances must be positive.')
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f'Config does not exist: {config_path}')
    if not checkpoint_path.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint_path}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but torch.cuda.is_available() is false.')

    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    log_path = resolve_log(args.log_or_results_dir)
    state = load_checkpoint(checkpoint_path, config['path']['param_key_g'])
    device = torch.device(args.device)
    baseline, identity, missing_keys = build_models(
        config['network_g'], state, device)
    plan = build_sample_plan(
        config, args.samples_per_dataset, args.seed)

    checkpoint_hash = sha256_file(checkpoint_path)
    print(f'Checkpoint SHA-256: {checkpoint_hash}')
    print(f'alpha_d: {identity.alpha_d.item():.8f}')
    print(f'effective tanh(alpha_d): {identity.effective_alpha_d.item():.8f}')
    print(f'Expected/actual missing adapter keys: {missing_keys}')
    print()
    print('Raw tensor identity:')
    raw_report = compare_raw_outputs(
        baseline, identity, plan, device, args.raw_tol)
    print(
        f'RAW OVERALL: {"PASS" if raw_report["passed"] else "FAIL"}; '
        f'mean={raw_report["mean_abs_difference"]:.3e}; '
        f'max={raw_report["max_abs_difference"]:.3e}; '
        f'all_exact={raw_report["all_bitwise_equal"]}')
    print()
    print('Five-dataset metric identity:')
    metric_report, metrics_passed = audit_metrics(
        log_path, args.psnr_tol, args.ssim_tol)
    passed = raw_report['passed'] and metrics_passed
    print('-' * 88)
    print(f'R1-A OVERALL: {"PASS" if passed else "FAIL"}')

    report = {
        'experiment': 'Stage1R R1-A Depth Identity Adapter',
        'passed': passed,
        'config': str(config_path),
        'log': str(log_path),
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': checkpoint_hash,
        'alpha_d': identity.alpha_d.item(),
        'effective_alpha_d': identity.effective_alpha_d.item(),
        'missing_adapter_keys': missing_keys,
        'thresholds': {
            'raw_max_abs_lte': args.raw_tol,
            'abs_psnr_lt': args.psnr_tol,
            'abs_ssim_lt': args.ssim_tol,
        },
        'raw_identity': raw_report,
        'metrics': metric_report,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        print(f'JSON report: {args.json_out.resolve()}')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
