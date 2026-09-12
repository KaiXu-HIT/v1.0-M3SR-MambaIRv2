#!/usr/bin/env python3
"""Stage1R R0-A: compare a BasicSR test log with the recorded v1.0 metrics."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


# Stage1R R0-A reference values supplied in the experiment protocol.
REFERENCE = {
    'Set5': {'psnr': 32.7645, 'ssim': 0.9029},
    'Set14': {'psnr': 29.0786, 'ssim': 0.7928},
    'B100': {'psnr': 27.8639, 'ssim': 0.7468},
    'Urban100': {'psnr': 27.3198, 'ssim': 0.8215},
    'Manga109': {'psnr': 31.8458, 'ssim': 0.9239},
}

DEFAULT_CHECKPOINT = Path(
    '/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/'
    'experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_500000.pth')

METRIC_PATTERN = re.compile(
    r'Validation\s+(Set5|Set14|B100|Urban100|Manga109)\s*\r?\n'
    r'\s*#\s*psnr:\s*([-+0-9.eE]+)[^\r\n]*\r?\n'
    r'\s*#\s*ssim:\s*([-+0-9.eE]+)',
    re.MULTILINE)


def resolve_log(path):
    """Accept either one log file or the BasicSR result directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f'Log file/directory does not exist: {path}')
    candidates = list(path.glob('test_*.log'))
    if not candidates:
        candidates = list(path.rglob('test_*.log'))
    if not candidates:
        raise FileNotFoundError(f'No test_*.log found under: {path}')
    return max(candidates, key=lambda item: item.stat().st_mtime)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_metrics(log_path):
    text = log_path.read_text(encoding='utf-8', errors='replace')
    metrics = {}
    for dataset, psnr, ssim in METRIC_PATTERN.findall(text):
        # If a log contains repeated evaluations, keep the final occurrence.
        metrics[dataset] = {'psnr': float(psnr), 'ssim': float(ssim)}
    missing = [name for name in REFERENCE if name not in metrics]
    if missing:
        raise ValueError(
            'Missing validation metrics for: '
            f'{", ".join(missing)}. Parsed log: {log_path}')
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Audit Stage1R R0-A results against recorded v1.0 metrics.')
    parser.add_argument(
        'log_or_results_dir', type=Path,
        help='BasicSR test log or results/test_S1R_R0_RGBBaseline_x4 directory.')
    parser.add_argument(
        '--checkpoint', type=Path, default=DEFAULT_CHECKPOINT,
        help='Exact checkpoint used by the test; its SHA-256 is recorded.')
    parser.add_argument('--psnr-tol', type=float, default=0.005)
    parser.add_argument('--ssim-tol', type=float, default=0.0002)
    parser.add_argument(
        '--json-out', type=Path,
        help='Optional path for a machine-readable audit report.')
    args = parser.parse_args()

    if args.psnr_tol <= 0 or args.ssim_tol <= 0:
        parser.error('Tolerances must be positive.')
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint}')

    try:
        log_path = resolve_log(args.log_or_results_dir)
        observed = parse_metrics(log_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    checkpoint_hash = sha256_file(checkpoint)
    report = {
        'experiment': 'Stage1R R0-A v1.0 baseline reproduction',
        'log': str(log_path),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash,
        'thresholds': {
            'abs_psnr_lt': args.psnr_tol,
            'abs_ssim_lt': args.ssim_tol,
        },
        'datasets': {},
    }

    print(f'Log: {log_path}')
    print(f'Checkpoint: {checkpoint}')
    print(f'Checkpoint SHA-256: {checkpoint_hash}')
    print()
    print('Dataset    PSNR(ref/obs/delta)          SSIM(ref/obs/delta)        Result')
    print('-' * 88)
    all_passed = True
    for dataset, reference in REFERENCE.items():
        result = observed[dataset]
        psnr_delta = result['psnr'] - reference['psnr']
        ssim_delta = result['ssim'] - reference['ssim']
        passed = (
            abs(psnr_delta) < args.psnr_tol
            and abs(ssim_delta) < args.ssim_tol)
        all_passed = all_passed and passed
        report['datasets'][dataset] = {
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

    report['passed'] = all_passed
    print('-' * 88)
    print(f'R0-A OVERALL: {"PASS" if all_passed else "FAIL"}')

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        print(f'JSON report: {args.json_out.resolve()}')

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
