# Stage1R R0-A: v1.0 Baseline Reproduction

This is the first and only experiment implemented in this iteration. It does
not change MambaIRv2, the dataset pipeline, or any metric. It re-evaluates the
existing v1.0 500k checkpoint in the Stage1R environment.

## Added files

| File | R0-A purpose |
|---|---|
| `options/test/mambairv2/test_S1R_R0_RGBBaseline_x4.yml` | Exact five-dataset v1.0 evaluation with the user-confirmed checkpoint path |
| `scripts/stage1r/verify_r0_baseline.py` | Parse the BasicSR log, hash the checkpoint, and apply the Stage1R pass thresholds |

Every addition is marked `Stage1R R0-A`. No later Depth or Text experiment is
included.

## Reference and pass criteria

| Dataset | Reference PSNR | Reference SSIM |
|---|---:|---:|
| Set5 | 32.7645 | 0.9029 |
| Set14 | 29.0786 | 0.7928 |
| B100 | 27.8639 | 0.7468 |
| Urban100 | 27.3198 | 0.8215 |
| Manga109 | 31.8458 | 0.9239 |

Every dataset must satisfy both:

```text
absolute PSNR difference < 0.005 dB
absolute SSIM difference < 0.0002
```

Do not continue to R1-A if R0-A fails.

## Server procedure

Enter the v1.0 project and confirm the checkpoint exists:

```bash
cd /home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2

test -f experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_500000.pth \
  && echo "checkpoint found" \
  || echo "checkpoint missing"
```

Record the software/GPU environment in the terminal output:

```bash
python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cudnn', torch.backends.cudnn.version())"
nvidia-smi
```

Run the unchanged v1.0 model on all five benchmarks:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
  -opt options/test/mambairv2/test_S1R_R0_RGBBaseline_x4.yml \
  --launcher none
```

BasicSR writes the log under:

```text
results/test_S1R_R0_RGBBaseline_x4/test_*.log
```

Automatically compare the latest log with the recorded metrics and save a JSON
audit record:

```bash
python scripts/stage1r/verify_r0_baseline.py \
  results/test_S1R_R0_RGBBaseline_x4 \
  --checkpoint experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_500000.pth \
  --json-out results/test_S1R_R0_RGBBaseline_x4/r0_a_audit.json
```

The verifier exits with code `0` only when all five datasets pass. A metric at
exactly the tolerance boundary fails because the protocol requires `<`, not
`<=`.

## What to return before R1-A

Send back:

1. the verifier's complete table and `R0-A OVERALL` line;
2. `r0_a_audit.json`;
3. the Python/PyTorch/CUDA/cuDNN and `nvidia-smi` output;
4. the BasicSR test log if any dataset fails.

For a failure, first check checkpoint SHA-256, dataset roots and file counts,
`crop_border: 4`, `test_y_channel: true`, image reading/color conversion, and
the PyTorch/CUDA/library versions. Do not adjust thresholds to force a pass.
