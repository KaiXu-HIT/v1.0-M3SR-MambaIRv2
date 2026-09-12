# Stage1R R1-A: Depth Identity Adapter

R0-A passed on 12 September 2026: all five PSNR/SSIM values exactly reproduced
the recorded v1.0 results with `net_g_490000.pth` (SHA-256
`05f7427a28e004f7cddd2f1c62ecdfd53d5221ca6e5dc9575fd90802ec9dbaa1`).
R1-A is therefore permitted.

This iteration implements only the gate-zero identity experiment. It does not
train the adapter and does not implement Depth Zero, Noise, Shuffle, or True.

## Baseline-preserving structure

```text
RGB -> v1.0 conv_first -> F_rgb ------------------\
                                                    + -> unchanged backbone -> SR
Depth -> Conv3x3 -> GELU -> Conv1x1 -> residual --/
                                      * tanh(alpha_d)
```

`alpha_d` is a scalar `nn.Parameter` initialized to exactly zero. Thus:

```text
F_in = F_rgb + tanh(0) * depth_residual = F_rgb
```

The v1.0 path is not replaced or reprojected. Loading is rejected unless every
v1.0 tensor matches and the only missing checkpoint keys are exactly:

```text
alpha_d
depth_adapter.0.weight
depth_adapter.0.bias
depth_adapter.2.weight
depth_adapter.2.bias
```

All code changes are marked `Stage1R R1-A`.

## Files

| File | Purpose |
|---|---|
| `basicsr/archs/rgb_depth_residual_mambairv2_arch.py` | Zero-gated residual Depth adapter while retaining the exact v1.0 RGB path |
| `basicsr/data/rgb_depth_paired_image_dataset.py` | Strict RGB/LR/depth pairing, scalar depth normalization, and aligned transforms |
| `basicsr/models/rgb_depth_residual_mambairv2_model.py` | Aligned tiled evaluation and strict v1.0 missing-key audit |
| `options/test/mambairv2/test_S1R_R1_DepthIdentity_x4.yml` | Five-dataset R1-A test using the 490k checkpoint |
| `scripts/stage1r/verify_r1_depth_identity.py` | Ten-image raw-tensor comparison plus five-dataset metric audit |

## Run on the server

Update the repository and enter it:

```bash
cd /home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2
git pull origin main
```

Confirm the fixed checkpoint:

```bash
sha256sum experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth
```

Expected SHA-256:

```text
05f7427a28e004f7cddd2f1c62ecdfd53d5221ca6e5dc9575fd90802ec9dbaa1
```

Run the untrained gate-zero adapter on all five datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
  -opt options/test/mambairv2/test_S1R_R1_DepthIdentity_x4.yml \
  --launcher none
```

The loading log must contain:

```text
Stage1R R1-A checkpoint audit passed
```

It must state that the five missing keys are exactly the new adapter keys and
that there are no unexpected keys.

Then run the stricter audit. It deterministically samples two images from each
benchmark (10 total), compares raw float tensors using aligned tiles, and also
checks the five logged metrics:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage1r/verify_r1_depth_identity.py \
  results/test_S1R_R1_DepthIdentity_x4 \
  --checkpoint experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth \
  --samples-per-dataset 2 \
  --raw-tol 1e-6 \
  --json-out results/test_S1R_R1_DepthIdentity_x4/r1_a_identity_audit.json
```

## Pass criteria

The experiment passes only when all conditions hold:

```text
alpha_d = 0
effective tanh(alpha_d) = 0
only the five adapter keys are missing from the v1.0 checkpoint
raw max absolute output difference <= 1e-6 on all 10 sampled images
each dataset |Delta PSNR| < 0.005 dB
each dataset |Delta SSIM| < 0.0002
R1-A OVERALL: PASS
```

Ideally the raw output reports `all_exact=True` and zero mean/max difference.
Do not continue if R1-A fails. Return the complete verifier output,
`r1_a_identity_audit.json`, and the BasicSR test log before the next experiment.
