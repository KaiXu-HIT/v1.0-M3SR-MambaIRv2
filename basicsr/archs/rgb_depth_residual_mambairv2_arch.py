import torch
from torch import nn

from basicsr.archs.mambairv2_arch import MambaIRv2
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RGBDepthResidualMambaIRv2(MambaIRv2):
    """Stage1R R1-A: baseline-preserving depth identity adapter.

    The complete v1.0 RGB path is retained. Depth contributes only through
    ``F_rgb + tanh(alpha_d) * G_d(depth)`` and ``alpha_d`` is initialized to
    exactly zero. Before training, the network must therefore reproduce v1.0.
    """

    def __init__(self, depth_in_chans=1, **kwargs):
        super().__init__(**kwargs)
        self.depth_in_chans = int(depth_in_chans)
        if self.depth_in_chans <= 0:
            raise ValueError('depth_in_chans must be positive.')

        # Stage1R R1-A change: this minimal branch never replaces F_rgb. Its
        # residual is added only after multiplication by the zero-initialized
        # scalar gate, preserving the loaded v1.0 function at initialization.
        self.depth_adapter = nn.Sequential(
            nn.Conv2d(self.depth_in_chans, self.embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 1, 1))
        self.alpha_d = nn.Parameter(torch.zeros(1))

    @staticmethod
    def stage1r_expected_missing_keys():
        """Keys intentionally absent when warm-starting from a v1.0 checkpoint."""
        return {
            'alpha_d',
            'depth_adapter.0.weight',
            'depth_adapter.0.bias',
            'depth_adapter.2.weight',
            'depth_adapter.2.bias',
        }

    @property
    def effective_alpha_d(self):
        """Bound the learned gate while keeping alpha=0 exactly transparent."""
        return torch.tanh(self.alpha_d)

    @staticmethod
    def _mirror_pad_to_size(x, target_h, target_w):
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :target_h, :]
        return torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :target_w]

    def forward(self, rgb, depth):
        if rgb.ndim != 4 or depth.ndim != 4:
            raise ValueError('RGB and depth inputs must both be BCHW tensors.')
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError(
                f'RGB/depth batch or spatial mismatch: {rgb.shape} vs {depth.shape}.')
        if depth.shape[1] != self.depth_in_chans:
            raise ValueError(
                f'Expected {self.depth_in_chans} depth channel(s), got {depth.shape[1]}.')

        # Stage1R R1-A: reproduce the v1.0 padding and normalization order
        # exactly; depth is padded to the same size but never alters RGB here.
        h_ori, w_ori = rgb.size()[-2], rgb.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        rgb = self._mirror_pad_to_size(rgb, h, w)
        depth = self._mirror_pad_to_size(depth, h, w)

        self.mean = self.mean.type_as(rgb)
        rgb = (rgb - self.mean) * self.img_range
        depth = depth.type_as(rgb)

        attn_mask = self.calculate_mask([h, w]).to(rgb.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            rgb_feature = self.conv_first(rgb)
            depth_residual = self.depth_adapter(depth)
            feature = rgb_feature + self.effective_alpha_d * depth_residual
            feature = self.conv_after_body(
                self.forward_features(feature, params)) + feature
            feature = self.conv_before_upsample(feature)
            output = self.conv_last(self.upsample(feature))
        elif self.upsampler == 'pixelshuffledirect':
            rgb_feature = self.conv_first(rgb)
            depth_residual = self.depth_adapter(depth)
            feature = rgb_feature + self.effective_alpha_d * depth_residual
            feature = self.conv_after_body(
                self.forward_features(feature, params)) + feature
            output = self.upsample(feature)
        elif self.upsampler == 'nearest+conv':
            rgb_feature = self.conv_first(rgb)
            depth_residual = self.depth_adapter(depth)
            feature = rgb_feature + self.effective_alpha_d * depth_residual
            feature = self.conv_after_body(
                self.forward_features(feature, params)) + feature
            feature = self.conv_before_upsample(feature)
            feature = self.lrelu(self.conv_up1(
                torch.nn.functional.interpolate(feature, scale_factor=2, mode='nearest')))
            feature = self.lrelu(self.conv_up2(
                torch.nn.functional.interpolate(feature, scale_factor=2, mode='nearest')))
            output = self.conv_last(self.lrelu(self.conv_hr(feature)))
        else:
            rgb_feature = self.conv_first(rgb)
            depth_residual = self.depth_adapter(depth)
            feature = rgb_feature + self.effective_alpha_d * depth_residual
            residual = self.conv_after_body(
                self.forward_features(feature, params)) + feature
            output = rgb + self.conv_last(residual)

        output = output / self.img_range + self.mean
        return output[..., :h_ori * self.upscale, :w_ori * self.upscale]
