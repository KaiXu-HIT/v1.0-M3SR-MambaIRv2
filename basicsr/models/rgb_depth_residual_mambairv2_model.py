import torch
from torch.nn import functional as F

from basicsr.models.mambairv2_model import MambaIRv2Model
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class RGBDepthResidualMambaIRv2Model(MambaIRv2Model):
    """Stage1R R1-A wrapper for aligned RGB/depth identity evaluation."""

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Fail unless a v1.0 warm-start misses only the new R1-A adapter."""
        bare_net = self.get_bare_model(crt_net)
        if not strict and hasattr(bare_net, 'stage1r_expected_missing_keys'):
            current_state = bare_net.state_dict()
            current_keys = set(current_state)
            loaded_keys = set(load_net)
            missing = current_keys - loaded_keys
            unexpected = loaded_keys - current_keys
            shape_mismatches = {
                key: (tuple(current_state[key].shape), tuple(load_net[key].shape))
                for key in current_keys & loaded_keys
                if current_state[key].shape != load_net[key].shape
            }
            expected = set(bare_net.stage1r_expected_missing_keys())
            if missing != expected or unexpected or shape_mismatches:
                raise RuntimeError(
                    'Stage1R R1-A rejected the checkpoint. '
                    f'Expected missing adapter keys: {sorted(expected)}; '
                    f'actual missing: {sorted(missing)}; '
                    f'unexpected: {sorted(unexpected)}; '
                    f'shape mismatches: {shape_mismatches}.')
            get_root_logger().info(
                'Stage1R R1-A checkpoint audit passed. Missing keys are '
                f'exactly the new adapter: {sorted(missing)}; unexpected keys: none.')
            return
        super()._print_different_keys_loading(crt_net, load_net, strict)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.depth = data['depth'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def test(self):
        """Use identical aligned RGB/depth tiles and preserve raw output dtype."""
        _, channels, h, w = self.lq.size()
        split_token_h = h // 200 + 1
        split_token_w = w // 200 + 1
        mod_pad_h = (split_token_h - h % split_token_h) % split_token_h
        mod_pad_w = (split_token_w - w % split_token_w) % split_token_w
        rgb = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        depth = F.pad(self.depth, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, padded_h, padded_w = rgb.size()
        split_h = padded_h // split_token_h
        split_w = padded_w // split_token_w
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)

        slices = []
        for i in range(split_token_h):
            for j in range(split_token_w):
                top_start = i * split_h if i == 0 else i * split_h - shave_h
                top_end = (
                    (i + 1) * split_h if i == split_token_h - 1
                    else (i + 1) * split_h + shave_h)
                left_start = j * split_w if j == 0 else j * split_w - shave_w
                left_end = (
                    (j + 1) * split_w if j == split_token_w - 1
                    else (j + 1) * split_w + shave_w)
                slices.append((
                    slice(top_start, top_end), slice(left_start, left_end)))

        test_net = (
            self.net_g_ema if hasattr(self, 'net_g_ema')
            else self.get_bare_model(self.net_g))
        was_training = test_net.training
        test_net.eval()
        with torch.no_grad():
            outputs = [
                test_net(rgb[..., top, left], depth[..., top, left])
                for top, left in slices]
            merged = torch.zeros(
                (rgb.shape[0], channels, padded_h * scale, padded_w * scale),
                device=outputs[0].device, dtype=outputs[0].dtype)
            for i in range(split_token_h):
                for j in range(split_token_w):
                    top = slice(
                        i * split_h * scale, (i + 1) * split_h * scale)
                    left = slice(
                        j * split_w * scale, (j + 1) * split_w * scale)
                    source_top = (
                        slice(0, split_h * scale) if i == 0
                        else slice(shave_h * scale,
                                   (shave_h + split_h) * scale))
                    source_left = (
                        slice(0, split_w * scale) if j == 0
                        else slice(shave_w * scale,
                                   (shave_w + split_w) * scale))
                    merged[..., top, left] = outputs[
                        i * split_token_w + j][..., source_top, source_left]
        if was_training:
            test_net.train()
        self.output = merged[..., :h * scale, :w * scale]

    def get_current_visuals(self):
        visuals = super().get_current_visuals()
        visuals['depth'] = self.depth.detach().cpu()
        return visuals
