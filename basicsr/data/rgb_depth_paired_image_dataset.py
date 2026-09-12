import os.path as osp

import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor, imfrombytes, scandir
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


def _stage1r_root_list(value, option_name):
    roots = value if isinstance(value, list) else [value]
    if not roots or any(not isinstance(root, str) or not root for root in roots):
        raise ValueError(f'{option_name} must contain one or more non-empty paths.')
    return roots


@DATASET_REGISTRY.register()
class RGBDepthPairedImageDataset(data.Dataset):
    """Stage1R R1-A: strictly aligned LR RGB/depth and HR RGB dataset.

    Depth is scalarized and robustly normalized on the complete LR image.
    Whenever training is introduced in a later experiment, RGB, depth, and GT
    will share the same crop and spatial augmentation coordinates.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = dict(opt['io_backend'])
        if self.io_backend_opt.get('type') != 'disk':
            raise NotImplementedError(
                'RGBDepthPairedImageDataset currently supports disk folders only.')

        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.filename_tmpl_depth = opt.get(
            'filename_tmpl_depth', self.filename_tmpl)
        self.depth_percentile_low = float(
            opt.get('depth_percentile_low', 0.02))
        self.depth_percentile_high = float(
            opt.get('depth_percentile_high', 0.98))
        self.depth_normalize_eps = float(
            opt.get('depth_normalize_eps', 1e-6))

        if not 0.0 <= self.depth_percentile_low < self.depth_percentile_high <= 1.0:
            raise ValueError('Depth percentiles must satisfy 0 <= low < high <= 1.')
        if self.depth_normalize_eps <= 0.0:
            raise ValueError('depth_normalize_eps must be positive.')

        gt_roots = _stage1r_root_list(opt['dataroot_gt'], 'dataroot_gt')
        lq_roots = _stage1r_root_list(opt['dataroot_lq'], 'dataroot_lq')
        depth_roots = _stage1r_root_list(
            opt['dataroot_depth'], 'dataroot_depth')
        if not (len(gt_roots) == len(lq_roots) == len(depth_roots)):
            raise ValueError(
                'dataroot_gt, dataroot_lq, and dataroot_depth must have equal lengths.')

        self.paths = []
        for gt_root, lq_root, depth_root in zip(
                gt_roots, lq_roots, depth_roots):
            lq_names = set(scandir(lq_root))
            depth_names = set(scandir(depth_root))
            for gt_name in sorted(scandir(gt_root)):
                basename, extension = osp.splitext(osp.basename(gt_name))
                lq_name = f'{self.filename_tmpl.format(basename)}{extension}'
                depth_name = f'{self.filename_tmpl_depth.format(basename)}{extension}'
                if lq_name not in lq_names:
                    raise FileNotFoundError(
                        f'Missing paired LR RGB image: {osp.join(lq_root, lq_name)}')
                if depth_name not in depth_names:
                    raise FileNotFoundError(
                        f'Missing paired depth image: {osp.join(depth_root, depth_name)}')
                self.paths.append({
                    'gt_path': osp.join(gt_root, gt_name),
                    'lq_path': osp.join(lq_root, lq_name),
                    'depth_path': osp.join(depth_root, depth_name),
                })

        if not self.paths:
            raise ValueError('RGBDepthPairedImageDataset found no paired samples.')

    @staticmethod
    def _to_scalar_depth(depth):
        if depth.ndim == 2:
            return depth[..., None]
        if depth.ndim != 3:
            raise ValueError(f'Depth must be HW or HWC, but got shape {depth.shape}.')
        if depth.shape[2] == 1:
            return depth
        return np.mean(depth, axis=2, keepdims=True, dtype=np.float32)

    def _normalize_depth(self, depth):
        depth = self._to_scalar_depth(depth).astype(np.float32, copy=False)
        finite = np.isfinite(depth)
        if not finite.all():
            depth = depth.copy()
            depth[~finite] = 0.0
        low, high = np.quantile(
            depth, [self.depth_percentile_low, self.depth_percentile_high])
        depth = (depth - low) / (high - low + self.depth_normalize_eps)
        return np.clip(depth, 0.0, 1.0).astype(np.float32, copy=False)

    def __getitem__(self, index):
        if self.file_client is None:
            backend_type = self.io_backend_opt.pop('type')
            self.file_client = FileClient(backend_type, **self.io_backend_opt)

        paths = self.paths[index]
        img_gt = imfrombytes(
            self.file_client.get(paths['gt_path'], 'gt'), float32=True)
        img_lq = imfrombytes(
            self.file_client.get(paths['lq_path'], 'lq'), float32=True)
        raw_depth = imfrombytes(
            self.file_client.get(paths['depth_path'], 'depth'),
            flag='unchanged', float32=False)
        if raw_depth is None:
            raise ValueError(f'Failed to decode depth image: {paths["depth_path"]}')
        img_depth = self._normalize_depth(raw_depth)

        if img_lq.shape[:2] != img_depth.shape[:2]:
            raise ValueError(
                'Stage1R requires aligned LR RGB and depth images, but got '
                f'{img_lq.shape[:2]} for {paths["lq_path"]} and '
                f'{img_depth.shape[:2]} for {paths["depth_path"]}.')

        scale = self.opt['scale']
        if self.opt['phase'] == 'train':
            img_gt, (img_lq, img_depth) = paired_random_crop(
                img_gt, [img_lq, img_depth], self.opt['gt_size'],
                scale, paths['gt_path'])
            img_gt, img_lq, img_depth = augment(
                [img_gt, img_lq, img_depth],
                self.opt['use_hflip'], self.opt['use_rot'])

        if self.opt.get('color') == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[
                :img_lq.shape[0] * scale, :img_lq.shape[1] * scale, :]

        img_gt, img_lq, img_depth = img2tensor(
            [img_gt, img_lq, img_depth], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'depth': img_depth,
            'gt': img_gt,
            'lq_path': paths['lq_path'],
            'depth_path': paths['depth_path'],
            'gt_path': paths['gt_path'],
        }

    def __len__(self):
        return len(self.paths)
