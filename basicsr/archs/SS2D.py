import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numbers
from functools import partial
from typing import Optional, Callable
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.modules.mamba_simple import Mamba    # 这里容易报错
from einops import rearrange, repeat
from basicsr.archs import LSConv
from math import gcd


# 原始SS2D
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,         # 时间步长的最大和最小值
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)                                    # 计算内部维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank     # dt_rank自动设置为d_model/16的向上取整

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)    # 将输入投影到2倍维度，用于后续分割为x和z
        self.conv2d = nn.Conv2d(                                                                 # 深度可分离卷积DWConv，用于局部特征提取
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # LSConv:用LSConv替换原有模型中的DWConv
        self.lsconv = LSConv.LSConv(self.d_inner)
        self.act = nn.SiLU()

        self.x_proj = (                  # 四个线性层（对应4个扫描方向），用于从x生成dt，B，C，这里表示为元组，后面会合并为权重矩阵
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj                  # 将x_proj的权重合并为x_proj_weight，并删除x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)      #状态矩阵A的对数，
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)                            # 参数矩阵D

        self.selective_scan = selective_scan_fn                     # 选择性扫描函数

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod              # A_log_init静态方法，初始化A_log，使用实数初始化，A为1到d_state的序列，然后取对数
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod                  # 初始化D矩阵参数，全为1
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        # 输入x: (B, C, H, W)；将x重排为四个方向（上下左右和翻转）并拼接，得到xs: (B, 4, d_inner, L) 其中L=H*W
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 使用einsum进行批量矩阵乘法，通过x_proj_weight将xs投影得到x_dbl，然后分割为dts, Bs, Cs
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # 将Δt从秩空间投影到特征空间
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # 将xs, dts, Bs, Cs, As, Ds等参数传递给selective_scan函数，核心的mamba扫描操作
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # 得到四个方向的输出，然后合并为y1, y2, y3, y4，将四个方向的输出重新排列回原始的空间布局
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        # 输入x: (B, H, W, C)；通过in_proj得到xz，然后分割为x和z；通过in_proj得到xz，然后分割为x和z
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))
        # x = self.act(self.lsconv(x))          # 用新模块LSConv替换原模型中的DWConv

        # 通过forward_core四个方向扫描得到四个输出，并求和得到y
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        # 将y置换回(B, H, W, d_inner)，并通过层归一化；将y置换回(B, H, W, d_inner)，并通过层归一化；通过输出投影层和dropout（如果有）
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# CLC-Scan
class SS2D_CLCScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,  # 时间步长的最大和最小值
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 计算内部维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # dt_rank自动设置为d_model/16的向上取整

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)  # 将输入投影到2倍维度，用于后续分割为x和z
        self.conv2d = nn.Conv2d(  # 深度可分离卷积DWConv，用于局部特征提取
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # LSConv:用LSConv替换原有模型中的DWConv
        self.lsconv = LSConv.LSConv(self.d_inner)
        self.act = nn.SiLU()

        self.x_proj = (  # 四个线性层（对应4个扫描方向），用于从x生成dt，B，C，这里表示为元组，后面会合并为权重矩阵
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj  # 将x_proj的权重合并为x_proj_weight，并删除x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)      #状态矩阵A的对数，
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)                            # 参数矩阵D

        self.selective_scan = selective_scan_fn  # 选择性扫描函数

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod  # A_log_init静态方法，初始化A_log，使用实数初始化，A为1到d_state的序列，然后取对数
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod  # 初始化D矩阵参数，全为1
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def _forward_core_ss2d_reference(self, x: torch.Tensor):
        # 输入x: (B, C, H, W)；将x重排为四个方向（上下左右和翻转）并拼接，得到xs: (B, 4, d_inner, L) 其中L=H*W
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        # 使用einsum进行批量矩阵乘法，通过x_proj_weight将xs投影得到x_dbl，然后分割为dts, Bs, Cs
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # 将Δt从秩空间投影到特征空间
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # 将xs, dts, Bs, Cs, As, Ds等参数传递给selective_scan函数，核心的mamba扫描操作
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # 得到四个方向的输出，然后合并为y1, y2, y3, y4，将四个方向的输出重新排列回原始的空间布局
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    @staticmethod
    def _clc_local_candidates(top, left, bottom, right, W):
        """Build a small dictionary of cache-friendly curves for one tile."""
        rows = list(range(top, bottom))
        cols = list(range(left, right))
        candidates = []

        row_curve = []
        for i, r in enumerate(rows):
            local_cols = cols if i % 2 == 0 else cols[::-1]
            row_curve.extend(r * W + c for c in local_cols)
        candidates.extend([row_curve, row_curve[::-1]])

        col_curve = []
        for i, c in enumerate(cols):
            local_rows = rows if i % 2 == 0 else rows[::-1]
            col_curve.extend(r * W + c for r in local_rows)
        candidates.extend([col_curve, col_curve[::-1]])

        row_curve_flip = []
        for i, r in enumerate(rows[::-1]):
            local_cols = cols if i % 2 == 0 else cols[::-1]
            row_curve_flip.extend(r * W + c for c in local_cols)
        col_curve_flip = []
        for i, c in enumerate(cols[::-1]):
            local_rows = rows if i % 2 == 0 else rows[::-1]
            col_curve_flip.extend(r * W + c for r in local_rows)
        candidates.extend([
            row_curve_flip, row_curve_flip[::-1],
            col_curve_flip, col_curve_flip[::-1],
        ])
        return candidates

    @classmethod
    def _clc_path(cls, H, W, tile_size, vertical_first=False, reverse_tiles=False):
        """Build one CLC path by stitching the closest tile entry and exit."""
        n_th = (H + tile_size - 1) // tile_size
        n_tw = (W + tile_size - 1) // tile_size
        tiles = []

        if vertical_first:
            for tc in range(n_tw):
                tile_rows = range(n_th) if tc % 2 == 0 else range(n_th - 1, -1, -1)
                tiles.extend((tr, tc) for tr in tile_rows)
        else:
            for tr in range(n_th):
                tile_cols = range(n_tw) if tr % 2 == 0 else range(n_tw - 1, -1, -1)
                tiles.extend((tr, tc) for tc in tile_cols)
        if reverse_tiles:
            tiles.reverse()

        path = []
        previous_exit = None
        for tr, tc in tiles:
            top, left = tr * tile_size, tc * tile_size
            bottom, right = min(top + tile_size, H), min(left + tile_size, W)
            candidates = cls._clc_local_candidates(top, left, bottom, right, W)

            if previous_exit is None:
                corner = (H - 1 if reverse_tiles else 0) * W
                corner += W - 1 if vertical_first else 0
                cr, cc = divmod(corner, W)
                best = min(candidates, key=lambda p: abs(p[0] // W - cr) + abs(p[0] % W - cc))
            else:
                pr, pc = divmod(previous_exit, W)
                best = min(candidates, key=lambda p: abs(p[0] // W - pr) + abs(p[0] % W - pc))

            path.extend(best)
            previous_exit = best[-1]
        return path

    @classmethod
    def _clc_scan_indices(cls, H, W, device, tile_size=8):
        """Return four CLC permutations: horizontal/vertical and reverse variants."""
        paths = [
            cls._clc_path(H, W, tile_size, vertical_first=False, reverse_tiles=False),
            cls._clc_path(H, W, tile_size, vertical_first=True, reverse_tiles=False),
            cls._clc_path(H, W, tile_size, vertical_first=False, reverse_tiles=True),
            cls._clc_path(H, W, tile_size, vertical_first=True, reverse_tiles=True),
        ]
        return torch.tensor(paths, dtype=torch.long, device=device)

    def forward_core(self, x: torch.Tensor):
        # Only the original SS2D scan and merge operations are replaced by CLC.
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # Cache the deterministic permutation to avoid rebuilding it every forward.
        cache_key = (H, W, x.device.type, x.device.index)
        clc_cache = getattr(self, "_clc_indices_cache", None)
        if clc_cache is None:
            clc_cache = {}
            self._clc_indices_cache = clc_cache
        if cache_key not in clc_cache:
            clc_cache[cache_key] = self._clc_scan_indices(H, W, x.device)
        clc_indices = clc_cache[cache_key]
        x_flat = x.contiguous().view(B, C, L)
        xs = torch.gather(
            x_flat.unsqueeze(1).expand(-1, K, -1, -1),
            dim=-1,
            index=clc_indices.view(1, K, 1, L).expand(B, -1, C, -1),
        )

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        restored = torch.empty_like(out_y)
        restored.scatter_(
            dim=-1,
            index=clc_indices.view(1, K, 1, L).expand(B, -1, out_y.shape[2], -1),
            src=out_y,
        )
        return restored[:, 0], restored[:, 1], restored[:, 2], restored[:, 3]

    def forward(self, x: torch.Tensor, **kwargs):
        # 输入x: (B, H, W, C)；通过in_proj得到xz，然后分割为x和z；通过in_proj得到xz，然后分割为x和z
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))
        # x = self.act(self.lsconv(x))  # 用新模块LSConv替换原模型中的DWConv

        # 通过forward_core四个方向扫描得到四个输出，并求和得到y
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        # 将y置换回(B, H, W, d_inner)，并通过层归一化；将y置换回(B, H, W, d_inner)，并通过层归一化；通过输出投影层和dropout（如果有）
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# Retina Log-Polar Multi-Fovea Scan
class SS2D_RLPMFScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.lsconv = LSConv.LSConv(self.d_inner)
        self.act = nn.SiLU()

        # K=2：正向 Retina Log-Polar scan + 逆向 Retina Log-Polar scan
        self.K = 2

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self._scan_cache = {}

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)

        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)

        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)

        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _build_retina_logpolar_index(self, H, W, device):
        """
        生成 Retina Log-Polar Multi-Fovea scan 的一维索引。
        规则：
        1. 设置中心焦点 + 四个周边焦点；
        2. 每个 patch 分配给最近焦点；
        3. 每个焦点内部按照 log(radius) + angle 排序；
        4. 焦点顺序：中心 -> 左上 -> 右上 -> 左下 -> 右下。
        """
        key = ("retina_logpolar", H, W, device)
        if key in self._scan_cache:
            return self._scan_cache[key]

        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        centers = [
            (H // 2, W // 2),
            (H // 4, W // 4),
            (H // 4, max(0, 3 * W // 4)),
            (max(0, 3 * H // 4), W // 4),
            (max(0, 3 * H // 4), max(0, 3 * W // 4)),
        ]

        coords = []
        for i in range(H):
            for j in range(W):
                min_dist = None
                center_id = 0
                for k, (cy, cx) in enumerate(centers):
                    dist = (i - cy) ** 2 + (j - cx) ** 2
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        center_id = k

                cy, cx = centers[center_id]
                dy = i - cy
                dx = j - cx
                radius = math.sqrt(float(dy * dy + dx * dx))
                log_radius = math.log2(radius + 1.0)
                angle = math.atan2(float(dy), float(dx))

                coords.append((center_id, log_radius, angle, i, j))

        coords = sorted(coords, key=lambda t: (t[0], t[1], t[2]))
        index = torch.tensor([i * W + j for _, _, _, i, j in coords], dtype=torch.long, device=device)

        self._scan_cache[key] = index
        return index

    @staticmethod
    def _gather_by_index(x, index):
        B, C, H, W = x.shape
        L = H * W
        x_flat = x.view(B, C, L)
        index = index.view(1, 1, L).expand(B, C, L)
        return torch.gather(x_flat, dim=2, index=index)

    @staticmethod
    def _scatter_by_index(y, index, H, W):
        B, C, L = y.shape
        out = torch.zeros(B, C, L, device=y.device, dtype=y.dtype)
        index = index.view(1, 1, L).expand(B, C, L)
        out.scatter_(dim=2, index=index, src=y)
        return out

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = self.K

        forward_index = self._build_retina_logpolar_index(H, W, x.device)
        backward_index = torch.flip(forward_index, dims=[0])

        x_forward = self._gather_by_index(x, forward_index)
        x_backward = self._gather_by_index(x, backward_index)

        xs = torch.stack([x_forward, x_backward], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        y_forward = self._scatter_by_index(out_y[:, 0], forward_index, H, W)
        y_backward = self._scatter_by_index(out_y[:, 1], backward_index, H, W)

        return y_forward, y_backward

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2 = self.forward_core(x)
        assert y1.dtype == torch.float32

        y = y1 + y2

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


# Wavelet Pyramid Causal Scan
class SS2D_WaveletPyramidScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.lsconv = LSConv.LSConv(self.d_inner)
        self.act = nn.SiLU()

        # K=2：正向 Wavelet Pyramid scan + 逆向 Wavelet Pyramid scan
        self.K = 2

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self._scan_cache = {}

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)

        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)

        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)

        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _serpentine_order(coords):
        """
        对给定坐标做蛇形排序，减少行尾跳跃。
        coords: list[(i, j)]
        """
        if len(coords) == 0:
            return []

        rows = {}
        for i, j in coords:
            rows.setdefault(i, []).append(j)

        ordered = []
        sorted_rows = sorted(rows.keys())

        for ridx, i in enumerate(sorted_rows):
            js = sorted(rows[i])
            if ridx % 2 == 1:
                js = js[::-1]
            for j in js:
                ordered.append((i, j))

        return ordered

    def _build_wavelet_pyramid_index(self, H, W, device):
        """
        生成 Wavelet Pyramid Causal Scan 的一维索引。
        规则：
        1. 递归式粗到细；
        2. 每一级按 LL -> LH -> HL -> HH；
        3. 每个子带内部使用 serpentine scan。
        """
        key = ("wavelet_pyramid", H, W, device)
        if key in self._scan_cache:
            return self._scan_cache[key]

        coords = [(i, j) for i in range(H) for j in range(W)]

        def recursive_wavelet_order(coord_list):
            if len(coord_list) <= 4:
                return self._serpentine_order(coord_list)

            min_i = min(i for i, _ in coord_list)
            min_j = min(j for _, j in coord_list)

            LL, LH, HL, HH = [], [], [], []

            for i, j in coord_list:
                pi = (i - min_i) % 2
                pj = (j - min_j) % 2

                if pi == 0 and pj == 0:
                    LL.append((i, j))
                elif pi == 0 and pj == 1:
                    LH.append((i, j))
                elif pi == 1 and pj == 0:
                    HL.append((i, j))
                else:
                    HH.append((i, j))

            ordered = []
            ordered += recursive_wavelet_order(LL) if len(LL) < len(coord_list) else self._serpentine_order(LL)
            ordered += self._serpentine_order(LH)
            ordered += self._serpentine_order(HL)
            ordered += self._serpentine_order(HH)

            return ordered

        ordered_coords = recursive_wavelet_order(coords)

        # 去重保险
        seen = set()
        unique_coords = []
        for i, j in ordered_coords:
            if (i, j) not in seen:
                unique_coords.append((i, j))
                seen.add((i, j))

        for i in range(H):
            for j in range(W):
                if (i, j) not in seen:
                    unique_coords.append((i, j))

        index = torch.tensor([i * W + j for i, j in unique_coords], dtype=torch.long, device=device)

        self._scan_cache[key] = index
        return index

    @staticmethod
    def _gather_by_index(x, index):
        B, C, H, W = x.shape
        L = H * W
        x_flat = x.view(B, C, L)
        index = index.view(1, 1, L).expand(B, C, L)
        return torch.gather(x_flat, dim=2, index=index)

    @staticmethod
    def _scatter_by_index(y, index, H, W):
        B, C, L = y.shape
        out = torch.zeros(B, C, L, device=y.device, dtype=y.dtype)
        index = index.view(1, 1, L).expand(B, C, L)
        out.scatter_(dim=2, index=index, src=y)
        return out

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = self.K

        forward_index = self._build_wavelet_pyramid_index(H, W, x.device)
        backward_index = torch.flip(forward_index, dims=[0])

        x_forward = self._gather_by_index(x, forward_index)
        x_backward = self._gather_by_index(x, backward_index)

        xs = torch.stack([x_forward, x_backward], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        y_forward = self._scatter_by_index(out_y[:, 0], forward_index, H, W)
        y_backward = self._scatter_by_index(out_y[:, 1], backward_index, H, W)

        return y_forward, y_backward

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2 = self.forward_core(x)
        assert y1.dtype == torch.float32

        y = y1 + y2

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)

        return out



# # --------------------------------------------------------------------------------SDM-Scan------------------------------------------------------------------------------------------
# SDM-Scan
class SS2D_SDMScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @classmethod
    def _block_serpentine(cls, H, W, block=8, reverse_blocks=False, transpose=False):
        # Fast SDM proxy: local-continuity-preserving block serpentine.
        # It minimizes long jumps without per-image greedy search.
        bh = max(1, math.ceil(H / block))
        bw = max(1, math.ceil(W / block))
        block_coords = []
        for bi in range(bh):
            cols = range(bw) if bi % 2 == 0 else range(bw - 1, -1, -1)
            for bj in cols:
                block_coords.append((bi, bj))
        if reverse_blocks:
            block_coords = list(reversed(block_coords))

        path = []
        for bi, bj in block_coords:
            rs, re = bi * block, min((bi + 1) * block, H)
            cs, ce = bj * block, min((bj + 1) * block, W)
            rows = range(rs, re)
            for local_r, i in enumerate(rows):
                cols = range(cs, ce) if local_r % 2 == 0 else range(ce - 1, cs - 1, -1)
                for j in cols:
                    if transpose:
                        ti, tj = min(j, H - 1), min(i, W - 1)
                        path.append(cls._rid(ti, tj, W))
                    else:
                        path.append(cls._rid(i, j, W))
        return cls._dedup_complete(path, H, W)

    @classmethod
    def _build_fast_routes(cls, H, W):
        p0 = cls._block_serpentine(H, W, block=8, reverse_blocks=False, transpose=False)
        p1 = cls._reverse(p0)
        p2 = cls._block_serpentine(H, W, block=8, reverse_blocks=False, transpose=True)
        p3 = cls._reverse(p2)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

# ----------------------------------------------------------------------------------IPF-Scan------------------------------------------------------------------------------------------
# IPF-Scan
class SS2D_IPFScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @classmethod
    def _potential_order(cls, H, W, center_i, center_j, invert=False, angular_offset=0.0):
        # Fast IPF proxy: cached iso-potential layers using squared distance to a potential source.
        coords = []
        for i in range(H):
            for j in range(W):
                di = i - center_i
                dj = j - center_j
                r2 = di * di + dj * dj
                angle = math.atan2(di, dj) + angular_offset
                # level first, then angular sweep for continuity on each potential ring
                key = (-r2 if invert else r2, angle)
                coords.append((key, cls._rid(i, j, W)))
        coords.sort(key=lambda x: x[0])
        return [p for _, p in coords]

    @classmethod
    def _build_fast_routes(cls, H, W):
        ci, cj = (H - 1) / 2.0, (W - 1) / 2.0
        p0 = cls._potential_order(H, W, ci, cj, invert=False, angular_offset=0.0)
        p1 = cls._potential_order(H, W, ci, cj, invert=True, angular_offset=0.0)
        p2 = cls._potential_order(H, W, 0.0, 0.0, invert=False, angular_offset=0.75)
        p3 = cls._potential_order(H, W, H - 1.0, W - 1.0, invert=False, angular_offset=-0.75)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ----------------------------------------------------------------------------------RDW-Scan------------------------------------------------------------------------------------------
# RDW-Scan
class SS2D_RDWScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @classmethod
    def _wavefront_order(cls, H, W, seeds):
        # Fast RDW proxy: multi-source wavefront arrival time by Manhattan distance.
        coords = []
        for i in range(H):
            for j in range(W):
                best = None
                best_sid = 0
                for sid, (si, sj) in enumerate(seeds):
                    d = abs(i - si) + abs(j - sj)
                    if best is None or d < best:
                        best = d
                        best_sid = sid
                # Secondary key keeps each wavefront locally coherent.
                si, sj = seeds[best_sid]
                angle = math.atan2(i - si, j - sj)
                coords.append(((best, best_sid, angle), cls._rid(i, j, W)))
        coords.sort(key=lambda x: x[0])
        return [p for _, p in coords]

    @classmethod
    def _build_fast_routes(cls, H, W):
        center = (H // 2, W // 2)
        corners = [(0, 0), (0, W - 1), (H - 1, W - 1), (H - 1, 0)]
        edges = [(0, W // 2), (H // 2, W - 1), (H - 1, W // 2), (H // 2, 0)]
        p0 = cls._wavefront_order(H, W, [center])
        p1 = cls._wavefront_order(H, W, corners)
        p2 = cls._wavefront_order(H, W, edges)
        p3 = cls._reverse(p2)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ----------------------------------------------------------------------------------ECI-Scan--------------------------------------------------------------------------------------
# ECI-Scan
class SS2D_ECIScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @staticmethod
    def _coprime_stride(L, target):
        s = max(1, int(target) % max(1, L))
        for delta in range(L):
            for cand in (s + delta, s - delta):
                cand = cand % L
                if cand > 0 and gcd(cand, L) == 1:
                    return cand
        return 1

    @classmethod
    def _interleaver(cls, H, W, stride, offset=0):
        # Communication interleaver: pi(t)=(stride*t+offset) mod L, stride coprime with L.
        L = H * W
        stride = cls._coprime_stride(L, stride)
        return [int((stride * t + offset) % L) for t in range(L)]

    @classmethod
    def _build_fast_routes(cls, H, W):
        L = H * W
        p0 = cls._interleaver(H, W, stride=max(3, W + 1), offset=0)
        p1 = cls._interleaver(H, W, stride=max(5, H + W + 1), offset=L // 3)
        p2 = cls._interleaver(H, W, stride=max(7, 2 * W + 1), offset=L // 5)
        p3 = cls._reverse(p0)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ----------------------------------------------------------------------------------FSS-Scan------------------------------------------------------------------------------------------------#
# FSS-Scan
class SS2D_FSSScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @classmethod
    def _foveated_order(cls, H, W, anchors):
        # Fast FSS proxy: fixed fovea anchors, log-distance shells, angular local sweep.
        visited = set()
        path = []
        for ai, aj in anchors:
            coords = []
            for i in range(H):
                for j in range(W):
                    p = cls._rid(i, j, W)
                    if p in visited:
                        continue
                    di, dj = i - ai, j - aj
                    dist2 = di * di + dj * dj
                    shell = int(math.log2(dist2 + 1)) if dist2 > 0 else 0
                    angle = math.atan2(di, dj)
                    coords.append(((shell, dist2, angle), p))
            coords.sort(key=lambda x: x[0])
            for _, p in coords:
                if p not in visited:
                    path.append(p)
                    visited.add(p)
        return cls._dedup_complete(path, H, W)

    @classmethod
    def _build_fast_routes(cls, H, W):
        c = (H // 2, W // 2)
        anchors0 = [c, (H // 4, W // 4), (H // 4, 3 * W // 4), (3 * H // 4, W // 4), (3 * H // 4, 3 * W // 4)]
        anchors1 = [(H // 4, W // 4), (3 * H // 4, 3 * W // 4), c, (H // 4, 3 * W // 4), (3 * H // 4, W // 4)]
        anchors2 = [(0, 0), (0, W - 1), (H - 1, W - 1), (H - 1, 0), c]
        p0 = cls._foveated_order(H, W, anchors0)
        p1 = cls._foveated_order(H, W, anchors1)
        p2 = cls._foveated_order(H, W, anchors2)
        p3 = cls._reverse(p0)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ----------------------------------------------------------------------------------VGF-Scan------------------------------------------------------------------------------------------------#
# VGF-Scan
class SS2D_VGFScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @classmethod
    def _flow_basis_order(cls, H, W, theta, swirl=0.0):
        # Fast VGF proxy: fixed fluid-flow basis. Projection gives stream direction;
        # swirl term adds circulation around image center without dynamic ODE tracing.
        ci, cj = (H - 1) / 2.0, (W - 1) / 2.0
        ux, uy = math.cos(theta), math.sin(theta)
        coords = []
        for i in range(H):
            for j in range(W):
                x = j - cj
                y = i - ci
                proj = ux * x + uy * y
                cross = -uy * x + ux * y
                r2 = x * x + y * y
                vortex = swirl * math.atan2(y, x) + 1e-4 * r2
                coords.append(((proj + vortex, cross), cls._rid(i, j, W)))
        coords.sort(key=lambda x: x[0])
        return [p for _, p in coords]

    @classmethod
    def _build_fast_routes(cls, H, W):
        p0 = cls._flow_basis_order(H, W, theta=0.0, swirl=0.00)
        p1 = cls._flow_basis_order(H, W, theta=math.pi / 2, swirl=0.00)
        p2 = cls._flow_basis_order(H, W, theta=math.pi / 4, swirl=0.25)
        p3 = cls._flow_basis_order(H, W, theta=-math.pi / 4, swirl=-0.25)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ----------------------------------------------------------------------------------STES-Scan------------------------------------------------------------------------------------------------#
# STES-Scan
class SS2D_STESScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @classmethod
    def _event_hash_order(cls, H, W, seed=0):
        # Fast STES proxy: deterministic low-discrepancy spike-time code.
        # Gives event-like nonlocal temporal order without per-batch energy sorting.
        coords = []
        for i in range(H):
            for j in range(W):
                # integer hash; stable across devices and Python versions
                h = ((i + 1) * 73856093) ^ ((j + 1) * 19349663) ^ (seed * 83492791)
                # weak spatial term prevents too much pure randomness
                t = (h & 0x7fffffff) / 2147483647.0 + 0.015 * ((i + j) / max(1, H + W - 2))
                coords.append((t, cls._rid(i, j, W)))
        coords.sort(key=lambda x: x[0])
        return [p for _, p in coords]

    @classmethod
    def _build_fast_routes(cls, H, W):
        p0 = cls._event_hash_order(H, W, seed=1)
        p1 = cls._event_hash_order(H, W, seed=7)
        p2 = cls._event_hash_order(H, W, seed=31)
        p3 = cls._reverse(p0)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ----------------------------------------------------------------------------------ICS-Scan------------------------------------------------------------------------------------------------#
# ICS-Scan
class SS2D_ICSScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @classmethod
    def _curriculum_order(cls, H, W, reverse=False, offset_i=0, offset_j=0):
        # Fast ICS proxy: coarse-to-fine curriculum. Cell centers are scanned first,
        # then progressively finer locations, preserving local neighborhoods at each scale.
        path = []
        seen = set()
        max_side = max(H, W)
        step = 1
        while step * 2 <= max_side:
            step *= 2
        while step >= 1:
            half = step // 2
            for i in range((half + offset_i) % step, H, step):
                row_ids = []
                for j in range((half + offset_j) % step, W, step):
                    p = cls._rid(i, j, W)
                    if p not in seen:
                        row_ids.append(p)
                        seen.add(p)
                if (i // max(1, step)) % 2 == 1:
                    row_ids.reverse()
                path.extend(row_ids)
            step //= 2
        path = cls._dedup_complete(path, H, W)
        return cls._reverse(path) if reverse else path

    @classmethod
    def _build_fast_routes(cls, H, W):
        p0 = cls._curriculum_order(H, W, reverse=False, offset_i=0, offset_j=0)
        p1 = cls._curriculum_order(H, W, reverse=True, offset_i=0, offset_j=0)
        p2 = cls._curriculum_order(H, W, reverse=False, offset_i=1, offset_j=1)
        p3 = cls._reverse(p2)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ----------------------------------------------------------------------------------TSF-Scan------------------------------------------------------------------------------------------------#
# TSF-Scan
class SS2D_TSFScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        self._scan_index_cache = {}

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.lsconv = LSConv.LSConv(self.d_inner) if LSConv is not None else self.conv2d
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is not available. Please import it from your Mamba/VMamba environment.")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _rid(i, j, W):
        return int(i) * W + int(j)

    @staticmethod
    def _dedup_complete(path, H, W):
        L = H * W
        seen = set()
        out = []
        for p in path:
            p = int(p)
            if 0 <= p < L and p not in seen:
                out.append(p)
                seen.add(p)
        if len(out) < L:
            for p in range(L):
                if p not in seen:
                    out.append(p)
        return out[:L]

    @staticmethod
    def _reverse(path):
        return list(reversed(path))

    @staticmethod
    def _transpose_path(path, H, W):
        # Map path built on HxW to a transposed-coordinate route and complete safely.
        out = []
        for p in path:
            i, j = divmod(int(p), W)
            ti = min(j, H - 1)
            tj = min(i, W - 1)
            out.append(ti * W + tj)
        return out


    @classmethod
    def _skeleton_order(cls, H, W, mode=0):
        # Fast TSF proxy: static medial skeleton first, then distance-to-skeleton rings.
        ci, cj = (H - 1) / 2.0, (W - 1) / 2.0
        coords = []
        for i in range(H):
            for j in range(W):
                if mode == 0:
                    d_skel = min(abs(i - ci), abs(j - cj))  # cross skeleton
                    along = abs(i - ci) + abs(j - cj)
                elif mode == 1:
                    d_skel = abs((i - ci) - (j - cj)) / math.sqrt(2.0)  # main diagonal
                    along = (i + j)
                elif mode == 2:
                    d_skel = abs((i - ci) + (j - cj)) / math.sqrt(2.0)  # anti diagonal
                    along = (i - j)
                else:
                    d_skel = min(abs(i - ci), abs(j - cj), abs((i - ci) - (j - cj)) / math.sqrt(2.0), abs((i - ci) + (j - cj)) / math.sqrt(2.0))
                    along = math.atan2(i - ci, j - cj)
                coords.append(((d_skel, along), cls._rid(i, j, W)))
        coords.sort(key=lambda x: x[0])
        return [p for _, p in coords]

    @classmethod
    def _build_fast_routes(cls, H, W):
        p0 = cls._skeleton_order(H, W, mode=0)
        p1 = cls._skeleton_order(H, W, mode=1)
        p2 = cls._skeleton_order(H, W, mode=2)
        p3 = cls._skeleton_order(H, W, mode=3)
        return [p0, p1, p2, p3]


    def _build_scan_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (int(H), int(W), str(device))
        if key not in self._scan_index_cache:
            routes = self._build_fast_routes(int(H), int(W))
            idx = torch.tensor(routes, dtype=torch.long, device=device)
            if idx.shape != (self.K, H * W):
                raise RuntimeError(f"scan index should be (4, {H*W}), got {idx.shape}")
            self._scan_index_cache[key] = idx
        return self._scan_index_cache[key]

    def _scan_and_inverse(self, x: torch.Tensor, scan_indices: torch.Tensor):
        # x: (B, C, H, W), scan_indices: (K, L)
        B, C, H, W = x.shape
        K, L = scan_indices.shape
        x_flat = x.view(B, C, L)

        # Vectorized gather over routes: (B, K, C, L)
        idx_expand = scan_indices.view(1, K, 1, L).expand(B, K, C, L)
        x_expand = x_flat.view(B, 1, C, L).expand(B, K, C, L)
        xs = torch.gather(x_expand, dim=3, index=idx_expand).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Vectorized inverse scatter to raster layout.
        C_out = out_y.shape[2]
        out = torch.zeros_like(out_y)
        idx_expand_out = scan_indices.view(1, K, 1, L).expand(B, K, C_out, L)
        out.scatter_(dim=3, index=idx_expand_out, src=out_y)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        scan_indices = self._build_scan_indices(H, W, x.device)
        return self._scan_and_inverse(x, scan_indices)

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # x = self.act(self.lsconv(x))
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# GAPS-Scan MVP-v1
# 直接替换原工程中的 SS2D 类使用。
# 依赖：保持你原文件中已有的 math / torch / nn / F / repeat / selective_scan_fn / LSConv 导入不变。

class SS2D_GAPSScan_MVPv1(nn.Module):
    """
    GAPS-Scan MVP-v1

    路由：
        1) region_size=s 的 micro-region；
        2) K_route 个严格 4-neighbor 连续的 S-derived 路径模板；
        3) local cost = feature continuity；
        4) fixed horizontal region-level snake route；
        5) stitching cost = endpoint Manhattan distance；
        6) Dynamic Programming 选择各 region 的局部路径；
        7) GAPS-2P: pi + reverse(pi) 两条 Selective Scan。

    输入/输出接口与原 SS2D 保持一致：
        input : (B, H, W, C)
        output: (B, H, W, C)
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # ------------------------- GAPS-v1 配置 -------------------------
        self.gaps_region_size = int(kwargs.pop("gaps_region_size", 8))
        self.gaps_num_paths = int(kwargs.pop("gaps_num_paths", 8))
        self.gaps_lambda_dist = float(kwargs.pop("gaps_lambda_dist", 2.0))
        self.gaps_eps = float(kwargs.pop("gaps_eps", 1e-6))
        self.gaps_debug = bool(kwargs.pop("gaps_debug", False))

        if self.gaps_region_size < 2:
            raise ValueError("gaps_region_size must be >= 2")

        path_bank = self._build_path_bank(self.gaps_region_size)
        if self.gaps_num_paths > path_bank.shape[0]:
            raise ValueError(
                f"gaps_num_paths={self.gaps_num_paths} exceeds available paths={path_bank.shape[0]}"
            )
        path_bank = path_bank[:self.gaps_num_paths]
        self.register_buffer("gaps_path_bank", path_bank, persistent=False)  # (K_route, s^2)

        local_r = torch.div(path_bank, self.gaps_region_size, rounding_mode="floor")
        local_c = torch.remainder(path_bank, self.gaps_region_size)
        local_xy = torch.stack([local_r, local_c], dim=-1)  # (K_route, s^2, 2)
        self.register_buffer("gaps_local_xy", local_xy, persistent=False)
        self.register_buffer("gaps_path_entry", local_xy[:, 0], persistent=False)   # (K_route, 2)
        self.register_buffer("gaps_path_exit", local_xy[:, -1], persistent=False)  # (K_route, 2)

        # ------------------------- 原 SS2D 主体 -------------------------
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # 保持你的 LSConv 替换逻辑
        self.lsconv = LSConv.LSConv(self.d_inner)
        self.act = nn.SiLU()

        # GAPS-2P：Selective Scan 只保留 2 路参数，而不是原来的 4 路
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    # =====================================================================
    # GAPS route utilities
    # =====================================================================
    @staticmethod
    def _build_path_bank(s):
        """
        由 horizontal S-scan 的 D4 对称变换构造最多 8 条连续路径。
        每条路径：
            - 覆盖 s*s 个位置一次且仅一次；
            - 相邻 token Manhattan distance 恒为 1。
        返回：LongTensor [K, s*s]
        """
        base = []
        for r in range(s):
            cols = range(s) if (r % 2 == 0) else range(s - 1, -1, -1)
            for c in cols:
                base.append((r, c))

        candidates = []
        seen = set()
        for rot in range(4):
            for flip in (False, True):
                coords = []
                for r, c in base:
                    rr, cc = r, c
                    for _ in range(rot):
                        rr, cc = cc, s - 1 - rr
                    if flip:
                        cc = s - 1 - cc
                    coords.append((rr, cc))

                idx = tuple(rr * s + cc for rr, cc in coords)
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(idx)

        path_bank = torch.tensor(candidates, dtype=torch.long)

        # 初始化时做严格正确性检查
        target = torch.arange(s * s, dtype=torch.long)
        for k in range(path_bank.shape[0]):
            p = path_bank[k]
            if not torch.equal(torch.sort(p).values.cpu(), target):
                raise RuntimeError("Invalid GAPS local path: duplicate or missing token")
            xy = torch.stack([p // s, p % s], dim=-1)
            jump = (xy[1:] - xy[:-1]).abs().sum(dim=-1)
            if not torch.all(jump == 1):
                raise RuntimeError("Invalid GAPS local path: non-continuous step detected")

        return path_bank

    @staticmethod
    def _build_h_snake_region_route(Rh, Rw, device):
        """固定 horizontal region-level S route，返回 [M] region id。"""
        grid = torch.arange(Rh * Rw, device=device, dtype=torch.long).view(Rh, Rw)
        if Rh > 1:
            grid = grid.clone()
            grid[1::2] = torch.flip(grid[1::2], dims=[1])
        return grid.reshape(-1)

    def _route_descriptor(self, x):
        """
        MVP-v1: channel mean + detach。
        做 per-sample 标准化，避免不同 batch/层的幅值改变 route cost 尺度。
        """
        route_feat = x.detach().float().mean(dim=1, keepdim=True)
        mean = route_feat.mean(dim=(2, 3), keepdim=True)
        std = route_feat.std(dim=(2, 3), keepdim=True, unbiased=False)
        route_feat = (route_feat - mean) / (std + self.gaps_eps)
        return route_feat

    def _extract_region_patches(self, feat, s):
        """
        feat: (B,1,Hp,Wp), Hp/Wp 可被 s 整除
        return: (B,M,s^2)
        """
        patches = F.unfold(feat, kernel_size=s, stride=s)  # (B, s^2, M)
        return patches.transpose(1, 2).contiguous()        # (B, M, s^2)

    def _compute_local_cost(self, route_feat_pad):
        """
        E[b,m,k] = mean |f(p_t)-f(p_{t+1})|
        """
        s = self.gaps_region_size
        patches = self._extract_region_patches(route_feat_pad, s)       # B,M,S2
        path_seq = patches[:, :, self.gaps_path_bank]                    # B,M,K,S2
        feat_cost = (path_seq[..., 1:] - path_seq[..., :-1]).abs().mean(dim=-1)
        return feat_cost                                                 # B,M,K

    def _build_transition_cost(self, route_ids, Rh, Rw, device):
        """
        只使用 endpoint Manhattan distance：
            T[m,i,j] = lambda_dist * ||exit_i(R_m)-entry_j(R_{m+1})||_1
        """
        s = self.gaps_region_size
        rr = torch.div(route_ids, Rw, rounding_mode="floor")
        rc = torch.remainder(route_ids, Rw)
        base_xy = torch.stack([rr * s, rc * s], dim=-1)                 # M,2

        entry = base_xy[:, None, :] + self.gaps_path_entry.to(device)[None, :, :]  # M,K,2
        exit_ = base_xy[:, None, :] + self.gaps_path_exit.to(device)[None, :, :]   # M,K,2

        # M-1, K_prev, K_curr, 2
        delta = exit_[:-1, :, None, :] - entry[1:, None, :, :]
        dist = delta.abs().sum(dim=-1).float()
        return self.gaps_lambda_dist * dist

    @staticmethod
    def _dp_stitch(local_cost_q, transition_cost):
        """
        local_cost_q  : (B,M,K)
        transition_cost: (M-1,K,K)
        return:
            path_ids_q: (B,M), 每个 route-position 的 path id
            total_cost: (B,)
        """
        B, M, K = local_cost_q.shape
        dp = local_cost_q[:, 0, :]                                     # B,K
        prev_states = []

        for m in range(1, M):
            score = dp.unsqueeze(-1) + transition_cost[m - 1].unsqueeze(0)  # B,Kprev,Kcurr
            best_cost, best_prev = torch.min(score, dim=1)              # B,Kcurr
            dp = local_cost_q[:, m, :] + best_cost
            prev_states.append(best_prev)

        total_cost, last = torch.min(dp, dim=1)                          # B

        path_ids = torch.empty((B, M), device=local_cost_q.device, dtype=torch.long)
        path_ids[:, -1] = last
        current = last
        for m in range(M - 1, 0, -1):
            prev = prev_states[m - 1]                                   # B,Kcurr
            current = torch.gather(prev, 1, current[:, None]).squeeze(1)
            path_ids[:, m - 1] = current

        return path_ids, total_cost

    def _build_global_permutation(self, route_ids, path_ids_q, Rh, Rw, Hp, Wp):
        """
        route_ids : (M,)
        path_ids_q: (B,M), 对 route 顺序中的每个 region 选择的模板
        return pi: (B,Hp*Wp)
        """
        B, M = path_ids_q.shape
        s = self.gaps_region_size
        K = self.gaps_path_bank.shape[0]
        S2 = s * s
        device = path_ids_q.device

        all_region_ids = torch.arange(Rh * Rw, device=device, dtype=torch.long)
        rr = torch.div(all_region_ids, Rw, rounding_mode="floor")
        rc = torch.remainder(all_region_ids, Rw)

        local_r = torch.div(self.gaps_path_bank, s, rounding_mode="floor")          # K,S2
        local_c = torch.remainder(self.gaps_path_bank, s)                           # K,S2

        global_r = rr[:, None, None] * s + local_r[None, :, :]                     # M,K,S2
        global_c = rc[:, None, None] * s + local_c[None, :, :]                     # M,K,S2
        global_idx = global_r * Wp + global_c                                      # M,K,S2

        global_idx_q = global_idx[route_ids]                                       # M,K,S2
        gather_idx = path_ids_q[:, :, None, None].expand(B, M, 1, S2)
        selected = torch.gather(
            global_idx_q.unsqueeze(0).expand(B, -1, -1, -1),
            dim=2,
            index=gather_idx,
        ).squeeze(2)                                                               # B,M,S2

        pi = selected.reshape(B, Hp * Wp).long()
        return pi

    @staticmethod
    def _invert_permutation(pi):
        """pi[b,t]=original_index -> inv_pi[b,original_index]=t"""
        B, L = pi.shape
        inv_pi = torch.empty_like(pi)
        seq_pos = torch.arange(L, device=pi.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        inv_pi.scatter_(1, pi, seq_pos)
        return inv_pi

    def plan_gaps_route(self, x):
        """
        对当前 x 规划 batch-wise GAPS permutation。
        x: (B,C,H,W)
        return: pi, inv_pi, meta
        """
        B, C, H, W = x.shape
        s = self.gaps_region_size
        pad_h = (s - H % s) % s
        pad_w = (s - W % s) % s
        Hp, Wp = H + pad_h, W + pad_w
        Rh, Rw = Hp // s, Wp // s

        with torch.no_grad():
            route_feat = self._route_descriptor(x)
            if pad_h > 0 or pad_w > 0:
                route_feat = F.pad(route_feat, (0, pad_w, 0, pad_h), mode="replicate")

            local_cost = self._compute_local_cost(route_feat)            # B,M,K
            route_ids = self._build_h_snake_region_route(Rh, Rw, x.device)
            local_cost_q = local_cost[:, route_ids, :]
            transition = self._build_transition_cost(route_ids, Rh, Rw, x.device)
            path_ids_q, total_cost = self._dp_stitch(local_cost_q, transition)
            pi = self._build_global_permutation(route_ids, path_ids_q, Rh, Rw, Hp, Wp)
            inv_pi = self._invert_permutation(pi)

            if self.gaps_debug:
                target = torch.arange(Hp * Wp, device=x.device, dtype=torch.long)
                sorted_pi = torch.sort(pi, dim=1).values
                if not torch.all(sorted_pi == target[None, :]):
                    raise RuntimeError("GAPS global permutation contains duplicate/missing indices")

        meta = {
            "H": H, "W": W, "Hp": Hp, "Wp": Wp,
            "pad_h": pad_h, "pad_w": pad_w,
            "Rh": Rh, "Rw": Rw,
            "route_ids": route_ids,
            "path_ids": path_ids_q,
            "route_cost": total_cost,
        }
        return pi, inv_pi, meta

    # =====================================================================
    # 原 Mamba 参数初始化
    # =====================================================================
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # =====================================================================
    # GAPS-2P Selective Scan
    # =====================================================================
    def forward_core(self, x: torch.Tensor, gaps_route_info=None):
        B, C, H, W = x.shape
        K = 2

        if gaps_route_info is None:
            pi, inv_pi, meta = self.plan_gaps_route(x)
        else:
            pi, inv_pi, meta = gaps_route_info

        Hp, Wp = meta["Hp"], meta["Wp"]
        pad_h, pad_w = meta["pad_h"], meta["pad_w"]
        Lp = Hp * Wp

        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x

        x_flat = x_pad.contiguous().view(B, C, Lp)
        gather_idx = pi[:, None, :].expand(B, C, Lp)
        seq_forward = torch.gather(x_flat, dim=2, index=gather_idx)       # B,C,Lp
        seq_reverse = torch.flip(seq_forward, dims=[-1])
        xs = torch.stack([seq_forward, seq_reverse], dim=1)               # B,2,C,Lp

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs_scan = xs.float().view(B, -1, Lp)
        dts = dts.contiguous().float().view(B, -1, Lp)
        Bs = Bs.float().view(B, K, -1, Lp)
        Cs = Cs.float().view(B, K, -1, Lp)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs_scan, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, Lp)
        assert out_y.dtype == torch.float

        # 第 2 路先 reverse 回 pi 顺序，再分别 inverse permutation 回二维原位置
        y1_pi = out_y[:, 0]
        y2_pi = torch.flip(out_y[:, 1], dims=[-1])

        D = y1_pi.shape[1]
        inv_idx = inv_pi[:, None, :].expand(B, D, Lp)
        y1_flat = torch.gather(y1_pi, dim=2, index=inv_idx)
        y2_flat = torch.gather(y2_pi, dim=2, index=inv_idx)

        y1_map = y1_flat.view(B, D, Hp, Wp)[:, :, :H, :W]
        y2_map = y2_flat.view(B, D, Hp, Wp)[:, :, :H, :W]

        return y1_map.contiguous().view(B, D, H * W), y2_map.contiguous().view(B, D, H * W)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))
        # x = self.act(self.lsconv(x))

        # 可选：由父模块传入同一组 route，实现 route-group sharing
        gaps_route_info = kwargs.get("gaps_route_info", None)
        y1, y2 = self.forward_core(x, gaps_route_info=gaps_route_info)
        assert y1.dtype == torch.float32

        # GAPS-2P 按方案采用等权平均
        y = 0.5 * (y1 + y2)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# GAPS-Scan MVP-v2
# 直接替换原工程中的 SS2D 类使用。
# 依赖：保持你原文件中已有的 math / torch / nn / F / repeat / selective_scan_fn / LSConv 导入不变。
class SS2D_GAPSScan_MVPv2(nn.Module):
    """
    GAPS-Scan MVP-v2

    相比 MVP-v1 新增：
        1) direction-sensitive edge crossing cost；
        2) structure-tensor orientation cost；
        3) horizontal / vertical macro region route 二选一；
        4) 对外暴露 plan_gaps_route() 和 gaps_route_info，支持父模块做 route-group sharing。

    Selective Scan 仍使用 GAPS-2P：pi + reverse(pi)。
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # ------------------------- GAPS-v2 配置 -------------------------
        self.gaps_region_size = int(kwargs.pop("gaps_region_size", 8))
        self.gaps_num_paths = int(kwargs.pop("gaps_num_paths", 8))

        self.gaps_alpha_feat = float(kwargs.pop("gaps_alpha_feat", 0.50))
        self.gaps_beta_edge = float(kwargs.pop("gaps_beta_edge", 0.30))
        self.gaps_gamma_ori = float(kwargs.pop("gaps_gamma_ori", 0.20))
        self.gaps_lambda_dist = float(kwargs.pop("gaps_lambda_dist", 2.0))

        self.gaps_eps = float(kwargs.pop("gaps_eps", 1e-6))
        self.gaps_debug = bool(kwargs.pop("gaps_debug", False))

        if self.gaps_region_size < 2:
            raise ValueError("gaps_region_size must be >= 2")

        path_bank = self._build_path_bank(self.gaps_region_size)
        if self.gaps_num_paths > path_bank.shape[0]:
            raise ValueError(
                f"gaps_num_paths={self.gaps_num_paths} exceeds available paths={path_bank.shape[0]}"
            )
        path_bank = path_bank[:self.gaps_num_paths]
        self.register_buffer("gaps_path_bank", path_bank, persistent=False)

        s = self.gaps_region_size
        local_r = torch.div(path_bank, s, rounding_mode="floor")
        local_c = torch.remainder(path_bank, s)
        local_xy = torch.stack([local_r, local_c], dim=-1)
        self.register_buffer("gaps_local_xy", local_xy, persistent=False)
        self.register_buffer("gaps_path_entry", local_xy[:, 0], persistent=False)
        self.register_buffer("gaps_path_exit", local_xy[:, -1], persistent=False)

        # 路径 step 类型：horizontal step 对应 crossing Gx；vertical step 对应 crossing Gy
        step_xy = local_xy[:, 1:] - local_xy[:, :-1]                     # K,S2-1,2 (dr,dc)
        step_v = step_xy[..., 0].abs().float()                            # K,S2-1
        step_h = step_xy[..., 1].abs().float()                            # K,S2-1
        self.register_buffer("gaps_step_h", step_h, persistent=False)
        self.register_buffer("gaps_step_v", step_v, persistent=False)

        # 每条模板的整体主传播方向比例，用于 orientation cost
        path_h_ratio = step_h.mean(dim=-1)
        path_v_ratio = step_v.mean(dim=-1)
        denom = path_h_ratio + path_v_ratio + self.gaps_eps
        path_h_ratio = path_h_ratio / denom
        path_v_ratio = path_v_ratio / denom
        self.register_buffer("gaps_path_h_ratio", path_h_ratio, persistent=False)
        self.register_buffer("gaps_path_v_ratio", path_v_ratio, persistent=False)

        # Sobel：route descriptor 是单通道，所以 kernel 固定为 1x1x3x3
        sobel_x = torch.tensor(
            [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]], dtype=torch.float32
        ).unsqueeze(0) / 8.0
        sobel_y = torch.tensor(
            [[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]], dtype=torch.float32
        ).unsqueeze(0) / 8.0
        self.register_buffer("gaps_sobel_x", sobel_x, persistent=False)
        self.register_buffer("gaps_sobel_y", sobel_y, persistent=False)

        # ------------------------- 原 SS2D 主体 -------------------------
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.lsconv = LSConv.LSConv(self.d_inner)
        self.act = nn.SiLU()

        # GAPS-2P：2 套 SSM projection / A / D 参数
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    # =====================================================================
    # GAPS path bank / region routes
    # =====================================================================
    @staticmethod
    def _build_path_bank(s):
        base = []
        for r in range(s):
            cols = range(s) if (r % 2 == 0) else range(s - 1, -1, -1)
            for c in cols:
                base.append((r, c))

        candidates = []
        seen = set()
        for rot in range(4):
            for flip in (False, True):
                coords = []
                for r, c in base:
                    rr, cc = r, c
                    for _ in range(rot):
                        rr, cc = cc, s - 1 - rr
                    if flip:
                        cc = s - 1 - cc
                    coords.append((rr, cc))
                idx = tuple(rr * s + cc for rr, cc in coords)
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(idx)

        path_bank = torch.tensor(candidates, dtype=torch.long)
        target = torch.arange(s * s, dtype=torch.long)
        for k in range(path_bank.shape[0]):
            p = path_bank[k]
            if not torch.equal(torch.sort(p).values.cpu(), target):
                raise RuntimeError("Invalid GAPS local path: duplicate or missing token")
            xy = torch.stack([p // s, p % s], dim=-1)
            jump = (xy[1:] - xy[:-1]).abs().sum(dim=-1)
            if not torch.all(jump == 1):
                raise RuntimeError("Invalid GAPS local path: non-continuous step detected")
        return path_bank

    @staticmethod
    def _build_h_snake_region_route(Rh, Rw, device):
        grid = torch.arange(Rh * Rw, device=device, dtype=torch.long).view(Rh, Rw)
        if Rh > 1:
            grid = grid.clone()
            grid[1::2] = torch.flip(grid[1::2], dims=[1])
        return grid.reshape(-1)

    @staticmethod
    def _build_v_snake_region_route(Rh, Rw, device):
        # 在转置后的 region grid 上做 horizontal snake，再映射回原 region id
        grid = torch.arange(Rh * Rw, device=device, dtype=torch.long).view(Rh, Rw).transpose(0, 1).contiguous()
        if Rw > 1:
            grid = grid.clone()
            grid[1::2] = torch.flip(grid[1::2], dims=[1])
        return grid.reshape(-1)

    # =====================================================================
    # GAPS geometry costs
    # =====================================================================
    def _route_descriptor(self, x):
        route_feat = x.detach().float().mean(dim=1, keepdim=True)
        mean = route_feat.mean(dim=(2, 3), keepdim=True)
        std = route_feat.std(dim=(2, 3), keepdim=True, unbiased=False)
        return (route_feat - mean) / (std + self.gaps_eps)

    def _extract_region_patches(self, feat, s):
        patches = F.unfold(feat, kernel_size=s, stride=s)
        return patches.transpose(1, 2).contiguous()                       # B,M,S2

    def _compute_geometry_maps(self, route_feat_pad):
        gx = F.conv2d(route_feat_pad, self.gaps_sobel_x.to(route_feat_pad.dtype), padding=1)
        gy = F.conv2d(route_feat_pad, self.gaps_sobel_y.to(route_feat_pad.dtype), padding=1)
        return gx, gy

    def _compute_local_cost(self, route_feat_pad, gx, gy):
        """
        E = alpha * C_feat + beta * C_edge + gamma * C_ori

        C_edge 是方向敏感的：
            horizontal movement 使用 |Gx|；
            vertical movement   使用 |Gy|。
        """
        s = self.gaps_region_size

        feat_patch = self._extract_region_patches(route_feat_pad, s)      # B,M,S2
        gx_patch = self._extract_region_patches(gx, s)
        gy_patch = self._extract_region_patches(gy, s)

        feat_seq = feat_patch[:, :, self.gaps_path_bank]                  # B,M,K,S2
        gx_seq = gx_patch[:, :, self.gaps_path_bank]
        gy_seq = gy_patch[:, :, self.gaps_path_bank]

        # 1) feature continuity
        feat_cost = (feat_seq[..., 1:] - feat_seq[..., :-1]).abs().mean(dim=-1)

        # 2) direction-sensitive edge crossing
        edge_h = 0.5 * (gx_seq[..., :-1].abs() + gx_seq[..., 1:].abs())
        edge_v = 0.5 * (gy_seq[..., :-1].abs() + gy_seq[..., 1:].abs())
        step_h = self.gaps_step_h[None, None, :, :].to(edge_h.dtype)
        step_v = self.gaps_step_v[None, None, :, :].to(edge_v.dtype)
        edge_cost = (edge_h * step_h + edge_v * step_v).mean(dim=-1)

        # 3) structure tensor orientation
        # region-wise J = [[Gx^2, GxGy], [GxGy, Gy^2]]
        Jxx = (gx_patch * gx_patch).mean(dim=-1)                          # B,M
        Jyy = (gy_patch * gy_patch).mean(dim=-1)
        Jxy = (gx_patch * gy_patch).mean(dim=-1)

        theta_g = 0.5 * torch.atan2(2.0 * Jxy, Jxx - Jyy + self.gaps_eps)
        # 梯度方向的法向量 -> 边缘切向方向 theta_t = theta_g + pi/2
        tangent_x = torch.sin(theta_g).abs()                              # |cos(theta_t)| = |-sin(theta_g)|
        tangent_y = torch.cos(theta_g).abs()                              # |sin(theta_t)| = | cos(theta_g)|

        coherence = torch.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2 + self.gaps_eps)
        coherence = coherence / (Jxx + Jyy + self.gaps_eps)
        coherence = coherence.clamp(0.0, 1.0)

        h_ratio = self.gaps_path_h_ratio[None, None, :].to(tangent_x.dtype)
        v_ratio = self.gaps_path_v_ratio[None, None, :].to(tangent_x.dtype)
        alignment = tangent_x[..., None] * h_ratio + tangent_y[..., None] * v_ratio
        ori_cost = coherence[..., None] * (1.0 - alignment)

        local_cost = (
            self.gaps_alpha_feat * feat_cost
            + self.gaps_beta_edge * edge_cost
            + self.gaps_gamma_ori * ori_cost
        )
        return local_cost, feat_cost, edge_cost, ori_cost

    def _build_transition_cost(self, route_ids, Rh, Rw, device):
        s = self.gaps_region_size
        rr = torch.div(route_ids, Rw, rounding_mode="floor")
        rc = torch.remainder(route_ids, Rw)
        base_xy = torch.stack([rr * s, rc * s], dim=-1)

        entry = base_xy[:, None, :] + self.gaps_path_entry.to(device)[None, :, :]
        exit_ = base_xy[:, None, :] + self.gaps_path_exit.to(device)[None, :, :]
        delta = exit_[:-1, :, None, :] - entry[1:, None, :, :]
        dist = delta.abs().sum(dim=-1).float()
        return self.gaps_lambda_dist * dist

    @staticmethod
    def _dp_stitch(local_cost_q, transition_cost):
        B, M, K = local_cost_q.shape
        dp = local_cost_q[:, 0, :]
        prev_states = []

        for m in range(1, M):
            score = dp.unsqueeze(-1) + transition_cost[m - 1].unsqueeze(0)
            best_cost, best_prev = torch.min(score, dim=1)
            dp = local_cost_q[:, m, :] + best_cost
            prev_states.append(best_prev)

        total_cost, last = torch.min(dp, dim=1)
        path_ids = torch.empty((B, M), device=local_cost_q.device, dtype=torch.long)
        path_ids[:, -1] = last
        current = last
        for m in range(M - 1, 0, -1):
            prev = prev_states[m - 1]
            current = torch.gather(prev, 1, current[:, None]).squeeze(1)
            path_ids[:, m - 1] = current
        return path_ids, total_cost

    def _build_global_permutation(self, route_ids, path_ids_q, Rh, Rw, Hp, Wp):
        B, M = path_ids_q.shape
        s = self.gaps_region_size
        S2 = s * s
        device = path_ids_q.device

        all_region_ids = torch.arange(Rh * Rw, device=device, dtype=torch.long)
        rr = torch.div(all_region_ids, Rw, rounding_mode="floor")
        rc = torch.remainder(all_region_ids, Rw)

        local_r = torch.div(self.gaps_path_bank, s, rounding_mode="floor")
        local_c = torch.remainder(self.gaps_path_bank, s)
        global_r = rr[:, None, None] * s + local_r[None, :, :]
        global_c = rc[:, None, None] * s + local_c[None, :, :]
        global_idx = global_r * Wp + global_c                               # M,K,S2

        global_idx_q = global_idx[route_ids]
        gather_idx = path_ids_q[:, :, None, None].expand(B, M, 1, S2)
        selected = torch.gather(
            global_idx_q.unsqueeze(0).expand(B, -1, -1, -1),
            2,
            gather_idx,
        ).squeeze(2)
        return selected.reshape(B, Hp * Wp).long()

    @staticmethod
    def _invert_permutation(pi):
        B, L = pi.shape
        inv_pi = torch.empty_like(pi)
        seq_pos = torch.arange(L, device=pi.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        inv_pi.scatter_(1, pi, seq_pos)
        return inv_pi

    def _solve_macro_route(self, local_cost, route_ids, Rh, Rw, Hp, Wp):
        local_cost_q = local_cost[:, route_ids, :]
        transition = self._build_transition_cost(route_ids, Rh, Rw, local_cost.device)
        path_ids_q, total_cost = self._dp_stitch(local_cost_q, transition)
        pi = self._build_global_permutation(route_ids, path_ids_q, Rh, Rw, Hp, Wp)
        return pi, path_ids_q, total_cost

    def plan_gaps_route(self, x):
        """
        v2 完整 deterministic route planner。
        可由父模块显式调用一次，然后把返回值作为 gaps_route_info 传给同组 block。
        """
        B, C, H, W = x.shape
        s = self.gaps_region_size
        pad_h = (s - H % s) % s
        pad_w = (s - W % s) % s
        Hp, Wp = H + pad_h, W + pad_w
        Rh, Rw = Hp // s, Wp // s

        with torch.no_grad():
            route_feat = self._route_descriptor(x)
            if pad_h > 0 or pad_w > 0:
                route_feat = F.pad(route_feat, (0, pad_w, 0, pad_h), mode="replicate")

            gx, gy = self._compute_geometry_maps(route_feat)
            local_cost, feat_cost, edge_cost, ori_cost = self._compute_local_cost(route_feat, gx, gy)

            route_h = self._build_h_snake_region_route(Rh, Rw, x.device)
            route_v = self._build_v_snake_region_route(Rh, Rw, x.device)

            pi_h, path_h, cost_h = self._solve_macro_route(local_cost, route_h, Rh, Rw, Hp, Wp)
            pi_v, path_v, cost_v = self._solve_macro_route(local_cost, route_v, Rh, Rw, Hp, Wp)

            choose_v = cost_v < cost_h                                            # B
            pi = torch.where(choose_v[:, None], pi_v, pi_h)
            inv_pi = self._invert_permutation(pi)

            if self.gaps_debug:
                target = torch.arange(Hp * Wp, device=x.device, dtype=torch.long)
                sorted_pi = torch.sort(pi, dim=1).values
                if not torch.all(sorted_pi == target[None, :]):
                    raise RuntimeError("GAPS global permutation contains duplicate/missing indices")

        meta = {
            "H": H, "W": W, "Hp": Hp, "Wp": Wp,
            "pad_h": pad_h, "pad_w": pad_w,
            "Rh": Rh, "Rw": Rw,
            "choose_v": choose_v,
            "route_h": route_h,
            "route_v": route_v,
            "path_h": path_h,
            "path_v": path_v,
            "route_cost_h": cost_h,
            "route_cost_v": cost_v,
        }
        return pi, inv_pi, meta

    # =====================================================================
    # 原 Mamba 参数初始化
    # =====================================================================
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # =====================================================================
    # GAPS-2P Selective Scan
    # =====================================================================
    def forward_core(self, x: torch.Tensor, gaps_route_info=None):
        B, C, H, W = x.shape
        K = 2

        if gaps_route_info is None:
            pi, inv_pi, meta = self.plan_gaps_route(x)
        else:
            pi, inv_pi, meta = gaps_route_info

        Hp, Wp = meta["Hp"], meta["Wp"]
        pad_h, pad_w = meta["pad_h"], meta["pad_w"]
        Lp = Hp * Wp

        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x

        x_flat = x_pad.contiguous().view(B, C, Lp)
        gather_idx = pi[:, None, :].expand(B, C, Lp)
        seq_forward = torch.gather(x_flat, 2, gather_idx)
        seq_reverse = torch.flip(seq_forward, dims=[-1])
        xs = torch.stack([seq_forward, seq_reverse], dim=1)                 # B,2,C,Lp

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs_scan = xs.float().view(B, -1, Lp)
        dts = dts.contiguous().float().view(B, -1, Lp)
        Bs = Bs.float().view(B, K, -1, Lp)
        Cs = Cs.float().view(B, K, -1, Lp)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs_scan, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, Lp)
        assert out_y.dtype == torch.float

        y1_pi = out_y[:, 0]
        y2_pi = torch.flip(out_y[:, 1], dims=[-1])

        D = y1_pi.shape[1]
        inv_idx = inv_pi[:, None, :].expand(B, D, Lp)
        y1_flat = torch.gather(y1_pi, 2, inv_idx)
        y2_flat = torch.gather(y2_pi, 2, inv_idx)

        y1_map = y1_flat.view(B, D, Hp, Wp)[:, :, :H, :W]
        y2_map = y2_flat.view(B, D, Hp, Wp)[:, :, :H, :W]
        return y1_map.contiguous().view(B, D, H * W), y2_map.contiguous().view(B, D, H * W)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))
        # x = self.act(self.lsconv(x))

        gaps_route_info = kwargs.get("gaps_route_info", None)
        y1, y2 = self.forward_core(x, gaps_route_info=gaps_route_info)
        assert y1.dtype == torch.float32
        y = 0.5 * (y1 + y2)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# GAPS-Scan Final-v3
# 直接替换原工程中的 SS2D 类使用。
# 依赖：保持你原文件中已有的 math / torch / nn / F / repeat / selective_scan_fn / LSConv 导入不变。

class SS2D_GAPSScan_Finalv3(nn.Module):
    """
    GAPS-Scan Final-v3

    完整路由：
        - continuous local path bank
        - grouped-channel feature continuity
        - direction-sensitive edge crossing
        - structure-tensor orientation
        - H/V macro route selection
        - cross-region transition:
              endpoint distance + bridge feature + bridge edge
        - DP stitching

    高效扫描：
        - GAPS-1P：每个 block 仅执行 1 条 Selective Scan
        - 通过 gaps_block_id / gaps_reverse 控制相邻 block 的 pi / reverse(pi) 交替

    IMPORTANT:
        真正的 cross-layer alternation 需要父模块在构造各 SS2D 时传入不同 gaps_block_id。
        如果完全不改父模块，本类默认 gaps_block_id=0，会始终使用正向 pi，仍然可以正常训练，
        但不等同于论文方案中的跨层正反交替。
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # ------------------------- GAPS-v3 配置 -------------------------
        self.gaps_region_size = int(kwargs.pop("gaps_region_size", 8))
        self.gaps_num_paths = int(kwargs.pop("gaps_num_paths", 8))
        self.gaps_route_dim = int(kwargs.pop("gaps_route_dim", 8))

        self.gaps_alpha_feat = float(kwargs.pop("gaps_alpha_feat", 0.50))
        self.gaps_beta_edge = float(kwargs.pop("gaps_beta_edge", 0.30))
        self.gaps_gamma_ori = float(kwargs.pop("gaps_gamma_ori", 0.20))

        self.gaps_lambda_dist = float(kwargs.pop("gaps_lambda_dist", 2.0))
        self.gaps_lambda_bridge = float(kwargs.pop("gaps_lambda_bridge", 0.30))
        self.gaps_lambda_bridge_edge = float(kwargs.pop("gaps_lambda_bridge_edge", 0.20))
        self.gaps_dist_penalty = float(kwargs.pop("gaps_dist_penalty", 5.0))

        # 跨层正/反交替：父模块最好显式传 gaps_block_id=0,1,2,...
        self.gaps_block_id = int(kwargs.pop("gaps_block_id", 0))
        self.gaps_reverse = bool(kwargs.pop("gaps_reverse", (self.gaps_block_id % 2 == 1)))

        self.gaps_eps = float(kwargs.pop("gaps_eps", 1e-6))
        self.gaps_debug = bool(kwargs.pop("gaps_debug", False))

        if self.gaps_region_size < 2:
            raise ValueError("gaps_region_size must be >= 2")

        path_bank = self._build_path_bank(self.gaps_region_size)
        if self.gaps_num_paths > path_bank.shape[0]:
            raise ValueError(
                f"gaps_num_paths={self.gaps_num_paths} exceeds available paths={path_bank.shape[0]}"
            )
        path_bank = path_bank[:self.gaps_num_paths]
        self.register_buffer("gaps_path_bank", path_bank, persistent=False)

        s = self.gaps_region_size
        local_r = torch.div(path_bank, s, rounding_mode="floor")
        local_c = torch.remainder(path_bank, s)
        local_xy = torch.stack([local_r, local_c], dim=-1)
        self.register_buffer("gaps_local_xy", local_xy, persistent=False)
        self.register_buffer("gaps_path_entry", local_xy[:, 0], persistent=False)
        self.register_buffer("gaps_path_exit", local_xy[:, -1], persistent=False)
        self.register_buffer("gaps_path_entry_idx", path_bank[:, 0], persistent=False)
        self.register_buffer("gaps_path_exit_idx", path_bank[:, -1], persistent=False)

        step_xy = local_xy[:, 1:] - local_xy[:, :-1]
        step_v = step_xy[..., 0].abs().float()
        step_h = step_xy[..., 1].abs().float()
        self.register_buffer("gaps_step_h", step_h, persistent=False)
        self.register_buffer("gaps_step_v", step_v, persistent=False)

        path_h_ratio = step_h.mean(dim=-1)
        path_v_ratio = step_v.mean(dim=-1)
        denom = path_h_ratio + path_v_ratio + self.gaps_eps
        self.register_buffer("gaps_path_h_ratio", path_h_ratio / denom, persistent=False)
        self.register_buffer("gaps_path_v_ratio", path_v_ratio / denom, persistent=False)

        sobel_x = torch.tensor(
            [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]], dtype=torch.float32
        ).unsqueeze(0) / 8.0
        sobel_y = torch.tensor(
            [[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]], dtype=torch.float32
        ).unsqueeze(0) / 8.0
        self.register_buffer("gaps_sobel_x", sobel_x, persistent=False)
        self.register_buffer("gaps_sobel_y", sobel_y, persistent=False)

        # ------------------------- 原 SS2D 主体 -------------------------
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.lsconv = LSConv.LSConv(self.d_inner)
        self.act = nn.SiLU()

        # GAPS-1P：仅 1 套 SSM 参数
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    # =====================================================================
    # GAPS path bank / macro route
    # =====================================================================
    @staticmethod
    def _build_path_bank(s):
        base = []
        for r in range(s):
            cols = range(s) if (r % 2 == 0) else range(s - 1, -1, -1)
            for c in cols:
                base.append((r, c))

        candidates = []
        seen = set()
        for rot in range(4):
            for flip in (False, True):
                coords = []
                for r, c in base:
                    rr, cc = r, c
                    for _ in range(rot):
                        rr, cc = cc, s - 1 - rr
                    if flip:
                        cc = s - 1 - cc
                    coords.append((rr, cc))
                idx = tuple(rr * s + cc for rr, cc in coords)
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(idx)

        path_bank = torch.tensor(candidates, dtype=torch.long)
        target = torch.arange(s * s, dtype=torch.long)
        for k in range(path_bank.shape[0]):
            p = path_bank[k]
            if not torch.equal(torch.sort(p).values.cpu(), target):
                raise RuntimeError("Invalid GAPS local path: duplicate or missing token")
            xy = torch.stack([p // s, p % s], dim=-1)
            jump = (xy[1:] - xy[:-1]).abs().sum(dim=-1)
            if not torch.all(jump == 1):
                raise RuntimeError("Invalid GAPS local path: non-continuous step detected")
        return path_bank

    @staticmethod
    def _build_h_snake_region_route(Rh, Rw, device):
        grid = torch.arange(Rh * Rw, device=device, dtype=torch.long).view(Rh, Rw)
        if Rh > 1:
            grid = grid.clone()
            grid[1::2] = torch.flip(grid[1::2], dims=[1])
        return grid.reshape(-1)

    @staticmethod
    def _build_v_snake_region_route(Rh, Rw, device):
        grid = torch.arange(Rh * Rw, device=device, dtype=torch.long).view(Rh, Rw).transpose(0, 1).contiguous()
        if Rw > 1:
            grid = grid.clone()
            grid[1::2] = torch.flip(grid[1::2], dims=[1])
        return grid.reshape(-1)

    # =====================================================================
    # Descriptor / geometry maps
    # =====================================================================
    def _route_descriptors(self, x):
        """
        无额外学习参数的 route descriptor：

        route_vec:
            将 C 通道分成 g 个 group 后组内平均，再做 channel L2 normalize；
            用于 cosine feature continuity / bridge feature cost。

        route_scalar:
            channel mean + per-sample z-score；用于 Sobel / structure tensor。

        两者均 detach，保证离散路由不参与反向传播。
        """
        x_det = x.detach().float()
        B, C, H, W = x_det.shape

        # 选取能够整除 C 的 group 数，避免 padding channel
        g = math.gcd(C, max(1, self.gaps_route_dim))
        g = max(1, g)
        group_width = C // g
        route_vec = x_det.view(B, g, group_width, H, W).mean(dim=2)        # B,g,H,W
        route_vec = F.normalize(route_vec, p=2, dim=1, eps=self.gaps_eps)

        route_scalar = x_det.mean(dim=1, keepdim=True)
        mean = route_scalar.mean(dim=(2, 3), keepdim=True)
        std = route_scalar.std(dim=(2, 3), keepdim=True, unbiased=False)
        route_scalar = (route_scalar - mean) / (std + self.gaps_eps)

        return route_vec, route_scalar

    @staticmethod
    def _extract_scalar_patches(feat, s):
        # feat: B,1,H,W -> B,M,S2
        patches = F.unfold(feat, kernel_size=s, stride=s)
        return patches.transpose(1, 2).contiguous()

    @staticmethod
    def _extract_vector_patches(feat, s):
        # feat: B,D,H,W -> B,M,S2,D
        B, D, H, W = feat.shape
        patches = F.unfold(feat, kernel_size=s, stride=s)                 # B,D*S2,M
        M = patches.shape[-1]
        patches = patches.view(B, D, s * s, M).permute(0, 3, 2, 1).contiguous()
        return patches

    def _compute_geometry_maps(self, route_scalar_pad):
        gx = F.conv2d(route_scalar_pad, self.gaps_sobel_x.to(route_scalar_pad.dtype), padding=1)
        gy = F.conv2d(route_scalar_pad, self.gaps_sobel_y.to(route_scalar_pad.dtype), padding=1)
        return gx, gy

    # =====================================================================
    # Local path cost + endpoint descriptors
    # =====================================================================
    def _compute_local_cost_and_endpoints(self, route_vec_pad, route_scalar_pad, gx, gy):
        s = self.gaps_region_size

        vec_patch = self._extract_vector_patches(route_vec_pad, s)       # B,M,S2,D
        gx_patch = self._extract_scalar_patches(gx, s)                   # B,M,S2
        gy_patch = self._extract_scalar_patches(gy, s)

        vec_seq = vec_patch[:, :, self.gaps_path_bank, :]                # B,M,K,S2,D
        gx_seq = gx_patch[:, :, self.gaps_path_bank]                     # B,M,K,S2
        gy_seq = gy_patch[:, :, self.gaps_path_bank]

        # 1) cosine feature continuity
        cos_adj = (vec_seq[..., :-1, :] * vec_seq[..., 1:, :]).sum(dim=-1)
        feat_cost = (1.0 - cos_adj.clamp(-1.0, 1.0)).mean(dim=-1)

        # 2) direction-sensitive edge crossing
        edge_h = 0.5 * (gx_seq[..., :-1].abs() + gx_seq[..., 1:].abs())
        edge_v = 0.5 * (gy_seq[..., :-1].abs() + gy_seq[..., 1:].abs())
        step_h = self.gaps_step_h[None, None, :, :].to(edge_h.dtype)
        step_v = self.gaps_step_v[None, None, :, :].to(edge_v.dtype)
        edge_cost = (edge_h * step_h + edge_v * step_v).mean(dim=-1)

        # 3) structure tensor orientation
        Jxx = (gx_patch * gx_patch).mean(dim=-1)
        Jyy = (gy_patch * gy_patch).mean(dim=-1)
        Jxy = (gx_patch * gy_patch).mean(dim=-1)
        theta_g = 0.5 * torch.atan2(2.0 * Jxy, Jxx - Jyy + self.gaps_eps)
        tangent_x = torch.sin(theta_g).abs()
        tangent_y = torch.cos(theta_g).abs()
        coherence = torch.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2 + self.gaps_eps)
        coherence = (coherence / (Jxx + Jyy + self.gaps_eps)).clamp(0.0, 1.0)
        h_ratio = self.gaps_path_h_ratio[None, None, :].to(tangent_x.dtype)
        v_ratio = self.gaps_path_v_ratio[None, None, :].to(tangent_x.dtype)
        alignment = tangent_x[..., None] * h_ratio + tangent_y[..., None] * v_ratio
        ori_cost = coherence[..., None] * (1.0 - alignment)

        local_cost = (
            self.gaps_alpha_feat * feat_cost
            + self.gaps_beta_edge * edge_cost
            + self.gaps_gamma_ori * ori_cost
        )

        # candidate entry/exit descriptors，用于 cross-region bridge cost
        # vec_patch: B,M,S2,D；entry_idx/exit_idx: K
        entry_vec = vec_patch[:, :, self.gaps_path_entry_idx, :]          # B,M,K,D
        exit_vec = vec_patch[:, :, self.gaps_path_exit_idx, :]
        entry_gx = gx_patch[:, :, self.gaps_path_entry_idx]              # B,M,K
        exit_gx = gx_patch[:, :, self.gaps_path_exit_idx]
        entry_gy = gy_patch[:, :, self.gaps_path_entry_idx]
        exit_gy = gy_patch[:, :, self.gaps_path_exit_idx]

        endpoint = {
            "entry_vec": entry_vec,
            "exit_vec": exit_vec,
            "entry_gx": entry_gx,
            "exit_gx": exit_gx,
            "entry_gy": entry_gy,
            "exit_gy": exit_gy,
        }
        return local_cost, endpoint

    # =====================================================================
    # Full cross-region stitching cost
    # =====================================================================
    def _build_transition_cost(self, route_ids, endpoint, Rh, Rw):
        """
        return T: (B,M-1,K,K)

        T = lambda_d * distance_penalty
          + lambda_f * bridge_feature
          + lambda_e * bridge_edge
        """
        s = self.gaps_region_size
        device = route_ids.device

        rr = torch.div(route_ids, Rw, rounding_mode="floor")
        rc = torch.remainder(route_ids, Rw)
        base_xy = torch.stack([rr * s, rc * s], dim=-1)                  # M,2

        entry_xy = base_xy[:, None, :] + self.gaps_path_entry.to(device)[None, :, :]  # M,K,2
        exit_xy = base_xy[:, None, :] + self.gaps_path_exit.to(device)[None, :, :]    # M,K,2

        # current exit_i -> next entry_j
        delta = entry_xy[1:, None, :, :] - exit_xy[:-1, :, None, :]      # M-1,K,K,2 (dr,dc)
        dist = delta.abs().sum(dim=-1).float()                            # M-1,K,K
        dist_penalty = torch.clamp(dist - 1.0, min=0.0) * self.gaps_dist_penalty

        # endpoint descriptor 按 macro route 重排
        entry_vec = endpoint["entry_vec"][:, route_ids, :, :]           # B,M,K,D
        exit_vec = endpoint["exit_vec"][:, route_ids, :, :]
        entry_gx = endpoint["entry_gx"][:, route_ids, :]
        exit_gx = endpoint["exit_gx"][:, route_ids, :]
        entry_gy = endpoint["entry_gy"][:, route_ids, :]
        exit_gy = endpoint["exit_gy"][:, route_ids, :]

        # bridge feature cosine distance: B,M-1,Kprev,Kcurr
        cos_bridge = (
            exit_vec[:, :-1, :, None, :] * entry_vec[:, 1:, None, :, :]
        ).sum(dim=-1).clamp(-1.0, 1.0)
        bridge_feat = 1.0 - cos_bridge

        # bridge edge：按连接方向投影 Gx/Gy
        # horizontal displacement 对应 Gx；vertical displacement 对应 Gy
        abs_delta = delta.abs().float()
        norm = abs_delta.sum(dim=-1, keepdim=True).clamp_min(1.0)
        dir_v = abs_delta[..., 0] / norm[..., 0]                          # M-1,K,K
        dir_h = abs_delta[..., 1] / norm[..., 0]

        gx_bridge = 0.5 * (
            exit_gx[:, :-1, :, None].abs() + entry_gx[:, 1:, None, :].abs()
        )
        gy_bridge = 0.5 * (
            exit_gy[:, :-1, :, None].abs() + entry_gy[:, 1:, None, :].abs()
        )
        bridge_edge = gx_bridge * dir_h[None, ...] + gy_bridge * dir_v[None, ...]

        T = (
            self.gaps_lambda_dist * dist_penalty[None, ...]
            + self.gaps_lambda_bridge * bridge_feat
            + self.gaps_lambda_bridge_edge * bridge_edge
        )
        return T

    @staticmethod
    def _dp_stitch(local_cost_q, transition_cost):
        """
        local_cost_q: B,M,K
        transition_cost: B,M-1,K,K
        """
        B, M, K = local_cost_q.shape
        dp = local_cost_q[:, 0, :]
        prev_states = []

        for m in range(1, M):
            score = dp.unsqueeze(-1) + transition_cost[:, m - 1, :, :]
            best_cost, best_prev = torch.min(score, dim=1)
            dp = local_cost_q[:, m, :] + best_cost
            prev_states.append(best_prev)

        total_cost, last = torch.min(dp, dim=1)
        path_ids = torch.empty((B, M), device=local_cost_q.device, dtype=torch.long)
        path_ids[:, -1] = last
        current = last
        for m in range(M - 1, 0, -1):
            prev = prev_states[m - 1]
            current = torch.gather(prev, 1, current[:, None]).squeeze(1)
            path_ids[:, m - 1] = current
        return path_ids, total_cost

    def _build_global_permutation(self, route_ids, path_ids_q, Rh, Rw, Hp, Wp):
        B, M = path_ids_q.shape
        s = self.gaps_region_size
        S2 = s * s
        device = path_ids_q.device

        all_region_ids = torch.arange(Rh * Rw, device=device, dtype=torch.long)
        rr = torch.div(all_region_ids, Rw, rounding_mode="floor")
        rc = torch.remainder(all_region_ids, Rw)

        local_r = torch.div(self.gaps_path_bank, s, rounding_mode="floor")
        local_c = torch.remainder(self.gaps_path_bank, s)
        global_r = rr[:, None, None] * s + local_r[None, :, :]
        global_c = rc[:, None, None] * s + local_c[None, :, :]
        global_idx = global_r * Wp + global_c                               # M,K,S2

        global_idx_q = global_idx[route_ids]
        gather_idx = path_ids_q[:, :, None, None].expand(B, M, 1, S2)
        selected = torch.gather(
            global_idx_q.unsqueeze(0).expand(B, -1, -1, -1),
            2,
            gather_idx,
        ).squeeze(2)
        return selected.reshape(B, Hp * Wp).long()

    @staticmethod
    def _invert_permutation(pi):
        B, L = pi.shape
        inv_pi = torch.empty_like(pi)
        seq_pos = torch.arange(L, device=pi.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        inv_pi.scatter_(1, pi, seq_pos)
        return inv_pi

    def _solve_macro_route(self, local_cost, endpoint, route_ids, Rh, Rw, Hp, Wp):
        local_cost_q = local_cost[:, route_ids, :]
        transition = self._build_transition_cost(route_ids, endpoint, Rh, Rw)
        path_ids_q, total_cost = self._dp_stitch(local_cost_q, transition)
        pi = self._build_global_permutation(route_ids, path_ids_q, Rh, Rw, Hp, Wp)
        return pi, path_ids_q, total_cost

    def plan_gaps_route(self, x):
        B, C, H, W = x.shape
        s = self.gaps_region_size
        pad_h = (s - H % s) % s
        pad_w = (s - W % s) % s
        Hp, Wp = H + pad_h, W + pad_w
        Rh, Rw = Hp // s, Wp // s

        with torch.no_grad():
            route_vec, route_scalar = self._route_descriptors(x)
            if pad_h > 0 or pad_w > 0:
                route_vec = F.pad(route_vec, (0, pad_w, 0, pad_h), mode="replicate")
                route_scalar = F.pad(route_scalar, (0, pad_w, 0, pad_h), mode="replicate")

            gx, gy = self._compute_geometry_maps(route_scalar)
            local_cost, endpoint = self._compute_local_cost_and_endpoints(
                route_vec, route_scalar, gx, gy
            )

            route_h = self._build_h_snake_region_route(Rh, Rw, x.device)
            route_v = self._build_v_snake_region_route(Rh, Rw, x.device)

            pi_h, path_h, cost_h = self._solve_macro_route(
                local_cost, endpoint, route_h, Rh, Rw, Hp, Wp
            )
            pi_v, path_v, cost_v = self._solve_macro_route(
                local_cost, endpoint, route_v, Rh, Rw, Hp, Wp
            )

            choose_v = cost_v < cost_h
            pi = torch.where(choose_v[:, None], pi_v, pi_h)
            inv_pi = self._invert_permutation(pi)

            if self.gaps_debug:
                target = torch.arange(Hp * Wp, device=x.device, dtype=torch.long)
                sorted_pi = torch.sort(pi, dim=1).values
                if not torch.all(sorted_pi == target[None, :]):
                    raise RuntimeError("GAPS global permutation contains duplicate/missing indices")

        meta = {
            "H": H, "W": W, "Hp": Hp, "Wp": Wp,
            "pad_h": pad_h, "pad_w": pad_w,
            "Rh": Rh, "Rw": Rw,
            "choose_v": choose_v,
            "route_h": route_h,
            "route_v": route_v,
            "path_h": path_h,
            "path_v": path_v,
            "route_cost_h": cost_h,
            "route_cost_v": cost_v,
        }
        return pi, inv_pi, meta

    # =====================================================================
    # 原 Mamba 参数初始化
    # =====================================================================
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # =====================================================================
    # GAPS-1P Selective Scan
    # =====================================================================
    def forward_core(self, x: torch.Tensor, gaps_route_info=None, gaps_reverse=None):
        B, C, H, W = x.shape
        K = 1

        if gaps_route_info is None:
            pi, inv_pi, meta = self.plan_gaps_route(x)
        else:
            pi, inv_pi, meta = gaps_route_info

        reverse_flag = self.gaps_reverse if gaps_reverse is None else bool(gaps_reverse)

        Hp, Wp = meta["Hp"], meta["Wp"]
        pad_h, pad_w = meta["pad_h"], meta["pad_w"]
        Lp = Hp * Wp

        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x

        x_flat = x_pad.contiguous().view(B, C, Lp)
        gather_idx = pi[:, None, :].expand(B, C, Lp)
        seq = torch.gather(x_flat, 2, gather_idx)                           # B,C,Lp in pi order

        if reverse_flag:
            seq = torch.flip(seq, dims=[-1])

        xs = seq.unsqueeze(1)                                               # B,1,C,Lp
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs_scan = xs.float().view(B, -1, Lp)
        dts = dts.contiguous().float().view(B, -1, Lp)
        Bs = Bs.float().view(B, K, -1, Lp)
        Cs = Cs.float().view(B, K, -1, Lp)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs_scan, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, Lp)
        assert out_y.dtype == torch.float

        y_pi = out_y[:, 0]
        if reverse_flag:
            # 恢复到 pi sequence order，再执行 inverse permutation
            y_pi = torch.flip(y_pi, dims=[-1])

        D = y_pi.shape[1]
        inv_idx = inv_pi[:, None, :].expand(B, D, Lp)
        y_flat = torch.gather(y_pi, 2, inv_idx)
        y_map = y_flat.view(B, D, Hp, Wp)[:, :, :H, :W]
        return y_map.contiguous().view(B, D, H * W)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))
        # x = self.act(self.lsconv(x))

        gaps_route_info = kwargs.get("gaps_route_info", None)
        gaps_reverse = kwargs.get("gaps_reverse", None)
        y = self.forward_core(
            x,
            gaps_route_info=gaps_route_info,
            gaps_reverse=gaps_reverse,
        )
        assert y.dtype == torch.float32

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out