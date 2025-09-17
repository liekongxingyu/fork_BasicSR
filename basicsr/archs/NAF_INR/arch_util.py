import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_


# 改进的层归一化
class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        # 数值稳定性参数
        self.eps = eps
        # 梯度控制参数
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        # 基于标准差生成rescale（重缩放因子）
        self.meta1 = nn.Conv2d(1, dim, 1)
        # 基于均值生成rebias（重偏移因子）
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        # C,H,W都做标准化
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(
            (input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(
                std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - \
        coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(
        1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(
        relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


# 窗口自注意力
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim //
                          self.num_heads).permute(2, 0, 3, 1, 4)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=None, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3,
                          padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3,
                          padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5,
                                  padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h %
                     self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w %
                     self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            # nW*B, window_size**2, C
            qkv = window_partition(shifted_QKV, self.window_size)

            attn_windows = self.attn(qkv)

            # merge windows
            shifted_out = window_reverse(
                attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(
                self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(
            dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim,
                       hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn:
            x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn:
            x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm:
            x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            x = x * rescale + rebias
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'], use_degradation=True, inr_d=128,
                 injection_type='channel_modulation'):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_degradation = use_degradation
        self.injection_type = injection_type

        attn_depth = attn_ratio * depth

        # 不是每个block都用注意力机制，有时候用卷积
        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i <
                         (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (
                                 i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

        if self.use_degradation:
            # 维度转换：从 [B, H*W, C] 到 [B, C, H, W]
            self.to_spatial = ToSpatialConverter()
            self.to_sequence = ToSequenceConverter()

            # 退化注入器
            self.degradation_injector = DegradationInjector(
                inr_d=inr_d,
                target_channels=dim,
                injection_type=injection_type
            )

    def forward(self, x, degradation_inr=None, context_vector=None):
        for blk in self.blocks:
            x = blk(x)

        # print(x.shape)
        if not self.use_degradation or degradation_inr is None:
            return x

        # 获取空间维度
        B, C,H,W = x.shape

        # 生成当前尺度的退化图
        degradation_map = degradation_inr(
            context_vector, (H, W))  # [B, inr_d, H, W]

        # 退化注入
        x_injected = self.degradation_injector(
            x, degradation_map)  # [B, C, H, W]

        return x_injected


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class ToSpatialConverter(nn.Module):
    """将 [B, H*W, C] 转换为 [B, C, H, W]"""

    def __init__(self):
        super().__init__()

    def forward(self, x, H, W):
        B, N, C = x.shape
        return x.transpose(1, 2).view(B, C, H, W)


class ToSequenceConverter(nn.Module):
    """将 [B, C, H, W] 转换为 [B, H*W, C]"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, H*W).transpose(1, 2)


hidden_list = [256, 256, 256]
L = 4
# Context提取器 - 应该独立使用


class ContextExtractor(nn.Module):
    def __init__(self, context_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, context_dim)
        )

    def forward(self, image):
        return self.backbone(image)  # [B, 256]


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


# 隐式神经退化场 - 内部生成退化类型版本
class DegradationINR(nn.Module):
    def __init__(self, d, context_dim=256,
                 cell_decode=True, num_degradation_types=10):
        super().__init__()
        self.d = d
        self.context_dim = context_dim
        self.cell_decode = cell_decode
        self.deg_type_dim = num_degradation_types

        # 添加退化类型生成器 - 从上下文向量中预测退化类型
        self.degradation_predictor = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.deg_type_dim)
        )

        # 计算MLP输入维度
        imnet_in_dim = 2 + 4 * L + self.deg_type_dim + context_dim

        if self.cell_decode:
            imnet_in_dim += 2

        self.imnet = MLP(imnet_in_dim, d, hidden_list)

    def generate_degradation_type(self, context_vector):
        # 直接返回 softmax 概率，维度 [B, deg_type_dim]
        logits = self.degradation_predictor(context_vector)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def query_degradation_vector(self, input_size, coord, context_vector, cell=None):
        """
        Args:
            coord: [B, H*W, 2] 像素坐标
            context_vector: [B, context_dim] 外部传入的上下文向量
            cell: [B, H*W, 2] 单元格信息
        Returns:
            [B, D, H, W] 退化向量
        """
        B, N, _ = coord.shape
        H, W = input_size

        # print(B,N,H,W)

        # 位置编码
        points_enc = self.positional_encoding(coord)  # [B, H*W, 4*L]

        # 内部生成退化类型编码
        deg_encoded = self.generate_degradation_type(
            context_vector)  # [B, deg_type_dim]

        # print(deg_encoded)
        deg_expanded = deg_encoded.unsqueeze(1).expand(
            B, N, self.deg_type_dim)  # [B, H*W, deg_type_dim]

        # 扩展上下文向量到所有像素
        context_expanded = context_vector.unsqueeze(1).expand(
            B, N, self.context_dim)  # [B, H*W, context_dim]

        # 拼接所有特征
        inp = torch.cat([
            coord,              # [B, H*W, 2]
            points_enc,         # [B, H*W, 4*L]
            deg_expanded,       # [B, H*W, deg_type_dim]
            context_expanded    # [B, H*W, context_dim]
        ], dim=-1)

        if self.cell_decode and cell is not None:
            inp = torch.cat([inp, cell], dim=-1)

        # MLP预测退化向量
        degradation_vectors = self.imnet(inp)  # [B, H*W, D]

        # 重塑为图像格式
        ret = degradation_vectors.view(
            B, H, W, self.d).permute(0, 3, 1, 2)  # [B, D, H, W]

        return ret

    def forward(self, context_vector, input_size):
        """
        前向传播 - 简化接口，只需要上下文向量和尺寸
        Args:
            context_vector: [B, context_dim] 外部预先提取的上下文向量  
            input_size: (H, W) 目标图像尺寸
        Returns:
            [B, D, H, W] 退化向量图
        """
        H, W = input_size
        B = context_vector.shape[0]

        # 生成坐标网格
        device = context_vector.device
        coord = make_coord((H, W)).to(device)

        coord = coord.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]

        # 生成单元格信息
        cell = None
        if self.cell_decode:
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / H
            cell[:, :, 1] *= 2 / W

        # 查询退化向量（内部自动生成退化类型）
        ret = self.query_degradation_vector(
            input_size, coord, context_vector, cell)

        return ret

    def positional_encoding(self, input):
        """位置编码"""
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32,
                                 device=input.device) * np.pi
        spectrum = input[..., None] * freq
        sin, cos = spectrum.sin(), spectrum.cos()
        input_enc = torch.stack([sin, cos], dim=-2)
        input_enc = input_enc.view(*shape[:-1], -1)
        return input_enc


class DegradationInjector(nn.Module):
    """
    退化注入器：将 DegradationINR 生成的退化图注入到特征图中。
    injection_type:
      - 'channel_modulation'：通道权重调制
      - 'spatial_attention'  ：空间注意力
      - 'feature_fusion'     ：特征拼接融合
    """

    def __init__(self, inr_d: int, target_channels: int, injection_type: str = 'channel_modulation'):
        super().__init__()
        self.injection_type = injection_type
        self.inr_d = inr_d
        self.target_channels = target_channels

        if injection_type == 'channel_modulation':
            # 全局池化 + 1x1 映射到目标通道，Sigmoid 得到逐通道权重
            self.adapter = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inr_d, target_channels, kernel_size=1,
                          stride=1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif injection_type == 'spatial_attention':
            # 3x3 提取空间线索到1通道，Sigmoid 得到逐像素权重
            mid = max(target_channels // 4, 8)
            self.adapter = nn.Sequential(
                nn.Conv2d(inr_d, mid, kernel_size=3,
                          stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, 1, kernel_size=1,
                          stride=1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif injection_type == 'feature_fusion':
            # 先把退化图映射到 C 通道，再与 X 拼接，用 1x1 融合回 C
            self.map_to_c = nn.Conv2d(
                inr_d, target_channels, kernel_size=1, stride=1, padding=0, bias=True)
            self.fuse = nn.Conv2d(target_channels * 2, target_channels,
                                  kernel_size=1, stride=1, padding=0, bias=True)
        else:
            raise ValueError(f'Unsupported injection_type: {injection_type}')

    def forward(self, features: torch.Tensor, degradation_map: torch.Tensor) -> torch.Tensor:
        """
        features: [B, C, H, W]
        degradation_map: [B, inr_d, H, W]
        return: [B, C, H, W]
        """

        if self.injection_type == 'channel_modulation':
            w = self.adapter(degradation_map)            # [B, C, 1, 1]
            return features * w + features

        elif self.injection_type == 'spatial_attention':
            a = self.adapter(degradation_map)            # [B, 1, H, W]
            return features * a
        else:  # 'feature_fusion'
            d_c = self.map_to_c(degradation_map)         # [B, C, H, W]
            return self.fuse(torch.cat([features, d_c], dim=1))  # [B, C, H, W]


def test_basic_layer():
    # 基本参数
    batch_size, dim, height, width = 1, 24, 64, 64
    context_dim, inr_d = 256, 128
    
    # 创建输入
    x = torch.randn(batch_size, dim, height, width)
    img = torch.randn(batch_size, 3, height, width)
    
    print(f"输入: {x.shape}")
    
    # 测试不带退化
    layer1 = BasicLayer(network_depth=6, dim=dim, depth=2, num_heads=4, use_degradation=False)
    out1 = layer1(x)
    print(f"无退化输出: {out1.shape}")
    
    # 测试带退化
    layer2 = BasicLayer(network_depth=6, dim=dim, depth=2, num_heads=4, use_degradation=True, 
                       inr_d=inr_d, context_dim=context_dim)
    context_extractor = ContextExtractor(context_dim)
    degradation_inr = DegradationINR(inr_d, context_dim)
    
    ctx = context_extractor(img)
    out2 = layer2(x, degradation_inr, ctx)
    print(f"带退化输出: {out2.shape}")

if __name__ == "__main__":
    test_basic_layer()
