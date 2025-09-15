import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.NAFNet_util import ContextExtractor, DegradationINR, LayerNorm2d


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * \
                self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * \
                self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(
                    h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] -
                       s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(
                    out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(
                s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-
                                                    k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2,
                     (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            # compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size,
                             fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class BaselineBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(
            (1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


@ARCH_REGISTRY.register()
class NAF_Baseline(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dw_expand=1, ffn_expand=2):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        flag_enc = 1
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

            if flag_enc == 1:
                feature1 = x
            if flag_enc == 3:
                feature2 = x

            flag_enc += 1

        x = self.middle_blks(x)

        feature3 = x

        flag_dec = 1
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

            if flag_dec == 2:
                feature4 = x
            if flag_dec == 4:
                feature5 = x

            flag_dec += 1

        x = self.ending(x)
        x = x + inp

        x = x[:, :, :H, :W]

        return {
            'output': x,
            'feature1': feature1,  # 编码器1后
            'feature2': feature2,  # 编码器3后
            'feature3': feature3,  # 中间块后
            'feature4': feature4,  # 解码器2后
            'feature5': feature5   # 解码器4后
        }

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h %
                     self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w %
                     self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class BaselineLocal(Local_Base, NAF_Baseline):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAF_Baseline.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size,
                         train_size=train_size, fast_imp=fast_imp)


class EncoderBlockWithInjection(nn.Module):
    """
    编码器块，包含BaselineBlock + INR注入
    在编码特征提取后、下采样前进行退化注入
    """

    def __init__(self, chan, num_blocks, dw_expand, ffn_expand,
                 injector: nn.Module | None,   # DegradationInjector 或 None
                 inr: nn.Module | None):        # DegradationINR 或 None
        super().__init__()
        assert num_blocks >= 1

        # 编码器的基础块
        self.blocks = nn.Sequential(
            *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num_blocks)]
        )

        self.injector = injector
        self.inr = inr
        self.chan = chan

    def forward(self, x, context_vector):
        """
        x: [B, C, H, W]
        context_vector: [B, context_dim]
        """
        # 1. 先进行基础特征提取
        x = self.blocks(x)

        # 2. 在下采样前进行INR退化注入[web:94]
        do_inject = (self.injector is not None) and (self.inr is not None)
        if do_inject:
            B, C, H, W = x.shape
            deg_map = self.inr(context_vector, (H, W))   # [B, inr_d, H, W]
            x = self.injector(x, deg_map)                # [B, C, H, W]

        return x


class DecoderBlockWithInjection(nn.Module):
    """
    解码器块，包含INR注入 + BaselineBlock
    在skip connection融合后、特征解码前进行退化注入
    """

    def __init__(self, chan, num_blocks, dw_expand, ffn_expand,
                 injector: nn.Module | None,   # DegradationInjector 或 None
                 inr: nn.Module | None):        # DegradationINR 或 None
        super().__init__()
        assert num_blocks >= 1

        # 解码器的基础块
        self.blocks = nn.Sequential(
            *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num_blocks)]
        )

        self.injector = injector
        self.inr = inr
        self.chan = chan

    def forward(self, x, context_vector):
        """
        x: [B, C, H, W] (已经过up + skip connection)
        context_vector: [B, context_dim]
        """
        # 1. 在特征解码前进行INR退化注入[web:96]
        do_inject = (self.injector is not None) and (self.inr is not None)
        if do_inject:
            B, C, H, W = x.shape
            deg_map = self.inr(context_vector, (H, W))   # [B, inr_d, H, W]
            x = self.injector(x, deg_map)                # [B, C, H, W]

        # 2. 再进行特征解码处理
        x = self.blocks(x)

        return x


# 保持你原有的DegradationInjector不变
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


@ARCH_REGISTRY.register()
class NAF_Baseline_INR(nn.Module):
    """
    修改版的NAF_Baseline_INR：
    - 移除middle block的INR注入
    - 在编码器和解码器的每个层级都添加INR注入
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 dw_expand=1, ffn_expand=2, inr_d=128, context_dim=256, degradation_types=10,
                 injection_type='channel_modulation',
                 inject_encoder=True,    # 是否在编码器注入
                 inject_decoder=True):   # 是否在解码器注入
        super().__init__()

        # stem
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.inject_encoder = inject_encoder
        self.inject_decoder = inject_decoder
        self.injection_type = injection_type

        # degradation components
        self.context_extractor = ContextExtractor(context_dim)
        self.degradation_inr = DegradationINR(
            d=inr_d, context_dim=context_dim, num_degradation_types=degradation_types)

        # encoders with injection[web:99]
        self.encoders = nn.ModuleList()
        self.encoder_injectors = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width

        for i, num in enumerate(enc_blk_nums):
            # 创建编码器注入器
            encoder_injector = DegradationInjector(
                inr_d=inr_d,
                target_channels=chan,
                injection_type=injection_type
            ) if inject_encoder else None

            # 创建带注入的编码器块[web:94]
            encoder_block = EncoderBlockWithInjection(
                chan=chan,
                num_blocks=num,
                dw_expand=dw_expand,
                ffn_expand=ffn_expand,
                injector=encoder_injector,
                inr=self.degradation_inr
            )

            self.encoders.append(encoder_block)
            self.encoder_injectors.append(encoder_injector)
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan *= 2

        # middle channels (去掉INR注入)
        self.middle_channels = chan

        # middle blocks (简化为普通的BaselineBlock)[web:96]
        self.middle_blks = nn.ModuleList([
            nn.Sequential(
                *[BaselineBlock(self.middle_channels, dw_expand, ffn_expand)
                  for _ in range(1)]
            ) for _ in range(middle_blk_num)
        ])

        # decoders with injection[web:101]
        self.decoders = nn.ModuleList()
        self.decoder_injectors = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, num in enumerate(dec_blk_nums):
            # 上采样层
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2

            # 创建解码器注入器
            decoder_injector = DegradationInjector(
                inr_d=inr_d,
                target_channels=chan,
                injection_type=injection_type
            ) if inject_decoder else None

            # 创建带注入的解码器块[web:96]
            decoder_block = DecoderBlockWithInjection(
                chan=chan,
                num_blocks=num,
                dw_expand=dw_expand,
                ffn_expand=ffn_expand,
                injector=decoder_injector,
                inr=self.degradation_inr
            )

            self.decoders.append(decoder_block)
            self.decoder_injectors.append(decoder_injector)

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # context extraction
        context_vector = self.context_extractor(inp)  # [B, context_dim]

        # encode with multi-level INR injection[web:99]
        x = self.intro(inp)
        encs = []

        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            # 编码器块处理（内部包含INR注入）
            x = encoder(x, context_vector)  # [B, C, H, W]
            encs.append(x)
            x = down(x)  # 下采样

        # middle processing (无INR注入)[web:94]
        for middle_blk in self.middle_blks:
            x = middle_blk(x)

        # decode with multi-level INR injection[web:101]
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            # 上采样 + skip connection
            x = up(x)
            x = x + enc_skip

            # 解码器块处理（内部包含INR注入）
            x = decoder(x, context_vector)  # [B, C, H, W]

        # final output
        x = self.ending(x)
        x = x + inp
        x = x[:, :, :H, :W]

        return {
            'output': x
        }

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h %
                     self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w %
                     self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


if __name__ == "__main__":
    # 创建多层级INR注入的模型
    model = NAF_Baseline_INR(
        img_channel=3,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=[1, 1, 1, 28],  # 4个编码器层级
        dec_blk_nums=[1, 1, 1, 1],   # 4个解码器层级
        inr_d=64,
        context_dim=256,
        degradation_types=20,
        injection_type='channel_modulation',
        inject_encoder=True,    # 编码器注入
        inject_decoder=True     # 解码器注入
    )

    # 测试
    test_input = torch.randn(1, 3, 256, 256)
    output = model(test_input)
    print(f"Output shape: {output['output'].shape}")

    # 计算参数量（以M为单位显示）
    total_params = sum(p.numel() for p in model.parameters())
    total_params_M = total_params / 1_000_000
    print(f"Total parameters: {total_params_M:.2f}M")

    # 如果需要详细信息，也可以这样显示
    # print(f"Total parameters: {total_params:,} ({total_params_M:.2f}M)")
