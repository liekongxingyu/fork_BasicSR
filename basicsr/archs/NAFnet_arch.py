import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.NAFNet_util import ContextExtractor, DegradationINR, LayerNorm2d, AvgPool2d, Local_Base
from basicsr.archs.NAF_INR.Fusion import LowRankFusion


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
    增强版的NAF_Baseline_INR：
    - 在编码器和解码器之间添加三张量低秩融合
    - 利用退化信息引导编码器-解码器特征的自适应融合
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 dw_expand=1, ffn_expand=2, inr_d=128, context_dim=256, degradation_types=20,
                 injection_type='channel_modulation',
                 inject_encoder=True,
                 inject_decoder=True,
                 fusion_rank=8,           # 低秩融合的秩
                 fusion_locations=[]):    # 在哪些层级进行融合 [0,1,2,...]
        super().__init__()

        # stem
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.inject_encoder = inject_encoder
        self.inject_decoder = inject_decoder
        self.injection_type = injection_type
        self.fusion_locations = fusion_locations

        # degradation components
        self.context_extractor = ContextExtractor(context_dim)
        self.degradation_inr = DegradationINR(
            d=inr_d, context_dim=context_dim, num_degradation_types=degradation_types)

        # encoders with injection
        self.encoders = nn.ModuleList()
        self.encoder_injectors = nn.ModuleList()
        self.downs = nn.ModuleList()

        # 添加融合模块
        self.fusion_modules = nn.ModuleDict()

        chan = width
        channel_dims = [width]  # 记录每层的通道数

        for i, num in enumerate(enc_blk_nums):
            # 创建编码器注入器
            encoder_injector = DegradationInjector(
                inr_d=inr_d,
                target_channels=chan,
                injection_type=injection_type
            ) if inject_encoder else None

            # 创建带注入的编码器块
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

            # 如果当前层需要融合，创建融合模块
            if i in fusion_locations:
                fusion_module = LowRankFusion(
                    degradation_dim=inr_d,
                    feature_dim=chan,
                    rank=fusion_rank
                )
                self.fusion_modules[f'fusion_enc_{i}'] = fusion_module

            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan *= 2
            channel_dims.append(chan)

        # middle channels
        self.middle_channels = chan

        # middle blocks
        self.middle_blks = nn.ModuleList([
            nn.Sequential(
                *[BaselineBlock(self.middle_channels, dw_expand, ffn_expand)
                  for _ in range(1)]
            ) for _ in range(middle_blk_num)
        ])

        # decoders with injection and fusion
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

            # 创建带注入的解码器块
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

            # 解码器层的融合（对应编码器层）
            dec_level = len(enc_blk_nums) - 1 - i
            if dec_level in fusion_locations:
                fusion_module = LowRankFusion(
                    degradation_dim=inr_d,
                    feature_dim=chan,
                    rank=fusion_rank
                )
                self.fusion_modules[f'fusion_dec_{i}'] = fusion_module

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # context extraction and degradation generation
        context_vector = self.context_extractor(inp)  # [B, context_dim]

        # 生成多尺度退化向量
        degradation_vectors = {}

        # encode with multi-level INR injection
        x = self.intro(inp)
        encs = []
        current_h, current_w = x.shape[2], x.shape[3]

        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            # 编码器块处理
            x = encoder(x, context_vector)

            # 如果需要在此层进行融合，生成对应尺度的退化向量
            if i in self.fusion_locations:
                degradation_vectors[f'enc_{i}'] = self.degradation_inr(
                    context_vector, (current_h, current_w)
                )  # [B, inr_d, H, W]

            encs.append(x)
            x = down(x)
            current_h, current_w = current_h // 2, current_w // 2

        # middle processing
        for middle_blk in self.middle_blks:
            x = middle_blk(x)

        # decode with multi-level INR injection and fusion
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            # 上采样
            x = up(x)
            current_h, current_w = current_h * 2, current_w * 2

            # 检查是否需要在此层进行融合
            dec_level = len(self.encoders) - 1 - i
            fusion_key = f'fusion_dec_{i}'

            if fusion_key in self.fusion_modules:
                # 生成当前尺度的退化向量
                if f'enc_{dec_level}' not in degradation_vectors:
                    degradation_vectors[f'enc_{dec_level}'] = self.degradation_inr(
                        context_vector, (current_h, current_w)
                    )

                degradation_vec = degradation_vectors[f'enc_{dec_level}']

                # 三张量融合：退化向量 + 编码器特征 + 解码器特征
                fused_features = self.fusion_modules[fusion_key](
                    degradation_vec,  # [B, inr_d, H, W]
                    enc_skip,         # [B, C, H, W] 编码器特征
                    x                 # [B, C, H, W] 解码器特征
                )

                # 残差连接：使用融合结果增强skip connection
                enhanced_skip = enc_skip + fused_features
                x = x + enhanced_skip
            else:
                # 普通的skip connection
                x = x + enc_skip

            # 解码器块处理
            x = decoder(x, context_vector)

        # final output
        x = self.ending(x)
        x = x + inp
        x = x[:, :, :H, :W]

        return {
            'output': x,
            'degradation_vectors': degradation_vectors  # 可选：返回退化向量用于分析
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
        inject_decoder=True,     # 解码器注入
        fusion_rank=32,
        fusion_locations=[1, 2, 3],  # 在编码器层1和3进行融合
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
