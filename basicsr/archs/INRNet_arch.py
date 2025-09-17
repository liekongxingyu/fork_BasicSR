from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F
import torch
from basicsr.archs.NAF_INR.arch_util import RLN, PatchEmbed, PatchUnEmbed, BasicLayer, ContextExtractor, DegradationINR
from basicsr.archs.NAF_INR.Fusion import LowRankFusion


@ARCH_REGISTRY.register()
class INRNet(nn.Module):
    """
    集成退化感知机制的 INRNet
    - 在每个 BasicLayer 后注入退化信息
    - 使用多尺度退化向量增强特征表示
    - 保持原有的 U-Net 架构和 skip connection
    """

    def __init__(self, in_chans=3, out_chans=3, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[8, 16, 17, 9, 9, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN],
                 # 退化感知参数
                 use_degradation=True,
                 inr_d=64,
                 context_dim=256,
                 degradation_types=20,
                 injection_type='channel_modulation',
                 inject_layers=[0, 1, 2, 3, 4]):  # 在哪些层注入退化信息
        super(INRNet, self).__init__()

        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.use_degradation = use_degradation
        self.inject_layers = inject_layers

        # 退化感知组件
        if self.use_degradation:
            self.context_extractor = ContextExtractor(context_dim=context_dim)
            self.degradation_inr = DegradationINR(
                d=inr_d,
                context_dim=context_dim,
                num_degradation_types=degradation_types
            )

        # 主干网络的patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # 第一层 - 增强版BasicLayer
        self.layer1 = self._make_enhanced_layer(
            layer_id=0,
            network_depth=sum(depths),
            dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratios[0],
            norm_layer=norm_layer[0],
            window_size=window_size,
            attn_ratio=attn_ratio[0],
            conv_type=conv_type[0],
            inr_d=inr_d,
            injection_type=injection_type
        )

        # 下采样和skip连接
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        # 第二层
        self.layer2 = self._make_enhanced_layer(
            layer_id=1,
            network_depth=sum(depths),
            dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratios[1],
            norm_layer=norm_layer[1],
            window_size=window_size,
            attn_ratio=attn_ratio[1],
            conv_type=conv_type[1],
            inr_d=inr_d,
            injection_type=injection_type
        )

        # 下采样和skip连接
        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        # 第三层（瓶颈层）
        self.layer3 = self._make_enhanced_layer(
            layer_id=2,
            network_depth=sum(depths),
            dim=embed_dims[2],
            depth=depths[2],
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratios[2],
            norm_layer=norm_layer[2],
            window_size=window_size,
            attn_ratio=attn_ratio[2],
            conv_type=conv_type[2],
            inr_d=inr_d,
            injection_type=injection_type
        )

        # 上采样
        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        # 第四层
        assert embed_dims[1] == embed_dims[3]
        self.layer4 = self._make_enhanced_layer(
            layer_id=3,
            network_depth=sum(depths),
            dim=embed_dims[3],
            depth=depths[3],
            num_heads=num_heads[3],
            mlp_ratio=mlp_ratios[3],
            norm_layer=norm_layer[3],
            window_size=window_size,
            attn_ratio=attn_ratio[3],
            conv_type=conv_type[3],
            inr_d=inr_d,
            injection_type=injection_type
        )

        # 上采样
        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        # 第五层
        assert embed_dims[0] == embed_dims[4]
        self.layer5 = self._make_enhanced_layer(
            layer_id=4,
            network_depth=sum(depths),
            dim=embed_dims[4],
            depth=depths[4],
            num_heads=num_heads[4],
            mlp_ratio=mlp_ratios[4],
            norm_layer=norm_layer[4],
            window_size=window_size,
            attn_ratio=attn_ratio[4],
            conv_type=conv_type[4],
            inr_d=inr_d,
            injection_type=injection_type
        )

        # 精细化层
        self.refinement = self._make_enhanced_layer(
            layer_id=5,
            network_depth=sum(depths),
            dim=embed_dims[4],
            depth=depths[5],
            num_heads=num_heads[4],
            mlp_ratio=mlp_ratios[4],
            norm_layer=norm_layer[4],
            window_size=window_size,
            attn_ratio=attn_ratio[4],
            conv_type=conv_type[4],
            inr_d=inr_d,
            injection_type=injection_type
        )

        # 输出层
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def _make_enhanced_layer(self, layer_id, network_depth, dim, depth, num_heads,
                             mlp_ratio, norm_layer, window_size, attn_ratio, conv_type,
                             inr_d, injection_type):
        """
        创建增强的BasicLayer，根据配置决定是否启用退化注入
        """
        if self.use_degradation and layer_id in self.inject_layers:
            # 创建带退化注入的层
            return BasicLayer(
                network_depth=network_depth,
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                window_size=window_size,
                attn_ratio=attn_ratio,
                attn_loc='last',
                conv_type=conv_type,
                use_degradation=True,
                inr_d=inr_d,
                injection_type=injection_type
            )
        else:
            # 创建原始的BasicLayer
            return BasicLayer(
                network_depth=network_depth,
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                window_size=window_size,
                attn_ratio=attn_ratio,
                attn_loc='last',
                conv_type=conv_type
            )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x, context_vector=None):
        """
        特征提取的前向传播，集成退化信息
        Args:
            x: 输入特征图
            context_vector: 上下文向量（如果使用退化感知）
        """
        # 第一层编码
        x = self.patch_embed(x)
        if self.use_degradation and 0 in self.inject_layers:
            x = self.layer1(x, self.degradation_inr, context_vector)
        else:
            x = self.layer1(x)
        skip1 = x

        # 第二层编码
        x = self.patch_merge1(x)
        if self.use_degradation and 1 in self.inject_layers:
            x = self.layer2(x, self.degradation_inr, context_vector)
        else:
            x = self.layer2(x)
        skip2 = x

        # 第三层（瓶颈层）
        x = self.patch_merge2(x)
        if self.use_degradation and 2 in self.inject_layers:
            x = self.layer3(x, self.degradation_inr, context_vector)
        else:
            x = self.layer3(x)

        # 第四层解码
        x = self.patch_split1(x)
        x = x + self.skip2(skip2)  # Skip connection
        if self.use_degradation and 3 in self.inject_layers:
            x = self.layer4(x, self.degradation_inr, context_vector)
        else:
            x = self.layer4(x)

        # 第五层解码
        x = self.patch_split2(x)
        x = x + self.skip1(skip1)  # Skip connection
        if self.use_degradation and 4 in self.inject_layers:
            x = self.layer5(x, self.degradation_inr, context_vector)
        else:
            x = self.layer5(x)

        # 精细化层
        if self.use_degradation and 5 in self.inject_layers:
            x = self.refinement(x, self.degradation_inr, context_vector)
        else:
            x = self.refinement(x)

        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        """
        完整的前向传播，包含退化感知处理
        """
        input_ = x
        x = self.check_image_size(x)

        # 退化感知处理
        if self.use_degradation:
            # 提取上下文向量
            context_vector = self.context_extractor(input_)  # [B, context_dim]

            # 带退化感知的特征提取
            x = self.forward_features(x, context_vector)

        else:
            # 原始的前向传播
            x = self.forward_features(x)

        # 残差连接
        x = x + input_

        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256)).cuda()
    net = INRNet().cuda()
    y = net(x)

    from thop import profile, clever_format

    macs, params = profile(net, (x,), verbose=False)
    macs = macs / 1e6
    params = params / 1e6
    print(f"MACs: {macs:.2f}M, Params: {params:.2f}M")
