from torch import nn
import torch
from torch.nn import functional as F
from basicsr.archs.NAFNet_util import ContextExtractor, DegradationINR, LayerNorm2d
from basicsr.utils.registry import ARCH_REGISTRY


class BaselineBlock(nn.Module):
    def __init__(self, c ,DW_Expand=1, FFN_Expand=2, drop_out_rate=0., inr_d=128):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel, c, 1, padding=0, stride=1)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inr_d, c, 1, padding=0),
            nn.Sigmoid()
        )
        
        self.gelu = nn.GELU()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1)
        self.conv5 = nn.Conv2d(ffn_channel, c, 1, padding=0, stride=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, inr_feat):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)

        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        w = self.ca(inr_feat)
        y = y * w + y  # INR式通道注意力加残差

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


@ARCH_REGISTRY.register()
class INRCA_Net(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 dw_expand=1, ffn_expand=2, inr_d=128, context_dim=256, degradation_types=20):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1)

        self.context_extractor = ContextExtractor(context_dim)
        self.degradation_inr = DegradationINR(d=inr_d, context_dim=context_dim, num_degradation_types=degradation_types)
        self.inr_d = inr_d

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            blocks = nn.ModuleList([BaselineBlock(chan, dw_expand, ffn_expand, inr_d=inr_d) for _ in range(num)])
            self.encoders.append(blocks)
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.ModuleList([BaselineBlock(chan, dw_expand, ffn_expand, inr_d=inr_d) for _ in range(middle_blk_num)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            blocks = nn.ModuleList([BaselineBlock(chan, dw_expand, ffn_expand, inr_d=inr_d) for _ in range(num)])
            self.decoders.append(blocks)

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        device = x.device
        context_vector = self.context_extractor(inp.to(device))
        degradation_map = self.degradation_inr(context_vector, (x.shape[2], x.shape[3]))

        encs = []
        flag_enc = 1
        for encoder, down in zip(self.encoders, self.downs):
            for blk in encoder:
                x = blk(x, degradation_map)
            encs.append(x)
            x = down(x)

            if flag_enc == 1:
                feature1 = x
            if flag_enc == 3:
                feature2 = x
            flag_enc += 1

        for blk in self.middle_blks:
            x = blk(x, degradation_map)
        feature3 = x

        flag_dec = 1
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            for blk in decoder:
                x = blk(x, degradation_map)

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
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5
        }

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

def test_INRCA_Net():
    # 参数示例，按需修改
    img_channel = 3
    width = 16
    middle_blk_num = 4
    enc_blk_nums = [2, 2, 2,2]  # 编码器每阶段块数
    dec_blk_nums = [2, 2, 2,2]  # 解码器每阶段块数
    dw_expand = 1
    ffn_expand = 2
    inr_d = 128
    context_dim = 256
    degradation_types = 10

    # 实例化模型
    model = INRCA_Net(
        img_channel=img_channel,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
        dw_expand=dw_expand,
        ffn_expand=ffn_expand,
        inr_d=inr_d,
        context_dim=context_dim,
        degradation_types=degradation_types
    )

    # 随机生成输入，保证尺寸不小于padder_size
    batch_size = 2
    height = 128
    width = 128
    input_tensor = torch.randn(batch_size, img_channel, height, width)

    # 前向推理
    output_dict = model(input_tensor)

    # 打印输出中各个特征shape
    print("Output shapes:")
    for k, v in output_dict.items():
        print(f"{k}: {v.shape}")

if __name__ == "__main__":
    test_INRCA_Net()