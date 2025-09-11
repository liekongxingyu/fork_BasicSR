import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


# 基础卷积块
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, gap=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            AvgMax(in_channel) if gap else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
    

class AvgBranch(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        # 每个通道都自适应池化到1x1
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):
        
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out

        return out
    
class LocalAvgBranch(nn.Module):
    def __init__(self, dim) -> None:
        super(LocalAvgBranch, self).__init__()

        # 局部均值滤波
        self.gap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):
        
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency
        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out
        return out

class StripGlobalAvgBranch(nn.Module):
    def __init__(self, dim, size) -> None:
        super(StripGlobalAvgBranch, self).__init__()

        
        self.gap = nn.AdaptiveAvgPool2d(size)

        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):
        
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency
        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out
        return out

class StripGlobalMaxBranch(nn.Module):
    def __init__(self, dim, kernel) -> None:
        super().__init__()

        self.mp = nn.AdaptiveMaxPool2d((kernel))
        
        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):

        high_frequency = self.mp(x)
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out


class MaxBranch(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):

        high_frequency = self.mp(x)
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out
    

class MaxDilateBranch(nn.Module):
    def __init__(self, dim) -> None:
        super(MaxDilateBranch, self).__init__()

        dilation = 2
        kernel = 5

        self.pad = nn.ReflectionPad2d(dilation*(kernel-1)//2)

        self.mp = nn.MaxPool2d(kernel_size=kernel, dilation=dilation, stride=1)
        
        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):

        high_frequency = self.mp(self.pad(x))
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out


class AvgMax(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gap = AvgBranch(dim)
        self.mp = MaxBranch(dim)
        self.local_gap = LocalAvgBranch(dim)
        self.globa_horizontal_avg = StripGlobalAvgBranch(dim, (None,1))
        self.globa_vertical_avg = StripGlobalAvgBranch(dim, (1,None))
        self.global_horizontal_max = StripGlobalMaxBranch(dim, (None,1))
        self.global_vertial_max = StripGlobalMaxBranch(dim, (1, None))
        self.maxdilation = MaxDilateBranch(dim)
    def forward(self, x):
        x1 = self.gap(x)
        x2 = self.mp(x)
        x3 = self.local_gap(x)
        x4 = self.globa_horizontal_avg(x)
        x5 = self.globa_vertical_avg(x)
        x6 = self.global_horizontal_max(x)
        x7 = self.global_vertial_max(x)
        x8 = self.maxdilation(x)
        return x1+x2+x3+x4+x5+x6+x7+x8


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, gap=False):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, gap=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, gap=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))



@ARCH_REGISTRY.register()
class PoolNet(nn.Module):

    def __init__(self, version):
        super(PoolNet, self).__init__()
        
        if version == 'small':
            num_res = 4
        elif version == 'base':
            num_res = 8
        elif version == 'large':
            num_res = 16

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


