# 测试滤波器进行高低频分解

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class LightweightDerainNet(nn.Module):
    """超轻量去雨网络 - 专注高低频分解"""
    def __init__(self, in_ch=3, base_ch=16):
        super().__init__()
        
        # 极简编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        feat_ch = base_ch * 4  # 64通道
        
        # 可学习低通滤波器
        self.lowpass_filter = nn.Conv2d(feat_ch, feat_ch, 
                                       kernel_size=7, padding=3, 
                                       groups=feat_ch, bias=False)
        
        # 低频分支（背景恢复）
        self.low_branch = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch//2, feat_ch//4, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        
        # 高频分支（细节处理）  
        self.high_branch = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch//2, feat_ch//4, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        
        # 融合输出
        self.output_conv = nn.Sequential(
            nn.Conv2d(feat_ch//2, feat_ch//4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch//4, in_ch, 3, 1, 1)
        )
        
        # 残差权重
        self.alpha = nn.Parameter(torch.tensor(0.8))
        
        # 初始化低通滤波器
        self._init_lowpass()
    
    def _init_lowpass(self):
        """初始化为高斯样式低通滤波器"""
        k = 7
        sigma = 1.5
        
        # 创建高斯核
        coords = torch.arange(k) - k // 2
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        # 应用到每个通道
        with torch.no_grad():
            for i in range(self.lowpass_filter.weight.shape[0]):
                self.lowpass_filter.weight[i, 0] = gauss_2d
    
    def forward(self, x):
        # 特征提取
        features = self.encoder(x)
        
        # 频率分解
        low_freq_feat = self.lowpass_filter(features)
        high_freq_feat = features - low_freq_feat
        
        # 双分支处理
        low_processed = self.low_branch(low_freq_feat)
        high_processed = self.high_branch(high_freq_feat)
        
        # 特征融合
        fused = torch.cat([low_processed, high_processed], dim=1)
        output = self.output_conv(fused)
        
        # 残差连接
        result = self.alpha * output + (1 - self.alpha) * x
        
        return {
            'output': result,
            'low_freq_feat': low_freq_feat,
            'high_freq_feat': high_freq_feat,
            'low_processed': low_processed,
            'high_processed': high_processed
        }