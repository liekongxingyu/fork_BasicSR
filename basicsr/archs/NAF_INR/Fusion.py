import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankFusion(nn.Module):
    def __init__(self, degradation_dim, feature_dim, rank=8, num_heads=4):
        """
        简化的低秩融合模块 - 仅保留低秩分解和自注意力
        Args:
            degradation_dim: 退化向量维度
            feature_dim: 特征维度
            rank: 低秩分解的秩
            num_heads: 自注意力头数
        """
        super().__init__()
        self.D = degradation_dim
        self.C = feature_dim
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # ===== 低秩分解组件 =====
        self.mlp_U_e = nn.Linear(degradation_dim, feature_dim * rank)
        self.mlp_V_e = nn.Linear(degradation_dim, rank * feature_dim)
        self.mlp_U_d = nn.Linear(degradation_dim, feature_dim * rank)
        self.mlp_V_d = nn.Linear(degradation_dim, rank * feature_dim)

        # ===== 多头自注意力机制 =====
        self.self_attention = MultiHeadSelfAttention(
            feature_dim, num_heads, degradation_dim
        )

        # ===== 简化的融合权重学习 =====
        # 输入：退化特征 + 自注意力统计特征
        fusion_input_dim = degradation_dim + feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # α, β, γ (编码器、解码器、交互权重)
        )

        # ===== 特征增强模块 =====
        self.feature_enhancer = FeatureEnhancementModule(feature_dim)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, degradation, encoder_feat, decoder_feat):
        """
        简化的前向传播
        Args:
            degradation: (B, D, H, W) 退化特征
            encoder_feat: (B, C, H, W) 编码器特征
            decoder_feat: (B, C, H, W) 解码器特征
        Returns:
            result: (B, C, H, W) 融合后的特征
        """
        B, D, H, W = degradation.shape
        B, C, H, W = encoder_feat.shape

        # 检查维度匹配
        assert decoder_feat.shape == (
            B, C, H, W), f"解码器特征维度不匹配: {decoder_feat.shape}"

        # 展平所有张量
        deg_flat = degradation.permute(0, 2, 3, 1).contiguous().view(B*H*W, D)
        enc_flat = encoder_feat.permute(
            0, 2, 3, 1).contiguous().view(B*H*W, C, 1)
        dec_flat = decoder_feat.permute(
            0, 2, 3, 1).contiguous().view(B*H*W, C, 1)

        # ===== 1. 低秩变换 =====
        # 编码器分支变换
        U_e = self.mlp_U_e(deg_flat).view(B*H*W, self.C, self.rank)
        V_e = self.mlp_V_e(deg_flat).view(B*H*W, self.rank, self.C)
        z_e = torch.bmm(V_e, enc_flat)
        y_e = torch.bmm(U_e, z_e).squeeze(-1)  # (B*H*W, C)

        # 解码器分支变换
        U_d = self.mlp_U_d(deg_flat).view(B*H*W, self.C, self.rank)
        V_d = self.mlp_V_d(deg_flat).view(B*H*W, self.rank, self.C)
        z_d = torch.bmm(V_d, dec_flat)
        y_d = torch.bmm(U_d, z_d).squeeze(-1)  # (B*H*W, C)

        # ===== 2. 自注意力机制 =====
        # 应用自注意力到低秩变换后的特征
        y_e_reshaped = y_e.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        y_d_reshaped = y_d.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 对编码器和解码器特征分别应用自注意力
        y_e_att = self.self_attention(
            y_e_reshaped, degradation)  # (B, C, H, W)
        y_d_att = self.self_attention(
            y_d_reshaped, degradation)  # (B, C, H, W)

        # 重新展平
        y_e_att_flat = y_e_att.permute(0, 2, 3, 1).contiguous().view(B*H*W, C)
        y_d_att_flat = y_d_att.permute(0, 2, 3, 1).contiguous().view(B*H*W, C)

        # ===== 3. 融合权重学习 =====
        # 计算自注意力的统计特征用于融合
        att_stats = (y_e_att_flat + y_d_att_flat) / 2  # (B*H*W, C)

        # 构建融合输入
        fusion_input = torch.cat([deg_flat, att_stats], dim=-1)  # (B*H*W, D+C)

        fusion_weights = self.fusion_mlp(fusion_input)
        fusion_weights = F.softmax(fusion_weights, dim=-1)

        α = fusion_weights[:, 0:1]  # 编码器权重
        β = fusion_weights[:, 1:2]  # 解码器权重
        γ = fusion_weights[:, 2:3]  # 交互权重

        # ===== 4. 特征融合 =====
        # 基于权重融合自注意力后的特征
        final_fusion = (α * y_e_att_flat +
                        β * y_d_att_flat +
                        γ * (y_e_att_flat * y_d_att_flat))

        # ===== 5. 特征增强 =====
        result = final_fusion.view(B, H, W, self.C).permute(0, 3, 1, 2)
        enhanced_result = self.feature_enhancer(result)

        return enhanced_result


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制，结合退化特征进行调制
    """

    def __init__(self, feature_dim, num_heads, degradation_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim必须能被num_heads整除"

        # QKV投影层
        self.query_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.key_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, 1)

        # 退化特征调制层
        self.degradation_modulation = nn.Sequential(
            nn.Conv2d(degradation_dim, feature_dim, 1),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Conv2d(feature_dim, feature_dim, 1)

        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, x, degradation):
        """
        Args:
            x: (B, C, H, W) 输入特征
            degradation: (B, D, H, W) 退化特征
        Returns:
            output: (B, C, H, W) 自注意力后的特征
        """
        B, C, H, W = x.shape

        # 1. 生成QKV
        q = self.query_proj(x)   # (B, C, H, W)
        k = self.key_proj(x)     # (B, C, H, W)
        v = self.value_proj(x)   # (B, C, H, W)

        # 2. 退化特征调制
        deg_mod = self.degradation_modulation(degradation)  # (B, C, H, W)
        q = q * deg_mod
        k = k * deg_mod

        # 3. 重塑为多头格式
        q = q.view(B, self.num_heads, self.head_dim, H *
                   W).transpose(-2, -1)  # (B, heads, HW, head_dim)
        k = k.view(B, self.num_heads, self.head_dim, H *
                   W).transpose(-2, -1)  # (B, heads, HW, head_dim)
        v = v.view(B, self.num_heads, self.head_dim, H *
                   W).transpose(-2, -1)  # (B, heads, HW, head_dim)

        # 4. 计算注意力分数
        attention_scores = torch.matmul(
            q, k.transpose(-2, -1)) * self.scale  # (B, heads, HW, HW)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 5. 应用注意力权重
        attended_values = torch.matmul(
            attention_weights, v)  # (B, heads, HW, head_dim)

        # 6. 重塑回原始格式
        # (B, heads, head_dim, HW)
        attended_values = attended_values.transpose(-2, -1).contiguous()
        attended_values = attended_values.view(B, C, H, W)  # (B, C, H, W)

        # 7. 输出投影
        output = self.output_proj(attended_values)

        # 8. 残差连接
        return output + x


class FeatureEnhancementModule(nn.Module):
    """简化的特征增强模块"""

    def __init__(self, feature_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, 1)
        self.norm = nn.BatchNorm2d(feature_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm(self.conv1(x)))
        x = self.conv2(x)
        return x + residual  # 残差连接


def test_simplified_fusion():
    """测试简化的融合模块"""
    fusion = LowRankFusion(
        degradation_dim=64,
        feature_dim=128,
        rank=8,
        num_heads=4
    )

    # 创建测试输入
    B, D, C, H, W = 2, 64, 128, 32, 32
    degradation = torch.randn(B, D, H, W)
    encoder_feat = torch.randn(B, C, H, W)
    decoder_feat = torch.randn(B, C, H, W)

    print("测试输入维度:")
    print(f"  degradation: {degradation.shape}")
    print(f"  encoder_feat: {encoder_feat.shape}")
    print(f"  decoder_feat: {decoder_feat.shape}")

    # 前向传播
    result = fusion(degradation, encoder_feat, decoder_feat)

    print(f"\n输出形状: {result.shape}")
    print(f"形状保持: {result.shape == encoder_feat.shape}")
    print(f"参数量: {sum(p.numel() for p in fusion.parameters()):,}")

    return result


if __name__ == "__main__":
    test_simplified_fusion()
