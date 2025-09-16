import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankFusion(nn.Module):
    def __init__(self, degradation_dim, feature_dim, rank=8, use_spatial_attention=True, 
                 use_channel_attention=True, use_cross_attention=True):
        """
        增强的三张量低秩融合模块 - 集成多种注意力机制
        """
        super().__init__()
        self.D = degradation_dim
        self.C = feature_dim
        self.rank = rank
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_attention = use_channel_attention
        self.use_cross_attention = use_cross_attention

        # ===== 原有的低秩分解组件 =====
        self.mlp_U_e = nn.Linear(degradation_dim, feature_dim * rank)
        self.mlp_V_e = nn.Linear(degradation_dim, rank * feature_dim)
        self.mlp_U_d = nn.Linear(degradation_dim, feature_dim * rank)
        self.mlp_V_d = nn.Linear(degradation_dim, rank * feature_dim)

        # ===== 新增：空间注意力机制 =====
        if self.use_spatial_attention:
            self.spatial_attention = SpatialAttentionModule(
                degradation_dim, feature_dim
            )

        # ===== 新增：通道注意力机制 =====
        if self.use_channel_attention:
            self.channel_attention = ChannelAttentionModule(
                degradation_dim, feature_dim
            )

        # ===== 新增：编码器-解码器交叉注意力 =====
        if self.use_cross_attention:
            self.cross_attention = CrossAttentionModule(
                degradation_dim, feature_dim, rank
            )
            # 交叉注意力特征投影（预定义避免运行时创建）
            self.cross_proj = nn.Linear(rank, feature_dim)

        # ===== 增强的融合权重学习 =====
        fusion_input_dim = degradation_dim
        if self.use_spatial_attention:
            fusion_input_dim += 1  # 空间注意力贡献（标量统计）
        if self.use_channel_attention:
            fusion_input_dim += feature_dim  # 通道注意力贡献
        if self.use_cross_attention:
            fusion_input_dim += 1  # 交叉注意力贡献（标量统计）

        self.enhanced_fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4)  # α, β, γ, δ
        )

        # ===== 特征增强模块 =====
        self.feature_enhancer = FeatureEnhancementModule(feature_dim)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, degradation, encoder_feat, decoder_feat):
        """
        增强的前向传播 - 修复维度错误
        """
        B, D, H, W = degradation.shape
        B, C, H, W = encoder_feat.shape

        # 检查维度匹配
        assert decoder_feat.shape == (B, C, H, W), f"解码器特征维度不匹配: {decoder_feat.shape}"

        # 展平所有张量
        deg_flat = degradation.permute(0, 2, 3, 1).contiguous().view(B*H*W, D)
        enc_flat = encoder_feat.permute(0, 2, 3, 1).contiguous().view(B*H*W, C, 1)
        dec_flat = decoder_feat.permute(0, 2, 3, 1).contiguous().view(B*H*W, C, 1)

        # ===== 1. 原有的低秩变换 =====
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

        # ===== 2. 应用注意力机制 =====
        attention_features = []

        # 空间注意力
        if self.use_spatial_attention:
            spatial_att = self.spatial_attention(degradation, encoder_feat, decoder_feat)  # (B, C, H, W)
            spatial_att_flat = spatial_att.permute(0, 2, 3, 1).contiguous().view(B*H*W, C)
            y_e_spatial = y_e * spatial_att_flat
            y_d_spatial = y_d * spatial_att_flat
            # 添加空间注意力的统计信息到融合特征
            spatial_stat = spatial_att_flat.mean(dim=1, keepdim=True)  # (B*H*W, 1)
            attention_features.append(spatial_stat)
        else:
            y_e_spatial, y_d_spatial = y_e, y_d

        # 通道注意力 - 修复维度错误
        if self.use_channel_attention:
            channel_att = self.channel_attention(degradation, encoder_feat, decoder_feat)  # (B, C, 1, 1)
            # 正确的维度处理
            channel_att_expanded = channel_att.squeeze(-1).squeeze(-1)  # (B, C)
            channel_att_flat = channel_att_expanded.unsqueeze(1).repeat(1, H*W, 1).view(B*H*W, C)  # (B*H*W, C)
            
            y_e_channel = y_e_spatial * channel_att_flat
            y_d_channel = y_d_spatial * channel_att_flat
            # 添加通道注意力特征
            attention_features.append(channel_att_expanded.repeat_interleave(H*W, dim=0))  # (B*H*W, C)
        else:
            y_e_channel, y_d_channel = y_e_spatial, y_d_spatial

        # 交叉注意力
        cross_att_feat = None
        if self.use_cross_attention:
            cross_att_feat = self.cross_attention(degradation, encoder_feat, decoder_feat)  # (B, rank, H, W)
            cross_att_flat = cross_att_feat.permute(0, 2, 3, 1).contiguous().view(B*H*W, self.rank)
            # 添加交叉注意力的统计信息
            cross_stat = cross_att_flat.mean(dim=1, keepdim=True)  # (B*H*W, 1)
            attention_features.append(cross_stat)

        # ===== 3. 增强的融合权重学习 =====
        fusion_input = [deg_flat]
        fusion_input.extend(attention_features)
        fusion_input_concat = torch.cat(fusion_input, dim=-1)

        fusion_weights = self.enhanced_fusion_mlp(fusion_input_concat)
        fusion_weights = F.softmax(fusion_weights, dim=-1)

        α = fusion_weights[:, 0:1]  # 编码器权重
        β = fusion_weights[:, 1:2]  # 解码器权重  
        γ = fusion_weights[:, 2:3]  # 交互权重
        δ = fusion_weights[:, 3:4]  # 注意力权重

        # ===== 4. 多层次融合 =====
        # 基础融合
        basic_fusion = α * y_e_channel + β * y_d_channel + γ * (y_e_channel * y_d_channel)

        # 如果有交叉注意力，加入其贡献
        if cross_att_feat is not None:
            cross_contribution = self.cross_proj(cross_att_flat)  # (B*H*W, C)
            final_fusion = basic_fusion + δ * cross_contribution
        else:
            final_fusion = basic_fusion

        # ===== 5. 特征增强 =====
        result = final_fusion.view(B, H, W, self.C).permute(0, 3, 1, 2)
        enhanced_result = self.feature_enhancer(result)

        return enhanced_result


class SpatialAttentionModule(nn.Module):
    """空间注意力模块"""
    def __init__(self, degradation_dim, feature_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(degradation_dim + feature_dim * 2, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, feature_dim, 1)
        
    def forward(self, degradation, encoder_feat, decoder_feat):
        # 拼接输入特征
        combined = torch.cat([degradation, encoder_feat, decoder_feat], dim=1)
        
        # 生成空间注意力权重
        x = F.relu(self.conv1(combined))
        x = F.relu(self.conv2(x))
        attention = torch.sigmoid(self.conv3(x))
        
        return attention


class ChannelAttentionModule(nn.Module):
    """通道注意力模块 - 修复维度计算"""
    def __init__(self, degradation_dim, feature_dim):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 修复输入维度计算
        input_dim = degradation_dim + feature_dim * 2
        self.fc1 = nn.Linear(input_dim, max(input_dim // 4, 16))  # 确保最小维度
        self.fc2 = nn.Linear(max(input_dim // 4, 16), feature_dim)
        
    def forward(self, degradation, encoder_feat, decoder_feat):
        B = encoder_feat.shape[0]
        
        # 全局平均池化
        deg_pool = self.global_pool(degradation).view(B, -1)
        enc_pool = self.global_pool(encoder_feat).view(B, -1) 
        dec_pool = self.global_pool(decoder_feat).view(B, -1)
        
        # 拼接并生成通道注意力
        combined = torch.cat([deg_pool, enc_pool, dec_pool], dim=1)
        attention = torch.sigmoid(self.fc2(F.relu(self.fc1(combined))))
        
        return attention.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)


class CrossAttentionModule(nn.Module):
    """编码器-解码器交叉注意力模块"""
    def __init__(self, degradation_dim, feature_dim, rank):
        super().__init__()
        self.rank = rank
        self.query_proj = nn.Conv2d(feature_dim, rank, 1)
        self.key_proj = nn.Conv2d(feature_dim, rank, 1)
        self.value_proj = nn.Conv2d(feature_dim, rank, 1)
        self.degradation_modulation = nn.Conv2d(degradation_dim, rank, 1)
        
    def forward(self, degradation, encoder_feat, decoder_feat):
        B, C, H, W = encoder_feat.shape
        
        # 生成查询、键、值
        q = self.query_proj(encoder_feat)  # (B, rank, H, W)
        k = self.key_proj(decoder_feat)    # (B, rank, H, W)
        v = self.value_proj(decoder_feat)  # (B, rank, H, W)
        
        # 退化调制
        deg_mod = torch.sigmoid(self.degradation_modulation(degradation))
        
        # 计算注意力
        attention = F.softmax((q * k).sum(dim=1, keepdim=True) / (self.rank ** 0.5), dim=-1)
        attended_value = attention * v * deg_mod
        
        return attended_value


class FeatureEnhancementModule(nn.Module):
    """特征增强模块"""
    def __init__(self, feature_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, 1)
        self.norm = nn.LayerNorm([feature_dim])  # 使用LayerNorm避免batch维度问题
        
    def forward(self, x):
        residual = x
        B, C, H, W = x.shape
        
        # LayerNorm需要特殊处理
        x_norm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        x = F.relu(self.conv1(x_norm))
        x = self.conv2(x)
        return x + residual  # 残差连接


# 测试函数 - 修复
def test_enhanced_fusion():
    fusion = LowRankFusion(
        degradation_dim=64,
        feature_dim=128,
        rank=8,
        use_spatial_attention=True,
        use_channel_attention=True,
        use_cross_attention=True
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
    test_enhanced_fusion()
