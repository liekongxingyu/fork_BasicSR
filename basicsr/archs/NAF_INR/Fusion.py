import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankFusion(nn.Module):
    def __init__(self, degradation_dim, feature_dim, rank=8):
        """
        三张量低秩融合模块
        Args:
            degradation_dim (int): 退化向量维度 D
            feature_dim (int): 特征向量维度 C  
            rank (int): 低秩分解的秩
        """
        super().__init__()
        self.D = degradation_dim
        self.C = feature_dim
        self.rank = rank

        # 生成编码器分支的变换矩阵
        self.mlp_U_e = nn.Linear(degradation_dim, feature_dim * rank)
        self.mlp_V_e = nn.Linear(degradation_dim, rank * feature_dim)

        # 生成解码器分支的变换矩阵
        self.mlp_U_d = nn.Linear(degradation_dim, feature_dim * rank)
        self.mlp_V_d = nn.Linear(degradation_dim, rank * feature_dim)

        # 学习融合权重的MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(degradation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出α, β, γ三个权重
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, degradation, encoder_feat, decoder_feat):
        """
        前向传播
        Args:
            degradation: (B, D, H, W) 退化信息
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

        # 展平所有张量到 (B*H*W, dim) 格式
        deg_flat = degradation.permute(0, 2, 3, 1).contiguous().view(B*H*W, D)
        enc_flat = encoder_feat.permute(
            0, 2, 3, 1).contiguous().view(B*H*W, C, 1)
        dec_flat = decoder_feat.permute(
            0, 2, 3, 1).contiguous().view(B*H*W, C, 1)

        # 生成编码器分支的变换矩阵
        U_e_params = self.mlp_U_e(deg_flat)  # (B*H*W, C*rank)
        V_e_params = self.mlp_V_e(deg_flat)  # (B*H*W, rank*C)

        U_e = U_e_params.view(B*H*W, self.C, self.rank)     # (B*H*W, C, rank)
        V_e = V_e_params.view(B*H*W, self.rank, self.C)     # (B*H*W, rank, C)

        # 生成解码器分支的变换矩阵
        U_d_params = self.mlp_U_d(deg_flat)  # (B*H*W, C*rank)
        V_d_params = self.mlp_V_d(deg_flat)  # (B*H*W, rank*C)

        U_d = U_d_params.view(B*H*W, self.C, self.rank)     # (B*H*W, C, rank)
        V_d = V_d_params.view(B*H*W, self.rank, self.C)     # (B*H*W, rank, C)

        # 编码器分支的低秩变换：y_e = U_e × (V_e × x_e)
        z_e = torch.bmm(V_e, enc_flat)        # (B*H*W, rank, 1)
        y_e = torch.bmm(U_e, z_e).squeeze(-1)  # (B*H*W, C)

        # 解码器分支的低秩变换：y_d = U_d × (V_d × x_d)
        z_d = torch.bmm(V_d, dec_flat)        # (B*H*W, rank, 1)
        y_d = torch.bmm(U_d, z_d).squeeze(-1)  # (B*H*W, C)

        # 学习融合权重
        fusion_weights = self.fusion_mlp(deg_flat)              # (B*H*W, 3)
        fusion_weights = F.softmax(fusion_weights, dim=-1)      # 归一化权重

        α = fusion_weights[:, 0:1]  # (B*H*W, 1)
        β = fusion_weights[:, 1:2]  # (B*H*W, 1)
        γ = fusion_weights[:, 2:3]  # (B*H*W, 1)

        # 最终融合：y = α × y_e + β × y_d + γ × (y_e ⊙ y_d)
        result = α * y_e + β * y_d + γ * (y_e * y_d)  # (B*H*W, C)

        # 重塑回原始特征图形状
        result = result.view(B, H, W, self.C).permute(
            0, 3, 1, 2)  # (B, C, H, W)

        return result

# 修复后的测试函数


def test_triple_fusion():
    """测试函数"""
    # 设置参数
    B, D, C, H, W = 2, 16, 64, 32, 32
    rank = 8

    # 创建模块
    fusion_module = LowRankFusion(
        degradation_dim=D,
        feature_dim=C,
        rank=rank
    )

    # 创建测试输入 - 注意：需要 requires_grad=True 以支持梯度计算
    degradation = torch.randn(B, D, H, W, requires_grad=True)
    encoder_feat = torch.randn(B, C, H, W, requires_grad=True)
    decoder_feat = torch.randn(B, C, H, W, requires_grad=True)

    print("输入形状:")
    print(f"  degradation: {degradation.shape}")
    print(f"  encoder_feat: {encoder_feat.shape}")
    print(f"  decoder_feat: {decoder_feat.shape}")

    # 前向传播 - 移除 torch.no_grad()
    result = fusion_module(degradation, encoder_feat, decoder_feat)

    print(f"\n输出形状: {result.shape}")
    print(f"参数量: {sum(p.numel() for p in fusion_module.parameters()):,}")

    # 测试梯度传播
    loss = result.sum()
    loss.backward()
    print("梯度传播测试: 通过")

    # 检查梯度是否正确计算
    param_with_grad = 0
    for name, param in fusion_module.named_parameters():
        if param.grad is not None:
            param_with_grad += 1
        else:
            print(f"警告: 参数 {name} 没有梯度")

    print(
        f"有梯度的参数数量: {param_with_grad}/{len(list(fusion_module.parameters()))}")

    return fusion_module, result

# 实际使用示例


def example_usage():
    """实际使用的示例"""
    # 假设这些是从网络其他部分得到的特征
    batch_size = 4
    degradation_dim = 32
    feature_dim = 128
    height, width = 64, 64

    # 创建融合模块
    fusion = LowRankFusion(
        degradation_dim=degradation_dim,
        feature_dim=feature_dim,
        rank=16
    )

    # 模拟网络中的实际数据流
    degradation_info = torch.randn(batch_size, degradation_dim, height, width)
    encoder_features = torch.randn(batch_size, feature_dim, height, width)
    decoder_features = torch.randn(batch_size, feature_dim, height, width)

    # 融合
    fused_features = fusion(
        degradation_info, encoder_features, decoder_features)

    print(f"融合后特征形状: {fused_features.shape}")
    return fused_features

# 内存优化版本（适用于大尺寸图像）


class MemoryEfficientTripleFusion(LowRankFusion):
    def __init__(self, degradation_dim, feature_dim, rank=8, chunk_size=1024):
        super().__init__(degradation_dim, feature_dim, rank)
        self.chunk_size = chunk_size

    def forward(self, degradation, encoder_feat, decoder_feat):
        """内存优化的前向传播"""
        B, D, H, W = degradation.shape
        B, C, H, W = encoder_feat.shape
        total_positions = H * W

        # 展平输入
        deg_flat = degradation.permute(0, 2, 3, 1).contiguous().view(B*H*W, D)
        enc_flat = encoder_feat.permute(0, 2, 3, 1).contiguous().view(B*H*W, C)
        dec_flat = decoder_feat.permute(0, 2, 3, 1).contiguous().view(B*H*W, C)

        results = []

        # 分块处理
        for start_idx in range(0, B*H*W, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, B*H*W)

            # 当前块的数据
            deg_chunk = deg_flat[start_idx:end_idx]      # (chunk_size, D)
            # (chunk_size, C, 1)
            enc_chunk = enc_flat[start_idx:end_idx].unsqueeze(-1)
            # (chunk_size, C, 1)
            dec_chunk = dec_flat[start_idx:end_idx].unsqueeze(-1)

            # 处理当前块（与原始方法相同）
            chunk_result = self._process_chunk(deg_chunk, enc_chunk, dec_chunk)
            results.append(chunk_result)

        # 合并结果
        result = torch.cat(results, dim=0)  # (B*H*W, C)
        result = result.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return result

    def _process_chunk(self, deg_chunk, enc_chunk, dec_chunk):
        """处理单个数据块"""
        chunk_size = deg_chunk.shape[0]

        # 生成变换矩阵
        U_e = self.mlp_U_e(deg_chunk).view(chunk_size, self.C, self.rank)
        V_e = self.mlp_V_e(deg_chunk).view(chunk_size, self.rank, self.C)
        U_d = self.mlp_U_d(deg_chunk).view(chunk_size, self.C, self.rank)
        V_d = self.mlp_V_d(deg_chunk).view(chunk_size, self.rank, self.C)

        # 低秩变换
        z_e = torch.bmm(V_e, enc_chunk)
        y_e = torch.bmm(U_e, z_e).squeeze(-1)
        z_d = torch.bmm(V_d, dec_chunk)
        y_d = torch.bmm(U_d, z_d).squeeze(-1)

        # 融合权重
        fusion_weights = F.softmax(self.fusion_mlp(deg_chunk), dim=-1)
        α, β, γ = fusion_weights[:, 0:1], fusion_weights[:,
                                                         1:2], fusion_weights[:, 2:3]

        # 融合
        result = α * y_e + β * y_d + γ * (y_e * y_d)
        return result


if __name__ == "__main__":
    print("=== 基本测试 ===")
    try:
        fusion_module, result = test_triple_fusion()
        print("基本测试通过!")
    except Exception as e:
        print(f"基本测试失败: {e}")

    print("\n=== 实际使用示例 ===")
    try:
        fused_result = example_usage()
        print("实际使用示例通过!")
    except Exception as e:
        print(f"实际使用示例失败: {e}")

    print("\n测试完成!")
