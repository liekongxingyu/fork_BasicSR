import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankFusion(nn.Module):
    def __init__(self, degradation_dim, feature_dim, rank=8):
        """
        双向低秩融合模块（稳定性优化版本）
        Args:
            degradation_dim (int): 退化向量维度 D
            feature_dim (int): 特征向量维度 C  
            rank (int): 低秩分解的秩 r
        """
        super().__init__()
        self.D = degradation_dim
        self.C = feature_dim
        self.rank = rank

        # 使用 1×1 卷积代替全连接层，保持逐像素线性映射
        self.encoder_U = nn.Conv2d(degradation_dim, rank * feature_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.encoder_V = nn.Conv2d(degradation_dim, feature_dim * rank, kernel_size=1, stride=1, padding=0, bias=True)

        self.decoder_U = nn.Conv2d(degradation_dim, rank * feature_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.decoder_V = nn.Conv2d(degradation_dim, feature_dim * rank, kernel_size=1, stride=1, padding=0, bias=True)

        # # 编码器退化图分解
        # self.encoder_U = nn.Linear(degradation_dim, rank * feature_dim)  # U_e: D -> r*C
        # self.encoder_V = nn.Linear(degradation_dim, feature_dim * rank)  # V_e: D -> C*r

        # # 解码器退化图分解  
        # self.decoder_U = nn.Linear(degradation_dim, rank * feature_dim)  # U_d: D -> r*C
        # self.decoder_V = nn.Linear(degradation_dim, feature_dim * rank)  # V_d: D -> C*r

        # 稳定性优化：减小MLP复杂度，防止过拟合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(degradation_dim * 2, 128),  # 减小隐层维度
            nn.LayerNorm(128),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(0.1),    # 添加dropout防止过拟合
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)    # 输出α, β, γ三个权重
        )

        # 添加特征归一化层
        self.enc_norm = nn.LayerNorm(feature_dim)
        self.dec_norm = nn.LayerNorm(feature_dim)
        self.result_norm = nn.LayerNorm(feature_dim)

        self._init_weights()

    def _init_weights(self):
        """保守的权重初始化，防止梯度爆炸"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 对低秩分解层使用更小的初始化
                if 'encoder_U' in name or 'encoder_V' in name or 'decoder_U' in name or 'decoder_V' in name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)  # 非常保守的初始化
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 稍大一点但仍然保守
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, encoder_feat, encoder_deg, decoder_feat, decoder_deg):
        """
        前向传播（稳定性优化版本）
        """
        B, C, H, W = encoder_feat.shape
        B, D, H, W = encoder_deg.shape

        # 检查维度匹配
        assert decoder_feat.shape == (B, C, H, W), f"解码器特征维度不匹配: {decoder_feat.shape}"
        assert decoder_deg.shape == (B, D, H, W), f"解码器退化图维度不匹配: {decoder_deg.shape}"

        # 展平到 (B*H*W, dim) 格式
        enc_feat_flat = encoder_feat.permute(0, 2, 3, 1).contiguous().view(B*H*W, C)  # (BHW, C)
        enc_deg_flat = encoder_deg.permute(0, 2, 3, 1).contiguous().view(B*H*W, D)   # (BHW, D)
        dec_feat_flat = decoder_feat.permute(0, 2, 3, 1).contiguous().view(B*H*W, C) # (BHW, C)
        dec_deg_flat = decoder_deg.permute(0, 2, 3, 1).contiguous().view(B*H*W, D)   # (BHW, D)

        # 输入特征归一化，提高数值稳定性
        enc_feat_flat = self.enc_norm(enc_feat_flat)
        dec_feat_flat = self.dec_norm(dec_feat_flat)

        # 编码器退化图低秩分解 + 数值裁剪
        U_e_params = torch.clamp(self.encoder_U(enc_deg_flat), -5.0, 5.0)  # 裁剪防止爆炸
        V_e_params = torch.clamp(self.encoder_V(enc_deg_flat), -5.0, 5.0)
        
        U_e = U_e_params.view(B*H*W, self.rank, self.C)      # (BHW, r, C)
        V_e = V_e_params.view(B*H*W, self.C, self.rank)      # (BHW, C, r)

        # 解码器退化图低秩分解 + 数值裁剪
        U_d_params = torch.clamp(self.decoder_U(dec_deg_flat), -5.0, 5.0)
        V_d_params = torch.clamp(self.decoder_V(dec_deg_flat), -5.0, 5.0)
        
        U_d = U_d_params.view(B*H*W, self.rank, self.C)      # (BHW, r, C)
        V_d = V_d_params.view(B*H*W, self.C, self.rank)      # (BHW, C, r)

        # 编码器分支：F_e' = U_e × V_e × F_e（添加数值稳定性保护）
        enc_feat_expanded = enc_feat_flat.unsqueeze(-1)  # (BHW, C, 1)
        temp_e = torch.clamp(torch.bmm(V_e.transpose(-2, -1), enc_feat_expanded), -10.0, 10.0)  # (BHW, r, 1)
        transformed_enc = torch.clamp(torch.bmm(U_e.transpose(-2, -1), temp_e).squeeze(-1), -10.0, 10.0)  # (BHW, C)

        # 解码器分支：F_d' = U_d × V_d × F_d（添加数值稳定性保护）
        dec_feat_expanded = dec_feat_flat.unsqueeze(-1)  # (BHW, C, 1)
        temp_d = torch.clamp(torch.bmm(V_d.transpose(-2, -1), dec_feat_expanded), -10.0, 10.0)  # (BHW, r, 1)
        transformed_dec = torch.clamp(torch.bmm(U_d.transpose(-2, -1), temp_d).squeeze(-1), -10.0, 10.0)  # (BHW, C)

        # 变换后特征归一化
        transformed_enc = F.layer_norm(transformed_enc, [self.C])
        transformed_dec = F.layer_norm(transformed_dec, [self.C])

        # 自适应融合策略（添加输入裁剪）
        combined_deg = torch.clamp(torch.cat([enc_deg_flat, dec_deg_flat], dim=-1), -5.0, 5.0)  # (BHW, 2*D)
        fusion_weights = self.fusion_mlp(combined_deg)                  # (BHW, 3)
        fusion_weights = F.softmax(fusion_weights, dim=-1)             # 归一化权重

        α = fusion_weights[:, 0:1]  # (BHW, 1) - 编码器权重
        β = fusion_weights[:, 1:2]  # (BHW, 1) - 解码器权重  
        γ = fusion_weights[:, 2:3]  # (BHW, 1) - 交互权重

        # 最终融合：使用更稳定的交互项
        interaction_term = transformed_enc * transformed_dec  # 元素乘积比tanh(相加)更稳定
        result = α * transformed_enc + β * transformed_dec + γ * interaction_term  # (BHW, C)

        # 最终结果归一化和裁剪
        result = self.result_norm(result)
        result = torch.clamp(result, -10.0, 10.0)

        # 重塑回原始特征图形状
        result = result.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return result
    
    # 学习版
    def forward(self, encoder_feat, encoder_deg, decoder_feat, decoder_deg):
        """
        前向传播（简化版，无数值稳定性修饰）
        """
        B, C, H, W = encoder_feat.shape
        B, D, H, W = encoder_deg.shape

        # 检查维度匹配
        assert decoder_feat.shape == (B, C, H, W), f"解码器特征维度不匹配: {decoder_feat.shape}"
        assert decoder_deg.shape == (B, D, H, W), f"解码器退化图维度不匹配: {decoder_deg.shape}"

        # 展平到 (B*H*W, dim) 格式
        enc_feat_flat = encoder_feat.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)  # (BHW, C)
        enc_deg_flat = encoder_deg.permute(0, 2, 3, 1).contiguous().view(B * H * W, D)   # (BHW, D)
        dec_feat_flat = decoder_feat.permute(0, 2, 3, 1).contiguous().view(B * H * W, C) # (BHW, C)
        dec_deg_flat = decoder_deg.permute(0, 2, 3, 1).contiguous().view(B * H * W, D)   # (BHW, D)

        # 编码器退化图低秩分解
        U_e_params = self.encoder_U(enc_deg_flat)  # (BHW, r*C)
        V_e_params = self.encoder_V(enc_deg_flat)  # (BHW, C*r)
        U_e = U_e_params.view(B * H * W, self.rank, self.C)  # (BHW, r, C)
        V_e = V_e_params.view(B * H * W, self.C, self.rank)  # (BHW, C, r)

        # 解码器退化图低秩分解
        U_d_params = self.decoder_U(dec_deg_flat)  # (BHW, r*C)
        V_d_params = self.decoder_V(dec_deg_flat)  # (BHW, C*r)
        U_d = U_d_params.view(B * H * W, self.rank, self.C)  # (BHW, r, C)
        V_d = V_d_params.view(B * H * W, self.C, self.rank)  # (BHW, C, r)

        # 编码器分支：F_e' = U_e × V_e × F_e
        enc_feat_expanded = enc_feat_flat.unsqueeze(-1)               # (BHW, C, 1)
        temp_e = torch.bmm(V_e.transpose(-2, -1), enc_feat_expanded)  # (BHW, r, 1)
        transformed_enc = torch.bmm(U_e.transpose(-2, -1), temp_e).squeeze(-1)  # (BHW, C)

        # 解码器分支：F_d' = U_d × V_d × F_d
        dec_feat_expanded = dec_feat_flat.unsqueeze(-1)               # (BHW, C, 1)
        temp_d = torch.bmm(V_d.transpose(-2, -1), dec_feat_expanded)  # (BHW, r, 1)
        transformed_dec = torch.bmm(U_d.transpose(-2, -1), temp_d).squeeze(-1)  # (BHW, C)

        # 自适应融合策略
        combined_deg = torch.cat([enc_deg_flat, dec_deg_flat], dim=-1)  # (BHW, 2*D)
        fusion_weights = self.fusion_mlp(combined_deg)                  # (BHW, 3)
        fusion_weights = F.softmax(fusion_weights, dim=-1)              # (BHW, 3)

        α = fusion_weights[:, 0:1]  # (BHW, 1)
        β = fusion_weights[:, 1:2]  # (BHW, 1)
        γ = fusion_weights[:, 2:3]  # (BHW, 1)

        # 最终融合
        interaction_term = transformed_enc * transformed_dec
        result = α * transformed_enc + β * transformed_dec + γ * interaction_term  # (BHW, C)

        # 重塑回原始特征图形状
        result = result.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return result



def test_low_rank_fusion():
    """
    测试 LowRankFusion 模块的功能
    """
    print("=== 低秩融合模块测试 ===")
    
    # 设置测试参数
    B, C, H, W = 2, 64, 32, 32  # batch_size=2, channels=64, height=32, width=32
    D = 16  # 退化向量维度
    rank = 8  # 低秩分解的秩
    
    # 创建测试数据
    encoder_feat = torch.randn(B, C, H, W)  # 编码器特征图
    encoder_deg = torch.randn(B, D, H, W)   # 编码器退化图
    decoder_feat = torch.randn(B, C, H, W)  # 解码器特征图  
    decoder_deg = torch.randn(B, D, H, W)   # 解码器退化图
    
    print(f"输入张量尺寸:")
    print(f"  编码器特征: {encoder_feat.shape}")
    print(f"  编码器退化: {encoder_deg.shape}")
    print(f"  解码器特征: {decoder_feat.shape}")
    print(f"  解码器退化: {decoder_deg.shape}")
    
    # 创建模型
    model = LowRankFusion(
        degradation_dim=D,
        feature_dim=C,
        rank=rank
    )
    
    print(f"\n模型参数:")
    print(f"  退化维度: {D}")
    print(f"  特征维度: {C}")
    print(f"  低秩分解秩: {rank}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    
    # 前向传播测试
    print(f"\n=== 前向传播测试 ===")
    try:
        with torch.no_grad():
            output = model(encoder_feat, encoder_deg, decoder_feat, decoder_deg)
        
        print(f"✓ 前向传播成功!")
        print(f"  输出张量尺寸: {output.shape}")
        print(f"  输出数值范围: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  输出均值: {output.mean():.4f}")
        print(f"  输出标准差: {output.std():.4f}")
        
        # 检查输出形状是否正确
        assert output.shape == (B, C, H, W), f"输出形状错误: {output.shape}"
        print(f"✓ 输出形状验证通过!")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    # 梯度测试
    print(f"\n=== 梯度测试 ===")
    try:
        # 设置requires_grad
        encoder_feat.requires_grad_(True)
        decoder_feat.requires_grad_(True)
        
        output = model(encoder_feat, encoder_deg, decoder_feat, decoder_deg)
        loss = output.mean()  # 简单的损失函数
        loss.backward()
        
        # 检查梯度
        enc_grad_norm = encoder_feat.grad.norm()
        dec_grad_norm = decoder_feat.grad.norm()
        
        print(f"✓ 反向传播成功!")
        print(f"  编码器特征梯度范数: {enc_grad_norm:.6f}")
        print(f"  解码器特征梯度范数: {dec_grad_norm:.6f}")
        
        
    except Exception as e:
        print(f"✗ 梯度测试失败: {e}")
        return False
    
    # 不同输入尺寸测试
    print(f"\n=== 不同输入尺寸测试 ===")
    test_sizes = [(1, 16, 16), (4, 64, 64), (1, 8, 8)]
    
    for test_b, test_h, test_w in test_sizes:
        try:
            test_enc_feat = torch.randn(test_b, C, test_h, test_w)
            test_enc_deg = torch.randn(test_b, D, test_h, test_w)
            test_dec_feat = torch.randn(test_b, C, test_h, test_w)
            test_dec_deg = torch.randn(test_b, D, test_h, test_w)
            
            with torch.no_grad():
                test_output = model(test_enc_feat, test_enc_deg, test_dec_feat, test_dec_deg)
            
            assert test_output.shape == (test_b, C, test_h, test_w)
            print(f"✓ 尺寸 ({test_b}, {C}, {test_h}, {test_w}) 测试通过")
            
        except Exception as e:
            print(f"✗ 尺寸 ({test_b}, {C}, {test_h}, {test_w}) 测试失败: {e}")
            return False
    
    print(f"\n=== 所有测试通过! ===")
    return True

# 运行测试
if __name__ == "__main__":
    success = test_low_rank_fusion()
    if success:
        print("🎉 LowRankFusion 模块工作正常!")
    else:
        print("❌ 测试发现问题，请检查代码!")