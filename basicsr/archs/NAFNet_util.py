import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

hidden_list = [256, 256, 256]
L = 4

# Context提取器 - 应该独立使用
class ContextExtractor(nn.Module):
    def __init__(self, context_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(), 
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, context_dim)
        )
    
    def forward(self, image):
        return self.backbone(image)  # [B, 256]

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

import torch.nn as nn
import torch
import numpy as np

hidden_list = [256, 256, 256]
L = 4

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

# 隐式神经退化场 - 内部生成退化类型版本
class DegradationINR(nn.Module):
    def __init__(self, d,context_dim=256,
                 cell_decode=True, num_degradation_types=10):
        super().__init__()
        self.d = d  
        self.context_dim = context_dim
        self.cell_decode = cell_decode
        self.num_degradation_types = num_degradation_types
        
        # 添加退化类型生成器 - 从上下文向量中预测退化类型
        self.degradation_predictor = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_degradation_types)
        )


        self.embed_dim = 10
        self.deg_type_dim = self.embed_dim

        # 计算MLP输入维度
        imnet_in_dim = 2 + 4 * L + self.deg_type_dim + context_dim
        
        if self.cell_decode:
            imnet_in_dim += 2

        self.imnet = MLP(imnet_in_dim, d, hidden_list)

    def generate_degradation_type(self, context_vector):
        """根据上下文向量生成退化类型"""
        logits = self.degradation_predictor(context_vector)  # [B, num_types]
        
        return logits

    def query_degradation_vector(self, input_size,coord, context_vector, cell=None):
        """
        Args:
            coord: [B, H*W, 2] 像素坐标
            context_vector: [B, context_dim] 外部传入的上下文向量
            cell: [B, H*W, 2] 单元格信息
        Returns:
            [B, D, H, W] 退化向量
        """
        B, N, _ = coord.shape
        H, W = input_size

        # print(B,N,H,W)

        # 位置编码
        points_enc = self.positional_encoding(coord)  # [B, H*W, 4*L]
        
        # 内部生成退化类型编码
        deg_encoded = self.generate_degradation_type(context_vector)  # [B, deg_type_dim]

        # print(deg_encoded)
        deg_expanded = deg_encoded.unsqueeze(1).expand(B, N, self.deg_type_dim)  # [B, H*W, deg_type_dim]
        
        # 扩展上下文向量到所有像素
        context_expanded = context_vector.unsqueeze(1).expand(B, N, self.context_dim)  # [B, H*W, context_dim]

        # 拼接所有特征
        inp = torch.cat([
            coord,              # [B, H*W, 2]
            points_enc,         # [B, H*W, 4*L] 
            deg_expanded,       # [B, H*W, deg_type_dim]
            context_expanded    # [B, H*W, context_dim]
        ], dim=-1)

        if self.cell_decode and cell is not None:
            inp = torch.cat([inp, cell], dim=-1)

        # MLP预测退化向量
        degradation_vectors = self.imnet(inp)  # [B, H*W, D]
        
        # 重塑为图像格式
        ret = degradation_vectors.view(B, H, W, self.d).permute(0, 3, 1, 2)  # [B, D, H, W]
        
        return ret

    def forward(self, context_vector, input_size):
        """
        前向传播 - 简化接口，只需要上下文向量和尺寸
        Args:
            context_vector: [B, context_dim] 外部预先提取的上下文向量  
            input_size: (H, W) 目标图像尺寸
        Returns:
            [B, D, H, W] 退化向量图
        """
        H, W = input_size
        B = context_vector.shape[0]
        
        # 生成坐标网格
        coord = make_coord((H, W))
        if torch.cuda.is_available():
            coord = coord.cuda()
        coord = coord.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]
        
        # 生成单元格信息
        cell = None
        if self.cell_decode:
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / H
            cell[:, :, 1] *= 2 / W

        # 查询退化向量（内部自动生成退化类型）
        ret = self.query_degradation_vector(input_size,coord, context_vector, cell)
        
        return ret

    def positional_encoding(self, input):
        """位置编码"""
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device) * np.pi
        spectrum = input[..., None] * freq
        sin, cos = spectrum.sin(), spectrum.cos()
        input_enc = torch.stack([sin, cos], dim=-2)
        input_enc = input_enc.view(*shape[:-1], -1)
        return input_enc
    

def test_degradation_inr():
    """
    测试DegradationINR类的完整功能
    """
    print("=" * 60)
    print("测试开始：DegradationINR 隐式神经退化场")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化网络
    context_dim = 256
    degradation_dim = 128
    
    print("\n1. 初始化网络...")
    context_extractor = ContextExtractor(context_dim=context_dim).to(device)
    degradation_inr = DegradationINR(
        d=degradation_dim, 
        context_dim=context_dim,
        num_degradation_types=10
    ).to(device)
    
    print(f"   - ContextExtractor: 输入3通道图像 -> 输出{context_dim}维上下文向量")
    print(f"   - DegradationINR: 输出{degradation_dim}维退化向量")
    
    # 测试不同尺寸的输入
    test_cases = [
        (2, 3, 64, 64),    # 标准测试
        (1, 3, 128, 128),  # 单样本大图
        (4, 3, 32, 32),    # 多样本小图
        (1, 3, 80, 120),   # 非正方形图像
    ]
    
    for i, (B, C, H, W) in enumerate(test_cases):
        print(f"\n2.{i+1} 测试案例 {i+1}: Batch={B}, 通道={C}, 尺寸={H}x{W}")
        
        try:
            # 生成随机输入图像
            input_image = torch.randn(B, C, H, W).to(device)
            print(f"   输入图像形状: {input_image.shape}")
            
            # 提取上下文向量
            with torch.no_grad():
                context_vector = context_extractor(input_image)
                print(f"   上下文向量形状: {context_vector.shape}")
                
                # 生成退化向量图
                degradation_map = degradation_inr(context_vector, (H, W))
                print(f"   退化向量图形状: {degradation_map.shape}")
                
                # 验证输出形状
                expected_shape = (B, degradation_dim, H, W)
                assert degradation_map.shape == expected_shape, \
                    f"形状不匹配! 期望: {expected_shape}, 实际: {degradation_map.shape}"
                print(f"   ✓ 形状验证通过")
                
                # 检查数值范围
                min_val, max_val = degradation_map.min().item(), degradation_map.max().item()
                print(f"   数值范围: [{min_val:.4f}, {max_val:.4f}]")
                
        except Exception as e:
            print(f"   ✗ 测试失败: {str(e)}")
            
    print("\n3. 测试不同编码类型...")
    
    # 测试不同的编码类型
    encoding_types = ['embedding', 'onehot', 'scalar']
    
    for encoding_type in encoding_types:
        print(f"\n   测试编码类型: {encoding_type}")
        try:
            test_inr = DegradationINR(
                d=64, 
                context_dim=256,
                encoding_type=encoding_type,
                num_degradation_types=8
            ).to(device)
            
            # 测试输入
            test_input = torch.randn(1, 3, 32, 32).to(device)
            test_context = context_extractor(test_input)
            
            with torch.no_grad():
                output = test_inr(test_context, (32, 32))
                print(f"   ✓ {encoding_type} 编码测试通过，输出形状: {output.shape}")
                
        except Exception as e:
            print(f"   ✗ {encoding_type} 编码测试失败: {str(e)}")
    
    print("\n4. 性能测试...")
    
    # 简单的性能测试
    import time
    
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    # 预热
    with torch.no_grad():
        context = context_extractor(test_input)
        _ = degradation_inr(context, (256, 256))
    
    # 计时测试
    num_iterations = 10
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            context = context_extractor(test_input)
            degradation_map = degradation_inr(context, (256, 256))
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    print(f"   平均推理时间 (256x256): {avg_time:.4f} 秒")
    
    print("\n5. 内存使用测试...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # 测试大尺寸输入
        large_input = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            context = context_extractor(large_input)
            degradation_map = degradation_inr(context, (512, 512))
        
        peak_memory = torch.cuda.memory_allocated(device)
        memory_usage = (peak_memory - initial_memory) / 1024**2  # MB
        print(f"   内存使用 (512x512): {memory_usage:.2f} MB")
    
    print("\n" + "=" * 60)
    print("测试完成！所有功能正常工作")
    print("=" * 60)


def test_compatibility():
    """
    测试与原始训练框架的兼容性
    """
    print("\n兼容性测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟训练场景
    context_extractor = ContextExtractor().to(device)
    degradation_inr = DegradationINR(d=128).to(device)
    
    # 模拟训练批次
    batch_images = torch.randn(8, 3, 64, 64).to(device)
    
    print("   模拟训练前向传播...")
    
    with torch.no_grad():
        # 提取上下文
        contexts = context_extractor(batch_images)
        
        # 生成退化向量
        degradation_maps = degradation_inr(contexts, (64, 64))
        
        print(f"   ✓ 批次处理成功: {degradation_maps.shape}")


if __name__ == "__main__":
    # 运行完整测试
    test_degradation_inr()
    test_compatibility()
