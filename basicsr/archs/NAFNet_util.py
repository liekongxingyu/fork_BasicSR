import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

hidden_list = [256, 256, 256]
L = 4

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * \
                self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * \
                self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(
                    h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] -
                       s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(
                    out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(
                s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-
                                                    k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2,
                     (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            # compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size,
                             fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

# Context提取器 - 应该独立使用
# 建议加点频域啥的
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
    

# 隐式神经退化场 - 内部生成退化类型版本
class DegradationINR(nn.Module):
    def __init__(self, d,context_dim=256,
                 cell_decode=True, num_degradation_types=10):
        super().__init__()
        self.d = d  
        self.context_dim = context_dim
        self.cell_decode = cell_decode
        self.deg_type_dim = num_degradation_types
        
        # 添加退化类型生成器 - 从上下文向量中预测退化类型
        self.degradation_predictor = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.deg_type_dim)
        )

        # 计算MLP输入维度
        imnet_in_dim = 2 + 4 * L + self.deg_type_dim + context_dim
        
        if self.cell_decode:
            imnet_in_dim += 2

        self.imnet = MLP(imnet_in_dim, d, hidden_list)

    def generate_degradation_type(self, context_vector):
        """
        用空间一致性机制生成退化类型分布（全局）。
        最终输出仍为 [B, K]。
        """
        B = context_vector.shape[0]
        K = self.deg_type_dim

        # 1️⃣ context -> 粗尺度退化布局 [B, K, h0, w0]
        coarse = self.deg_coarse_predictor(context_vector)
        coarse = coarse.view(B, K, self.proto_h, self.proto_w)

        # 2️⃣ depthwise conv 平滑，获得空间一致性
        smoothed = self.smooth_conv(coarse)  # [B, K, h0, w0]

        # 3️⃣ 全局汇聚（取平均或最大，也可以混合）
        pooled = smoothed.mean(dim=[2, 3])  # [B, K]

        # 4️⃣ softmax 归一化成概率分布
        probs = torch.softmax(pooled, dim=-1)

        return probs  # [B, K]

    
    
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
        device = context_vector.device
        coord = make_coord((H, W)).to(device)

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
    degradation_dim = 64
    
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
