from basicsr.utils.registry import ARCH_REGISTRY

# 清理可能的冲突
if 'NAF_Baseline' in ARCH_REGISTRY._obj_map:
    print(f"发现冲突的Baseline类: {ARCH_REGISTRY._obj_map['NAF_Baseline']}")
    del ARCH_REGISTRY._obj_map['NAF_Baseline']

# 然后再导入你的架构文件
from basicsr.archs.NAFnet_arch import NAF_Baseline
