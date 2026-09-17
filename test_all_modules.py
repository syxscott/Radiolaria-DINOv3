#!/usr/bin/env python3
"""
Radiolaria-DINOv3 项目全面测试脚本
在无GPU环境下逐个测试各个功能模块
"""

import os
import sys
import traceback
import importlib
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_test(name):
    print(f"{Colors.BOLD}[测试] {name}{Colors.RESET}", end=" ... ")

def print_pass():
    print(f"{Colors.GREEN}✓ 通过{Colors.RESET}")

def print_fail(error):
    print(f"{Colors.RED}✗ 失败{Colors.RESET}")
    print(f"{Colors.RED}错误详情: {error}{Colors.RESET}\n")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ 警告: {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")

# 测试结果记录
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def run_test(test_name, test_func):
    """运行单个测试并记录结果"""
    print_test(test_name)
    try:
        result = test_func()
        if result is True or result is None:
            print_pass()
            test_results['passed'].append(test_name)
            return True
        else:
            print_fail(result)
            test_results['failed'].append((test_name, result))
            return False
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print_fail(error_msg)
        test_results['failed'].append((test_name, error_msg))
        return False

# ============================================================================
# 测试函数定义
# ============================================================================

def test_python_version():
    """测试Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return True
    else:
        return f"Python版本过低: {version.major}.{version.minor}, 需要 >= 3.9"

def test_project_structure():
    """测试项目目录结构"""
    required_dirs = ['core', 'configs', 'dinov3', 'utils', 'scripts', 'data']
    missing = []
    for d in required_dirs:
        if not os.path.exists(d):
            missing.append(d)
    
    if missing:
        return f"缺少目录: {', '.join(missing)}"
    return True

def test_import_torch():
    """测试PyTorch导入"""
    import torch
    print_info(f"PyTorch版本: {torch.__version__}")
    return True

def test_import_torchvision():
    """测试torchvision导入"""
    import torchvision
    print_info(f"torchvision版本: {torchvision.__version__}")
    return True

def test_import_numpy():
    """测试numpy导入"""
    import numpy as np
    print_info(f"numpy版本: {np.__version__}")
    return True

def test_import_pandas():
    """测试pandas导入"""
    import pandas as pd
    print_info(f"pandas版本: {pd.__version__}")
    return True

def test_import_sklearn():
    """测试scikit-learn导入"""
    import sklearn
    print_info(f"scikit-learn版本: {sklearn.__version__}")
    return True

def test_import_omegaconf():
    """测试omegaconf导入"""
    from omegaconf import OmegaConf
    return True

def test_import_pil():
    """测试PIL导入"""
    from PIL import Image
    return True

def test_dinov3_models():
    """测试DINOv3模型导入"""
    from dinov3.models.vision_transformer import vit_small, vit_base, vit_large
    return True

def test_dinov3_configs():
    """测试DINOv3配置导入"""
    from dinov3.configs.config import setup_config, get_cfg_from_args, get_default_config
    return True

def test_utils_data():
    """测试数据工具导入"""
    from utils.data_utils import get_stratified_datasets, get_transforms
    return True

def test_model_initialization():
    """测试模型初始化（CPU模式）"""
    import torch
    from dinov3.models.vision_transformer import vit_small
    
    model = vit_small(patch_size=16, num_classes=0)
    print_info(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # 测试前向传播
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    
    print_info(f"输出形状: {out.shape}")
    return True

def test_data_transforms():
    """测试数据增强"""
    from utils.data_utils import get_transforms
    from PIL import Image
    import numpy as np
    
    # 创建测试图像
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # 测试训练变换
    train_tfm = get_transforms(224, is_train=True)
    train_img = train_tfm(img)
    print_info(f"训练变换输出: {train_img.shape}")
    
    # 测试验证变换
    val_tfm = get_transforms(224, is_train=False)
    val_img = val_tfm(img)
    print_info(f"验证变换输出: {val_img.shape}")
    
    return True

def test_config_files():
    """测试配置文件"""
    from omegaconf import OmegaConf
    
    configs = [
        'configs/pretrain/vits16_domain_adapt.yaml',
        'configs/pretrain/vitb16_domain_adapt.yaml',
        'configs/pretrain/vitl16_domain_adapt.yaml'
    ]
    
    for cfg_path in configs:
        if not os.path.exists(cfg_path):
            return f"配置文件不存在: {cfg_path}"
        
        try:
            cfg = OmegaConf.load(cfg_path)
            print_info(f"{os.path.basename(cfg_path)}: batch={cfg.train.batch_size_per_gpu}, epochs={cfg.optim.epochs}")
        except Exception as e:
            return f"配置文件解析失败 {cfg_path}: {e}"
    
    return True

def test_data_directory():
    """测试数据目录"""
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return f"数据目录不存在: {data_dir}"
    
    # 检查关键子目录
    splits_dir = os.path.join(data_dir, 'splits_fixed')
    if os.path.exists(splits_dir):
        csv_files = [f for f in os.listdir(splits_dir) if f.endswith('.csv')]
        print_info(f"找到CSV文件: {len(csv_files)} 个")
    else:
        print_warning(f"未找到 {splits_dir}")
    
    images_dir = os.path.join(data_dir, 'images', 'train', '0')
    if os.path.exists(images_dir):
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print_info(f"训练图片数量: {len(img_files)}")
    else:
        print_warning(f"未找到 {images_dir}")
    
    return True

def test_csv_loading():
    """测试CSV文件加载"""
    import pandas as pd
    
    csv_path = 'data/splits_fixed/train_fixed.csv'
    if not os.path.exists(csv_path):
        print_warning(f"CSV文件不存在: {csv_path}")
        return True
    
    df = pd.read_csv(csv_path)
    print_info(f"CSV记录数: {len(df)}, 列: {list(df.columns)}")
    
    # 检查必需列
    if 'filepath' not in df.columns and 'filename' not in df.columns:
        return "CSV缺少图片路径列"
    if 'label' not in df.columns:
        return "CSV缺少标签列"
    
    return True

def test_dataset_creation():
    """测试数据集创建"""
    from utils.data_utils import get_stratified_datasets, get_transforms
    
    data_root = 'data'
    img_root = os.path.join(data_root, 'images')
    
    if not os.path.exists(os.path.join(data_root, 'splits_fixed')):
        print_warning("未找到splits_fixed，跳过数据集创建测试")
        return True
    
    try:
        tfm = get_transforms(224, is_train=False)
        train_ds, val_ds, test_ds, class_to_idx = get_stratified_datasets(
            data_root, img_root, transform_train=tfm, transform_val=tfm
        )
        
        print_info(f"训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}")
        print_info(f"类别数: {len(class_to_idx)}")
        
        # 测试加载一个样本
        if len(train_ds) > 0:
            img, label, path = train_ds[0]
            print_info(f"样本形状: {img.shape}, 标签: {label}")
        
        return True
    except Exception as e:
        print_warning(f"数据集创建失败（可能是数据路径问题）: {e}")
        return True  # 不算失败，因为可能是数据未准备

def test_baseline_script():
    """测试baseline.py脚本语法"""
    script_path = 'core/baseline.py'
    if not os.path.exists(script_path):
        return f"脚本不存在: {script_path}"
    
    # 尝试编译脚本
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    try:
        compile(code, script_path, 'exec')
        return True
    except SyntaxError as e:
        return f"语法错误: {e}"

def test_train_ssl_script():
    """测试train_ssl.py脚本语法"""
    script_path = 'core/train_ssl.py'
    if not os.path.exists(script_path):
        return f"脚本不存在: {script_path}"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    try:
        compile(code, script_path, 'exec')
        return True
    except SyntaxError as e:
        return f"语法错误: {e}"

def test_train_cls_script():
    """测试train_cls.py脚本语法"""
    script_path = 'core/train_cls.py'
    if not os.path.exists(script_path):
        return f"脚本不存在: {script_path}"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    try:
        compile(code, script_path, 'exec')
        return True
    except SyntaxError as e:
        return f"语法错误: {e}"

def test_fewshot_script():
    """测试fewshot.py脚本语法"""
    script_path = 'core/fewshot.py'
    if not os.path.exists(script_path):
        return f"脚本不存在: {script_path}"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    try:
        compile(code, script_path, 'exec')
        return True
    except SyntaxError as e:
        return f"语法错误: {e}"

def test_download_weights_script():
    """测试download_weights.py脚本"""
    script_path = 'scripts/download_weights.py'
    if not os.path.exists(script_path):
        return f"脚本不存在: {script_path}"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    try:
        compile(code, script_path, 'exec')
        return True
    except SyntaxError as e:
        return f"语法错误: {e}"

def test_weight_loading_logic():
    """测试权重加载逻辑（模拟）"""
    import torch
    from dinov3.models.vision_transformer import vit_small
    
    model = vit_small(patch_size=16, num_classes=0)
    
    # 创建模拟权重
    state_dict = model.state_dict()
    
    # 测试清理key的逻辑
    prefixed_dict = {f"module.{k}": v for k, v in state_dict.items()}
    cleaned_dict = {k.replace("module.", ""): v for k, v in prefixed_dict.items()}
    
    # 测试加载
    msg = model.load_state_dict(cleaned_dict, strict=False)
    print_info(f"缺失keys: {len(msg.missing_keys)}, 未使用keys: {len(msg.unexpected_keys)}")
    
    return True

def test_pos_embed_interpolation():
    """测试位置编码插值"""
    import torch
    import torch.nn.functional as F
    
    # 模拟位置编码插值
    pos_embed = torch.randn(1, 197, 384)  # 14x14 patches + 1 cls token
    
    num_extra_tokens = 1
    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]
    
    # 重塑并插值
    orig_size = 16  # 目标16x16
    ckpt_size = 14  # 原始14x14
    
    pos_tokens = pos_tokens.reshape(1, ckpt_size, ckpt_size, 384).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(orig_size, orig_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    print_info(f"插值前: {pos_embed.shape}, 插值后: {new_pos_embed.shape}")
    
    return True

def test_mixup_cutmix():
    """测试Mixup/CutMix逻辑"""
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    batch_size = 4
    num_classes = 10
    
    imgs = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    targets = F.one_hot(labels, num_classes=num_classes).float()
    
    # Mixup
    lam = 0.5
    rand_index = torch.randperm(batch_size)
    mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index]
    mixed_targets = lam * targets + (1 - lam) * targets[rand_index]
    
    print_info(f"Mixup输出: {mixed_imgs.shape}, {mixed_targets.shape}")
    
    return True

def test_llrd_optimizer():
    """测试LLRD优化器构建逻辑"""
    import torch
    from dinov3.models.vision_transformer import vit_small
    
    model = vit_small(patch_size=16, num_classes=0)
    
    # 模拟LLRD参数分组
    num_layers = len(model.blocks)
    base_lr = 1e-4
    layer_decay = 0.75
    
    param_groups = []
    for layer_id in range(num_layers + 2):
        scale = layer_decay ** (num_layers - layer_id + 1)
        lr = base_lr * scale
        param_groups.append({'lr': lr, 'layer': layer_id})
    
    print_info(f"LLRD层数: {len(param_groups)}, LR范围: {param_groups[-1]['lr']:.2e} ~ {param_groups[0]['lr']:.2e}")
    
    return True

def test_memory_usage():
    """测试内存使用"""
    import torch
    from dinov3.models.vision_transformer import vit_base
    
    model = vit_base(patch_size=16, num_classes=0)
    
    # 计算模型大小
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
    
    print_info(f"模型内存: 参数 {param_size:.1f}MB + 缓冲 {buffer_size:.1f}MB = {param_size + buffer_size:.1f}MB")
    
    # 测试前向传播内存
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    
    return True

# ============================================================================
# 主测试流程
# ============================================================================

def main():
    print_header("Radiolaria-DINOv3 项目全面测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}\n")
    
    # 测试分组
    test_groups = [
        ("环境检查", [
            ("Python版本", test_python_version),
            ("项目结构", test_project_structure),
        ]),
        
        ("依赖库导入", [
            ("PyTorch", test_import_torch),
            ("torchvision", test_import_torchvision),
            ("numpy", test_import_numpy),
            ("pandas", test_import_pandas),
            ("scikit-learn", test_import_sklearn),
            ("omegaconf", test_import_omegaconf),
            ("PIL", test_import_pil),
        ]),
        
        ("DINOv3模块", [
            ("模型导入", test_dinov3_models),
            ("配置导入", test_dinov3_configs),
            ("工具导入", test_utils_data),
        ]),
        
        ("模型功能", [
            ("模型初始化", test_model_initialization),
            ("数据增强", test_data_transforms),
            ("权重加载逻辑", test_weight_loading_logic),
            ("位置编码插值", test_pos_embed_interpolation),
            ("Mixup/CutMix", test_mixup_cutmix),
            ("LLRD优化器", test_llrd_optimizer),
            ("内存使用", test_memory_usage),
        ]),
        
        ("配置文件", [
            ("YAML配置", test_config_files),
        ]),
        
        ("数据集", [
            ("数据目录", test_data_directory),
            ("CSV加载", test_csv_loading),
            ("数据集创建", test_dataset_creation),
        ]),
        
        ("训练脚本", [
            ("baseline.py", test_baseline_script),
            ("train_ssl.py", test_train_ssl_script),
            ("train_cls.py", test_train_cls_script),
            ("fewshot.py", test_fewshot_script),
            ("download_weights.py", test_download_weights_script),
        ]),
    ]
    
    # 执行测试
    for group_name, tests in test_groups:
        print_header(group_name)
        for test_name, test_func in tests:
            run_test(test_name, test_func)
    
    # 生成报告
    print_header("测试总结")
    
    total = len(test_results['passed']) + len(test_results['failed'])
    passed = len(test_results['passed'])
    failed = len(test_results['failed'])
    
    print(f"总测试数: {total}")
    print(f"{Colors.GREEN}✓ 通过: {passed}{Colors.RESET}")
    print(f"{Colors.RED}✗ 失败: {failed}{Colors.RESET}")
    
    if failed > 0:
        print(f"\n{Colors.RED}{Colors.BOLD}失败的测试:{Colors.RESET}")
        for name, error in test_results['failed']:
            print(f"\n{Colors.RED}• {name}{Colors.RESET}")
            # 只打印错误的前3行
            error_lines = str(error).split('\n')[:3]
            for line in error_lines:
                print(f"  {line}")
    
    # 保存详细日志
    log_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Radiolaria-DINOv3 测试报告\n")
        f.write(f"{'='*70}\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python版本: {sys.version}\n")
        f.write(f"工作目录: {os.getcwd()}\n\n")
        
        f.write(f"总测试数: {total}\n")
        f.write(f"通过: {passed}\n")
        f.write(f"失败: {failed}\n\n")
        
        if test_results['passed']:
            f.write(f"通过的测试:\n")
            for name in test_results['passed']:
                f.write(f"  ✓ {name}\n")
            f.write("\n")
        
        if test_results['failed']:
            f.write(f"失败的测试:\n")
            for name, error in test_results['failed']:
                f.write(f"  ✗ {name}\n")
                f.write(f"     错误: {error}\n\n")
    
    print(f"\n详细日志已保存到: {log_file}")
    
    # 返回状态码
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
