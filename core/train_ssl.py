import os
import sys
import logging
import shutil
import torch
from omegaconf import DictConfig, OmegaConf

# 确保项目根目录在 path 中，以便能 import dinov3
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === 0. 兼容无 C 编译器环境（避免 Triton/Inductor 报错） ===
if shutil.which("gcc") is None and shutil.which("cc") is None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    print("[Compat] No C compiler found. Set TORCHDYNAMO_DISABLE=1", flush=True)

# === 1. 动态修复日志报错 (Monkey Patching) ===
original_info = logging.Logger.info

def patched_info(self, msg, *args, **kwargs):
    if isinstance(msg, (DictConfig, dict, list)):
        try:
            if isinstance(msg, DictConfig):
                msg = OmegaConf.to_yaml(msg)
            else:
                msg = str(msg)
        except:
            msg = str(msg)
    original_info(self, msg, *args, **kwargs)

logging.Logger.info = patched_info

# === 2. 拦截 _parse_dataset_str 以支持本地路径 ===
# 在导入 dinov3 之前，先导入并保存原始版本
import dinov3.data.loaders as loaders
from torchvision.datasets import ImageFolder

original_parse_dataset_str = loaders._parse_dataset_str


def _resolve_local_dataset_root(dataset_str: str):
    """尽可能把本地数据路径解析为真实目录。"""
    s = str(dataset_str)
    candidates = [
        s,
        os.path.abspath(s),
    ]
    if s.startswith("./"):
        candidates.append(os.path.join(project_root, s[2:]))
    else:
        candidates.append(os.path.join(project_root, s))

    seen = set()
    for c in candidates:
        c_norm = os.path.abspath(c)
        if c_norm in seen:
            continue
        seen.add(c_norm)
        if os.path.isdir(c_norm):
            return c_norm
    return None

def patched_parse_dataset_str(dataset_str):
    """
    新增逻辑：如果 dataset_str 看起来像本地路径，直接返回 ImageFolder 类
    否则调用原始的解析逻辑
    """
    dataset_str = str(dataset_str)
    logger = logging.getLogger("dinov3")

    # 内置数据集前缀，交给原始逻辑处理
    builtin_prefixes = {"ImageNet", "ImageNet22k", "ADE20K", "CocoCaptions", "NYU"}
    prefix = dataset_str.split(":", 1)[0]
    if prefix in builtin_prefixes:
        return original_parse_dataset_str(dataset_str)

    # 先尝试把它当作本地目录解析
    resolved_root = _resolve_local_dataset_root(dataset_str)
    if resolved_root is not None:
        logger.info(f"[INTERCEPT] Local path detected: {dataset_str} -> {resolved_root}")
        return ImageFolder, {"root": resolved_root}

    # 如果看起来像路径（例如 ./xxx 或 含 /），即使当前不存在也强制走 ImageFolder，
    # 这样可以得到更明确的文件路径错误，而不是 Unsupported dataset
    looks_like_path = dataset_str.startswith(".") or ("/" in dataset_str) or ("\\" in dataset_str)
    if looks_like_path:
        fallback_root = os.path.abspath(dataset_str)
        logger.info(
            f"[INTERCEPT] Treat as local path (not found yet): {dataset_str} -> {fallback_root}"
        )
        return ImageFolder, {"root": fallback_root}

    # 其他情况回退到原始解析
    return original_parse_dataset_str(dataset_str)

# 替换 loaders 模块中的 _parse_dataset_str
loaders._parse_dataset_str = patched_parse_dataset_str

# === 3. 同时修改 make_dataset 的调用方式 ===
# 保存原始的 make_dataset
original_make_dataset = loaders.make_dataset

def patched_make_dataset(
    *,
    dataset_str: str,
    transform = None,
    target_transform = None,
    transforms = None,
):
    """
    新的 make_dataset：使用我们已经 patch 过的 _parse_dataset_str
    """
    logger = logging.getLogger("dinov3")
    logger.info(f'using dataset: "{dataset_str}"')

    # 调用我们已经 patch 过的 _parse_dataset_str
    class_, kwargs = loaders._parse_dataset_str(dataset_str)
    
    # 特殊处理 ImageFolder 类
    if class_ is ImageFolder:
        # ImageFolder 的调用方式
        dataset = class_(
            root=kwargs.get("root", dataset_str),
            transform=transform,
            target_transform=target_transform
        )
    else:
        # 其他数据集（ImageNet 等）的调用方式
        dataset = class_(
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            **kwargs
        )

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # 添加属性
    if not hasattr(dataset, "transform"):
        dataset.transform = transform
    if not hasattr(dataset, "target_transform"):
        dataset.target_transform = target_transform
    if not hasattr(dataset, "transforms"):
        dataset.transforms = transforms

    return dataset

# 替换 loaders 模块中的 make_dataset
loaders.make_dataset = patched_make_dataset

# 如果 train.py 已经导入了 make_dataset，也替换那里的
try:
    import dinov3.train.train as train_module
    if hasattr(train_module, "make_dataset"):
        train_module.make_dataset = patched_make_dataset
except:
    pass

print("[PATCHED] Both _parse_dataset_str and make_dataset have been intercepted.", flush=True)

# === 4. 启动训练 ===
from dinov3.train.train import main

if __name__ == "__main__":
    # 无编译器时强制关闭 compile; 有编译器可用 train.compile=true 提速
    has_compile_flag = any(arg.startswith("train.compile=") for arg in sys.argv)
    has_compiler = shutil.which("gcc") is not None or shutil.which("cc") is not None
    if not has_compile_flag and not has_compiler:
        sys.argv.append("train.compile=false")

    # --output-dir 与 train.output_dir 只传其一时自动补齐另一个
    has_job_dir = any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in sys.argv)
    has_cfg_dir = any(arg.startswith("train.output_dir=") for arg in sys.argv)
    job_dir = None
    if "--output-dir" in sys.argv:
        job_dir = sys.argv[sys.argv.index("--output-dir") + 1]
    else:
        for arg in sys.argv:
            if arg.startswith("--output-dir="):
                job_dir = arg.split("=", 1)[1]
                break
    cfg_dir = None
    for arg in sys.argv:
        if arg.startswith("train.output_dir="):
            cfg_dir = arg.split("=", 1)[1]
            break
    if job_dir and not cfg_dir:
        sys.argv.append(f"train.output_dir={job_dir}")
        print(f"[Compat] --output-dir 已同步到 train.output_dir={job_dir}", flush=True)
    elif cfg_dir and not has_job_dir:
        sys.argv.extend(["--output-dir", cfg_dir])
        print(f"[Compat] train.output_dir 已同步到 --output-dir={cfg_dir}", flush=True)

    # === 自动断点续训逻辑 (Auto-Resume) ===
    try:
        output_dir = None
        for arg in sys.argv[1:]:
            if arg.startswith("train.output_dir="):
                output_dir = arg.split("=", 1)[1]
                break
        if output_dir is None and "--output-dir" in sys.argv:
            idx = sys.argv.index("--output-dir")
            if idx + 1 < len(sys.argv):
                output_dir = sys.argv[idx + 1]
        if output_dir is None:
            for arg in sys.argv[1:]:
                if arg.startswith("--output-dir="):
                    output_dir = arg.split("=", 1)[1]
                    break

        if output_dir and os.path.exists(output_dir):
            ckpt_path = os.path.join(output_dir, "checkpoint.pth")
            if os.path.exists(ckpt_path):
                print("=" * 60)
                print(f"[Auto-Resume] 检测到现有 Checkpoint: {ckpt_path}")
                print("=" * 60)
                has_resume = any(arg.startswith("train.resume=") for arg in sys.argv)
                if not has_resume:
                    sys.argv.append("train.resume=true")
    except Exception as e:
        print(f"[Auto-Resume] 检测逻辑出错 (不影响训练): {e}")

    main()
