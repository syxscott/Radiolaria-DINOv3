#!/usr/bin/env python3
"""torchvision 监督基线的特征缓存 (与 feature_cache.py 同布局, cls = 全局池化特征)。

用法:
  python scripts/feature_cache_tv.py --models resnet18,resnet50,... \
      --resolutions 224,448 --flip_resolutions 224,448
输出: features/tv-<model>_r<res>[_fliptta]/{cls,patchmean,labels,paths}.npy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import (  # noqa: E402
    RadiolariaDataset, build_radiolaria_index, get_eval_transform,
)


def load_tv(model_name: str) -> nn.Module:
    name = model_name.lower().strip()
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT); m.fc = nn.Identity()
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT); m.fc = nn.Identity()
    elif name == "resnet101":
        m = tvm.resnet101(weights=tvm.ResNet101_Weights.DEFAULT); m.fc = nn.Identity()
    elif name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT); m.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT); m.classifier = nn.Identity()
    elif name == "efficientnet_b3":
        m = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.DEFAULT); m.classifier = nn.Identity()
    elif name == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT); m.classifier = nn.Identity()
    elif name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT); m.classifier[-1] = nn.Identity()
    elif name == "swin_t":
        m = tvm.swin_t(weights=tvm.Swin_T_Weights.DEFAULT); m.head = nn.Identity()
    elif name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT); m.heads = nn.Identity()
    else:
        raise ValueError(name)
    return m


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, records, transform):
        self.ds = RadiolariaDataset(records, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _, _, _ = self.ds[idx]
        return img, idx


@torch.no_grad()
def extract(model, loader, device, flip):
    feats, idxs = [], []
    for imgs, idx in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            f = model(imgs)
            if f.ndim > 2:
                f = torch.flatten(f, 1)
            if flip:
                f2 = model(torch.flip(imgs, dims=[3]))
                if f2.ndim > 2:
                    f2 = torch.flatten(f2, 1)
                f = f + f2
        feats.append(f.float().cpu())
        idxs.append(idx)
    return torch.cat(feats).numpy(), torch.cat(idxs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="逗号分隔的 torchvision 模型名")
    ap.add_argument("--resolutions", default="224,448")
    ap.add_argument("--flip_resolutions", default="224,448")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, _, _ = build_radiolaria_index(args.data_root)
    rec_labels = np.array([r.label for r in records])
    rec_paths = np.array([r.path for r in records])

    for name in args.models.split(","):
        name = name.strip()
        if not name:
            continue
        model = load_tv(name).to(device).eval()
        for res in [int(r) for r in args.resolutions.split(",") if r]:
            for use_flip in [False] + ([True] if res in
                                       [int(r) for r in args.flip_resolutions.split(",") if r] else []):
                dirname = f"tv-{name}_r{res}" + ("_fliptta" if use_flip else "")
                out_dir = Path("features") / dirname
                out_dir.mkdir(parents=True, exist_ok=True)
                if (out_dir / "cls.npy").exists():
                    print(f"[cache:{name}] skip {dirname}")
                    continue
                try:
                    ds = IndexedDataset(records, get_eval_transform(res))
                    loader = torch.utils.data.DataLoader(
                        ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
                    t0 = time.time()
                    feats, idxs = extract(model, loader, device, use_flip)
                except AssertionError as e:
                    # torchvision ViT 等模型固定输入分辨率 (224), 更高分辨率直接跳过
                    print(f"[cache:{name}] skip {dirname}: {e}", flush=True)
                    continue
                np.save(out_dir / "cls.npy", feats)
                np.save(out_dir / "patchmean.npy", feats)  # tv 无 patch 特征, 占位
                np.save(out_dir / "labels.npy", rec_labels)
                np.save(out_dir / "paths.npy", rec_paths)
                print(f"[cache:{name}] {dirname}: {feats.shape} in {time.time()-t0:.0f}s",
                      flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
