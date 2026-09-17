#!/usr/bin/env python3
"""Supervised-contrastive (SupCon) adaptation on the SAME unlabeled corpus used by
TAPT — but WITH labels (2415 train-split images, 281 classes).

Frozen backbone; train a 2-layer projection head with SupCon loss on two
augmented views per image. Then extract eval-transform features through
backbone+projection into the standard cache layout (tag='supcon').
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import (  # noqa: E402
    RadiolariaDataset,
    build_radiolaria_index,
    get_eval_transform,
    get_train_transform,
    load_dinov3_backbone,
)


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, records, transform):
        self.ds = RadiolariaDataset(records, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _, _, _ = self.ds[idx]
        return img, idx


def supcon_loss(z, labels, temperature=0.07):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature
    n = len(labels)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    logits_mask = 1.0 - torch.eye(n, device=z.device)
    exp = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    return -mean_log_prob_pos.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./model/dinov3_vits16_pretrain.pth")
    ap.add_argument("--tag", default="supcon")
    ap.add_argument("--model_size", default="small")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--rotation_deg", type=int, default=180)
    ap.add_argument("--data_root", default="./data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, class_to_idx, _ = build_radiolaria_index(args.data_root)

    # TAPT 语料 = 训练切分 (与 tapt 对比严格同数据)
    import pandas as pd
    train_names = set(pd.read_csv("data/splits_fixed/train_fixed.csv")["filepath"]
                      .astype(str).map(lambda p: Path(p).name))
    train_records = [r for r in records if Path(r.path).name in train_names]
    print(f"SupCon 语料: {len(train_records)} 张 (与 TAPT 同源, 带标签)")

    backbone = load_dinov3_backbone(PROJECT_ROOT, args.model_size, args.checkpoint)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    embed_dim = backbone.embed_dim
    proj = nn.Sequential(
        nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim),
        nn.ReLU(inplace=True), nn.Linear(embed_dim, 128),
    ).to(device)

    ds = IndexedDataset(train_records, get_train_transform(224, 2, args.rotation_deg))
    loader1 = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, drop_last=True,
    )
    loader2 = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, drop_last=True,
    )
    opt = torch.optim.Adam(proj.parameters(), lr=args.lr, weight_decay=1e-4)

    proj.train()
    for epoch in range(args.epochs):
        tot, nb = 0.0, 0
        for (imgs, idx), (imgs2, idx2) in zip(loader1, loader2):
            imgs = imgs.to(device, non_blocking=True)
            idx = idx.to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    f = backbone(imgs)
                f = f.float()
            z1 = proj(f)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    f2 = backbone(imgs2.to(device, non_blocking=True))
                f2 = f2.float()
            z2 = proj(f2)
            z = torch.cat([z1, z2], 0)
            y = torch.cat([idx, idx2.to(device)], 0)
            loss = supcon_loss(z, y, args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        print(f"[supcon] epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    # 评估特征: backbone(cls) -> projection, eval transform, 全数据集
    proj.eval()
    eval_ds = IndexedDataset(records, get_eval_transform(224))
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=256, shuffle=False,
                                         num_workers=8, pin_memory=True)
    cls_all = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                f = backbone(imgs)
            z = proj(f.float())
            cls_all.append(z.float().cpu())
    cls = torch.cat(cls_all).numpy()

    out_dir = Path("features") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cls.npy", cls)
    np.save(out_dir / "patchmean.npy", cls)  # supcon 只有单一特征, 占位保持格式一致
    np.save(out_dir / "labels.npy", np.array([r.label for r in records]))
    np.save(out_dir / "paths.npy", np.array([r.path for r in records]))
    print(f"[supcon] cached -> {out_dir} ({cls.shape})")


if __name__ == "__main__":
    main()
