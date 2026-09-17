#!/usr/bin/env python3
"""Extract and cache DINOv3 features for the whole dataset (canonical order).

Usage:
  python scripts/feature_cache.py --checkpoint <pth> --tag <tag> \
      [--resolutions 224,336,448] [--flip-resolutions 224,448] [--data_root ./data] [--batch_size 128]

For each resolution r: features/<tag>_r<r>/  {cls.npy, patchmean.npy, labels.npy, paths.npy}
For each flip-resolution r (subset): features/<tag>_r<r>_fliptta/ same layout
(flip-TTA = average of L2-normalised CLS/patch features of image and its horizontal flip).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import (  # noqa: E402
    RadiolariaDataset,
    build_radiolaria_index,
    get_eval_transform,
    load_dinov3_backbone,
)


@torch.no_grad()
def extract(backbone, loader, device, flip):
    cls_all, patch_all, labels_all = [], [], []
    for imgs, idx in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outs = backbone.forward_features(imgs)
            cls = outs["x_norm_clstoken"]
            patch = outs["x_norm_patchtokens"].mean(dim=1)
            if flip:
                outs2 = backbone.forward_features(torch.flip(imgs, dims=[3]))
                cls = cls + outs2["x_norm_clstoken"]
                patch = patch + outs2["x_norm_patchtokens"].mean(dim=1)
        cls_all.append(cls.float().cpu())
        patch_all.append(patch.float().cpu())
        labels_all.append(idx)
    return (
        torch.cat(cls_all).numpy(),
        torch.cat(patch_all).numpy(),
        torch.cat(labels_all).numpy(),
    )


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, records, transform):
        self.ds = RadiolariaDataset(records, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _, _, _ = self.ds[idx]
        return img, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model_size", default="small")
    ap.add_argument("--resolutions", default="224,336,448")
    ap.add_argument("--flip_resolutions", default="224,448")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, class_to_idx, _ = build_radiolaria_index(args.data_root)
    print(f"[cache:{args.tag}] records={len(records)} classes={len(class_to_idx)}")

    backbone = load_dinov3_backbone(PROJECT_ROOT, args.model_size, args.checkpoint)
    backbone.to(device).eval()

    resolutions = [int(r) for r in args.resolutions.split(",") if r]
    flips = {int(r) for r in args.flip_resolutions.split(",") if r}

    for res in resolutions:
        variants = [("norm", False)] + ([("fliptta", True)] if res in flips else [])
        for vname, use_flip in variants:
            dirname = f"{args.tag}_r{res}" + ("_fliptta" if use_flip else "")
            out_dir = Path("features") / dirname
            out_dir.mkdir(parents=True, exist_ok=True)
            if (out_dir / "cls.npy").exists():
                print(f"[cache:{args.tag}] skip existing {out_dir}")
                continue
            ds = IndexedDataset(records, get_eval_transform(res))
            loader = torch.utils.data.DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            t0 = __import__("time").time()
            cls, patch, labels = extract(backbone, loader, device, use_flip)
            np.save(out_dir / "cls.npy", cls)
            np.save(out_dir / "patchmean.npy", patch)
            # labels.npy 存类别标签(与 records 顺序一致); 数据集索引另存 indices.npy
            np.save(out_dir / "labels.npy", np.array([r.label for r in records]))
            np.save(out_dir / "indices.npy", labels)
            np.save(out_dir / "paths.npy", np.array([r.path for r in records]))
            print(f"[cache:{args.tag}] {out_dir.name}: {cls.shape} in {__import__('time').time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
