#!/usr/bin/env python3
"""Diagnose the TAPT fine-tuning collapse: compare official vs TAPT weights.

Prints per-parameter-group drift statistics to see whether the TAPT checkpoint
is numerically pathological (e.g., exploded LayerScale gammas) or merely drifted.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)


def load_sd(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if "teacher" in sd and isinstance(sd.get("teacher"), dict):
        sd = sd["teacher"]
    if "backbone" in sd and isinstance(sd.get("backbone"), dict):
        sd = sd["backbone"]
    return sd


def main():
    official = load_sd("model/dinov3_vits16_pretrain.pth")
    paths = {
        "tapt-nogram": "outputs/tapt_vits16_100_nogram/teacher_checkpoint.pth",
        "tapt-gram": "outputs/tapt_vits16_100/teacher_checkpoint.pth",
    }
    groups = {
        "cls_token": lambda k: k == "cls_token",
        "storage_tokens": lambda k: k == "storage_tokens",
        "pos_embed/rope": lambda k: "rope" in k,
        "patch_embed": lambda k: "patch_embed" in k,
        "ls1.gamma (blk0)": lambda k: k == "blocks.0.ls1.gamma",
        "ls1.gamma (all)": lambda k: ".ls1.gamma" in k,
        "attn.qkv (all)": lambda k: ".attn.qkv.weight" in k,
        "mlp.fc2 (all)": lambda k: ".mlp.fc2.weight" in k,
        "norm (all)": lambda k: ".norm" in k,
    }

    print(f"{'参数组':<22} {'官方范数':>12} {'无Gram漂移%':>12} {'Gram漂移%':>12}")
    for gname, pred in groups.items():
        keys = [k for k in official if pred(k)]
        if not keys:
            continue
        base_norm = torch.cat([official[k].flatten().float() for k in keys]).norm().item()
        line = f"{gname:<22} {base_norm:>12.2f}"
        for tag, p in paths.items():
            sd = load_sd(p)
            try:
                drift = torch.cat([
                    sd[k].flatten().float() - official[k].flatten().float() for k in keys
                ]).norm().item()
                line += f" {100*drift/max(base_norm,1e-9):>11.2f}%"
            except KeyError:
                line += f" {'missing':>12}"
        print(line)

    print("\n=== 漂移最大的 10 个参数 (无Gram vs 官方) ===")
    sd = load_sd(paths["tapt-nogram"])
    drifts = []
    for k in official:
        if k in sd and official[k].shape == sd[k].shape:
            d = (sd[k].flatten().float() - official[k].flatten().float()).norm().item()
            n = official[k].flatten().float().norm().item()
            drifts.append((d / max(n, 1e-9), k, n))
    for rel, k, n in sorted(drifts, reverse=True)[:10]:
        print(f"  {k:<44} 相对漂移 {rel*100:8.2f}%  (范数 {n:.2f})")


if __name__ == "__main__":
    main()
