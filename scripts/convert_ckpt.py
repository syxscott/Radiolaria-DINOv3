#!/usr/bin/env python3
"""Convert a DINOv3 DCP checkpoint directory (ckpt/<iter>/) into a plain
teacher state_dict .pth that scripts/exp_utils.load_dinov3_backbone can load.

Usage: python scripts/convert_ckpt.py <ckpt_dir> <output.pth>

DCP layout (verified): dcp_to_torch_save yields {"iteration", "model", "optimizer"}
where payload["model"] holds flattened keys incl. "teacher.backbone.*".
We keep only the teacher backbone weights.
"""

import os
import sys

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ckpt_dir, out_path = sys.argv[1], sys.argv[2]

    tmp_path = out_path + ".tmp_full.pt"
    if not os.path.exists(tmp_path):
        print(f"[convert] DCP -> torch.save: {ckpt_dir} (may take a minute)")
        dcp_to_torch_save(ckpt_dir, tmp_path)
    payload = torch.load(tmp_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        print("[convert] ERROR: unexpected payload type:", type(payload))
        sys.exit(2)

    # Candidate layouts, in order:
    # 1) nested {"teacher": state_dict}
    # 2) top-level flat "teacher.*" keys
    # 3) payload["model"] with flat "teacher.backbone.*" keys (verified DINOv3 layout)
    state_dict = None
    if isinstance(payload.get("teacher"), dict):
        state_dict = dict(payload["teacher"])
    elif any(k.startswith("teacher.") for k in payload):
        state_dict = {k[len("teacher."):]: v for k, v in payload.items() if k.startswith("teacher.")}
    else:
        model_state = payload.get("model")
        if isinstance(model_state, dict):
            state_dict = {
                k[len("teacher."):]: v
                for k, v in model_state.items()
                if k.startswith("teacher.backbone.")
            }

    if not state_dict:
        print("[convert] ERROR: no teacher.* keys found. Top-level prefixes:",
              sorted({k.split(".")[0] for k in payload})[:10])
        sys.exit(3)

    # Keep only backbone weights; drop dino/ibot heads for a compact file.
    backbone = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
    if backbone:
        state_dict = {k[len("backbone."):]: v for k, v in backbone.items()}

    torch.save(state_dict, out_path)
    n = len(state_dict)
    n_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    print(f"[convert] saved {out_path}: {n} tensors, {n_params/1e6:.1f}M params")
    os.remove(tmp_path)


if __name__ == "__main__":
    main()
