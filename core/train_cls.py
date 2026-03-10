import argparse
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import f1_score, recall_score
import warnings

warnings.filterwarnings("ignore")

# 确保项目根目录在 path 中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 注意：这里的引用路径变了，适应新的 utils 结构
from utils.data_utils import get_stratified_datasets, get_transforms
# [Updated] 引入 vit_small
from dinov3.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


# -----------------------------------------------------------------------------
# 分布式工具
# -----------------------------------------------------------------------------
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return gpu, rank, world_size
    else:
        print("[Warning] Running in non-distributed mode.")
        return 0, 0, 1


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# 权重加载与位置编码插值
# -----------------------------------------------------------------------------
def interpolate_pos_embed(state_dict, model):
    if 'pos_embed' not in state_dict:
        return state_dict

    pos_embed_checkpoint = state_dict['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    orig_size = int(math.sqrt(num_patches))

    if pos_embed_checkpoint.shape[-2] == model.pos_embed.shape[-2]:
        return state_dict

    if is_main_process():
        print(f"⚠️ [PosEmbed] Resizing pos_embed: {pos_embed_checkpoint.shape} -> {model.pos_embed.shape}")

    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

    ckpt_num_patches = pos_tokens.shape[1]
    ckpt_size = int(math.sqrt(ckpt_num_patches))

    pos_tokens = pos_tokens.reshape(1, ckpt_size, ckpt_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(
        pos_tokens, size=(orig_size, orig_size), mode='bicubic', align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    state_dict['pos_embed'] = new_pos_embed
    return state_dict


def load_weights_robust(model, weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    if is_main_process():
        print(f"Loading weights from: {weights_path}")

    checkpoint = torch.load(weights_path, map_location='cpu')

    if 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 检查权重文件的维度是否匹配模型
    expected_dim = model.embed_dim
    actual_dim = None
    
    # 检查 cls_token 的维度
    if 'cls_token' in state_dict:
        actual_dim = state_dict['cls_token'].shape[-1]
    elif 'mask_token' in state_dict:
        # 如果没有 cls_token，尝试从 mask_token 推断维度
        actual_dim = state_dict['mask_token'].shape[-1]
    elif 'patch_embed.proj.weight' in state_dict:
        # 从 patch_embed 推断维度
        actual_dim = state_dict['patch_embed.proj.weight'].shape[0]
    
    if actual_dim and actual_dim != expected_dim:
        print(f"⚠️ 权重维度不匹配: 权重文件 {actual_dim}D vs 模型 {expected_dim}D")
        print(f"   请确保权重文件与模型大小一致")
        raise ValueError(f"模型大小不匹配: 期望 {expected_dim}D，得到 {actual_dim}D")

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        new_state_dict[k] = v

    new_state_dict = interpolate_pos_embed(new_state_dict, model)

    msg = model.load_state_dict(new_state_dict, strict=False)

    missing_blocks = [k for k in msg.missing_keys if "blocks" in k]
    if len(missing_blocks) > 0:
        if len(missing_blocks) > len(model.blocks) * 0.5:
            raise RuntimeError(
                f"❌ 严重错误: 权重加载丢失了大部分 Transformer Blocks! \nMissing: {missing_blocks[:5]}...")

    if is_main_process():
        print(f"✅ 权重加载成功. Missing keys: {len(msg.missing_keys)}")

    return model


# -----------------------------------------------------------------------------
# 辅助类
# -----------------------------------------------------------------------------
class MixupCutmixCollator:
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def rand_bbox(self, size, lam):
        W = size[2];
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat);
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W);
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W);
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W);
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        imgs, labels, paths = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)
        targets = F.one_hot(labels, num_classes=self.num_classes).float()

        if np.random.rand() > self.prob: return imgs, targets, paths

        use_cutmix = np.random.rand() > 0.5 and self.cutmix_alpha > 0
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_index = torch.randperm(imgs.size(0))
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))
            targets = lam * targets + (1 - lam) * targets[rand_index]
        elif self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_index = torch.randperm(imgs.size(0))
            imgs = lam * imgs + (1 - lam) * imgs[rand_index]
            targets = lam * targets + (1 - lam) * targets[rand_index]
        return imgs, targets, paths


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, x, target):
        return torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()


class SupervisedClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def set_train_mode(self, mode):
        if mode == 'linear':
            self.backbone.eval()
            for p in self.backbone.parameters(): p.requires_grad = False
            for p in self.head.parameters(): p.requires_grad = True
        elif mode == 'full_ft':
            self.backbone.train()
            for p in self.backbone.parameters(): p.requires_grad = True
            for p in self.head.parameters(): p.requires_grad = True


def build_optimizer_llrd(model, base_lr, weight_decay, layer_decay=0.75):
    real_model = model.module if hasattr(model, 'module') else model
    backbone = real_model.backbone
    head = real_model.head

    num_layers = len(backbone.blocks) if hasattr(backbone, 'blocks') else 12
    param_groups = {}

    def get_layer_id(name):
        if "patch_embed" in name or "pos_embed" in name or "cls_token" in name: return 0
        if "blocks" in name:
            try:
                return int(name.split('.')[1]) + 1
            except:
                return 0
        return num_layers + 1

    for name, param in backbone.named_parameters():
        if not param.requires_grad: continue
        layer_id = get_layer_id(name)
        scale = layer_decay ** (num_layers - layer_id + 1)

        grp = f"layer_{layer_id}"
        if grp not in param_groups:
            param_groups[grp] = {"params": [], "lr": base_lr * scale, "weight_decay": weight_decay}
        param_groups[grp]["params"].append(param)

    head_params = [p for p in head.parameters() if p.requires_grad]
    final_groups = list(param_groups.values())
    if head_params:
        final_groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay})

    return optim.AdamW(final_groups)


def train_epoch(model, loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    if hasattr(model.module, 'backbone') and not next(model.module.backbone.parameters()).requires_grad:
        model.module.backbone.eval()

    total_loss = torch.tensor(0.0).to(device)
    if dist.is_initialized(): loader.sampler.set_epoch(epoch)

    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(imgs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss /= dist.get_world_size()

    return total_loss.item() / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(imgs)
        _, preds = outputs.max(1)
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    if dist.is_initialized():
        preds_list = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
        labels_list = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]
        dist.all_gather(preds_list, all_preds)
        dist.all_gather(labels_list, all_labels)
        all_preds = torch.cat(preds_list).cpu().numpy()
        all_labels = torch.cat(labels_list).cpu().numpy()
    else:
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()

    if is_main_process():
        acc = 100. * np.mean(all_preds == all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_recall = recall_score(all_labels, all_preds, average='macro')
        return acc, macro_f1, avg_recall
    return 0, 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help="Path to pretrained weights")
    parser.add_argument('--data_root', required=True, help="Path to data root (containing csv files)")
    parser.add_argument('--img_root', default=None, help="Path to images (default: data_root/images)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', default='runs/supervised_cls')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--mode', default='two_stage', choices=['linear', 'full_ft', 'two_stage'])
    parser.add_argument('--img_size', type=int, default=224)
    # [Updated] 支持 vits16 和 vits16plus
    parser.add_argument('--model_size', default='vitl16',
                        choices=['vitb16', 'vitl16', 'vitg14', 'vits16', 'vits16plus'])
    args = parser.parse_args()

    gpu, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{gpu}")

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Config: {args}")

    # 处理图片路径默认值
    img_root = args.img_root if args.img_root else os.path.join(args.data_root, 'images')

    train_ds, val_ds, test_ds, class_to_idx = get_stratified_datasets(
        args.data_root,
        img_root,
        transform_train=get_transforms(args.img_size, True),
        transform_val=get_transforms(args.img_size, False)
    )

    mixup_fn = MixupCutmixCollator(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, num_classes=len(class_to_idx))
    criterion = SoftTargetCrossEntropy()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=DistributedSampler(train_ds),
                              num_workers=4, pin_memory=True, collate_fn=mixup_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=DistributedSampler(val_ds, shuffle=False),
                            num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=DistributedSampler(test_ds, shuffle=False),
                             num_workers=4)

    # [Updated] 初始化模型逻辑
    print(f"Initializing {args.model_size}...")
    if args.model_size == 'vitl16':
        backbone = vit_large(patch_size=16, num_classes=0)
    elif args.model_size == 'vitb16':
        backbone = vit_base(patch_size=16, num_classes=0)
    elif args.model_size == 'vitg14':
        backbone = vit_giant2(patch_size=14, num_classes=0)
    elif args.model_size == 'vits16' or args.model_size == 'vits16plus':
        backbone = vit_small(patch_size=16, num_classes=0)
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    backbone = load_weights_robust(backbone, args.weights)

    model = SupervisedClassifier(backbone, len(class_to_idx)).to(device)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    scaler = torch.cuda.amp.GradScaler()

    current_mode = 'linear' if args.mode == 'two_stage' else args.mode
    model.module.set_train_mode(current_mode)

    if current_mode == 'linear':
        optimizer = optim.AdamW(model.module.head.parameters(), lr=args.lr)
    else:
        optimizer = build_optimizer_llrd(model, args.lr, 0.05, args.layer_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    warmup_epochs = 20 if args.mode == 'two_stage' else 0

    for epoch in range(args.epochs):
        if args.mode == 'two_stage' and epoch == warmup_epochs:
            if is_main_process():
                print("\n>>> [Phase Switch] Switching from Linear to Full Finetuning with LLRD <<<")
            current_mode = 'full_ft'
            model.module.set_train_mode('full_ft')

            new_lr = args.lr * 0.5
            optimizer = build_optimizer_llrd(model, new_lr, 0.05, args.layer_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)

        loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        scheduler.step()

        if is_main_process():
            print(f"Ep {epoch}/{args.epochs} | Loss: {loss:.4f} | Mode: {current_mode}", end='\r')

        if epoch % 5 == 0 or epoch > args.epochs - 15:
            val_acc, val_f1, val_rec = evaluate(model, val_loader, device)
            if is_main_process():
                print(
                    f"\nEp {epoch} | Loss: {loss:.4f} | Val Acc: {val_acc:.2f} F1: {val_f1:.4f} Recall: {val_rec:.4f}")
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                    print("--> Best Model Saved.")

    dist.barrier()
    if is_main_process():
        print("\nLoading best model for testing...")
        model.module.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))

    test_acc, test_f1, test_rec = evaluate(model, test_loader, device)
    if is_main_process():
        print(f"Final Test Result -> Acc: {test_acc:.2f}%, F1: {test_f1:.4f}, Recall: {test_rec:.4f}")

    cleanup_distributed()


if __name__ == '__main__':
    main()