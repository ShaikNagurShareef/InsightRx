"""Multi-task fundus model: DINOv2-L + LoRA (q/v) + last blocks unfrozen (recipe from OculoMoE Stage F)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha=None, dropout=0.05):
        super().__init__()
        self.base = base
        self.scale = (alpha or 2 * rank) / rank
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.B(self.A(self.drop(x))) * self.scale


class RetiModel(nn.Module):
    def __init__(self, heads=C.HEADS, grad_checkpointing=True):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(C.BACKBONE)
        if grad_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        layers = self.backbone.encoder.layer
        for blk in layers:
            att = blk.attention.attention
            att.query = LoRALinear(att.query, C.LORA_RANK)
            att.value = LoRALinear(att.value, C.LORA_RANK)
        for blk in layers[-C.UNFREEZE_LAST:]:
            for n, p in blk.named_parameters():
                if '.A.' not in n and '.B.' not in n:
                    p.requires_grad_(True)
        for p in self.backbone.layernorm.parameters():
            p.requires_grad_(True)
        D = self.backbone.config.hidden_size
        self.heads = heads
        self.norm = nn.LayerNorm(2 * D)
        self.drop = nn.Dropout(0.1)
        self.out = nn.ModuleDict({n: nn.Linear(2 * D, k) for n, _, k, _ in heads})

    def forward(self, x, return_embedding=False):
        h = self.backbone(pixel_values=x).last_hidden_state
        g = self.norm(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1))
        logits = {n: self.out[n](self.drop(g)) for n, *_ in self.heads}
        return (logits, g) if return_embedding else logits

    def param_groups(self):
        lora, blocks, heads = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if '.A.' in n or '.B.' in n:
                lora.append(p)
            elif n.startswith('backbone'):
                blocks.append(p)
            else:
                heads.append(p)
        return [{'params': lora, 'lr': C.LR_LORA}, {'params': blocks, 'lr': C.LR_BLOCKS},
                {'params': heads, 'lr': C.LR_LORA * 5}]

    def trainable_state_dict(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}


# ------------------------------------------------------------------ ordinal (CORAL)
def ordinal_targets(y, num_classes):
    thresholds = torch.arange(num_classes - 1, device=y.device, dtype=y.dtype)
    return (y.unsqueeze(-1) > thresholds).float()


def ordinal_probs_np(z):
    p_gt = 1 / (1 + np.exp(-z))
    p_gt = np.minimum.accumulate(p_gt, axis=-1)
    return np.clip(np.concatenate([1 - p_gt[..., :1], p_gt[..., :-1] - p_gt[..., 1:], p_gt[..., -1:]], -1), 0, 1)


# ------------------------------------------------------------------ loss
def pos_weights(train_labels):
    w = {}
    for j, (name, typ, *_ ) in enumerate(C.HEADS):
        if typ == 'binary':
            v = train_labels[:, j]
            v = v[~np.isnan(v)]
            w[name] = float(np.clip(np.sqrt((v == 0).sum() / max((v == 1).sum(), 1)), 1.0, 20.0))
    return w


def multitask_loss(logits, y, pw):
    total = 0.0
    for j, (name, typ, k, weight) in enumerate(C.HEADS):
        yy = y[:, j]
        ok = ~torch.isnan(yy)
        if ok.sum() == 0:
            continue
        z, t = logits[name].float()[ok], yy[ok]
        if typ == 'binary':
            l = F.binary_cross_entropy_with_logits(z[:, 0], t, pos_weight=torch.tensor(pw[name], device=z.device))
        else:
            l = F.binary_cross_entropy_with_logits(z, ordinal_targets(t, k + 1))
        total = total + weight * l
    return total


def split_logits(logits_np):
    """Concatenated head logits [N, sum k] -> {head: array}."""
    out, c = {}, 0
    for name, _, k, _ in C.HEADS:
        out[name] = logits_np[:, c] if k == 1 else logits_np[:, c:c + k]
        c += k
    return out
