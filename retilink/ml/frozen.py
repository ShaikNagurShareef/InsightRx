"""
Frozen foundation-model encoders (no label exposure) for the systemic association models.

Reproduces OculoMoE Stage-B features exactly: FundusCrop -> bicubic resize -> ImageNet norm -> ViT-L/14,
feature = [CLS, mean(patch tokens)] of the final (layer-normed) hidden state (2 x 1024 = 2048-d).
The cached features for all 5,164 mBRSET images are reused for training; the same encoder runs at inference.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OCULOMOE_CACHE = "/data/users3/nshaik3/Projects/Oculomics/OculoMoE/cache/mbrset"
HF_CACHE = "/home/users/nshaik3/.cache/huggingface/hub"
RETFOUND_FILE = os.path.join(HF_CACHE, "models--YukunZhou--RETFound_dinov2_meh/snapshots/"
                                       "6dff4c76632983267a6ae3000f7852b1f8667d9c/RETFound_dinov2_meh.pth")
BACKBONES = {  # name -> (family, image size)
    "RETFound_dinov2_meh_224px": ("retfound", 224),
    "RETFound_dinov2_meh_448px": ("retfound", 448),
    "dinov2-large_224px": ("dinov2", 224),
}


def load_cached(name):
    """-> (image_ids, [N, 2048] float32) from the OculoMoE cache."""
    feats = np.load(os.path.join(OCULOMOE_CACHE, f"image_feats_{name}.npy"), mmap_mode="r")
    idx = pd.read_csv(os.path.join(OCULOMOE_CACHE, f"image_feats_{name}_index.csv"))
    return idx["image_id"].values, np.asarray(feats[:, :2, :], dtype=np.float32).reshape(len(idx), -1)


class FrozenEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        family, self.size = BACKBONES[name]
        self.name, self.family = name, family
        if family == "dinov2":
            from transformers import AutoModel
            self.vit = AutoModel.from_pretrained("facebook/dinov2-large")
        else:
            import timm
            from timm.layers import resample_abs_pos_embed
            path = RETFOUND_FILE
            if not os.path.exists(path):          # outside the research machine: fetch from the Hub (gated model)
                from huggingface_hub import hf_hub_download
                path = hf_hub_download("YukunZhou/RETFound_dinov2_meh", "RETFound_dinov2_meh.pth")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sd = ck.get("teacher", ck.get("model", ck))
            sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
            sd.pop("mask_token", None)
            g = self.size // 14
            self.vit = timm.create_model("vit_large_patch14_dinov2", pretrained=False, img_size=self.size, num_classes=0)
            sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=(g, g), num_prefix_tokens=1)
            self.vit.load_state_dict(sd, strict=True)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def tokens(self, x):
        """Final layer-normed tokens [B, 1+g*g, D]."""
        if self.family == "dinov2":
            return self.vit(pixel_values=x).last_hidden_state
        return self.vit.forward_features(x)

    @staticmethod
    def pool(h):
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)

    def forward(self, x):
        return self.pool(self.tokens(x))

    def transform(self):
        from .data import eval_transform
        return eval_transform(self.size)

    @torch.no_grad()
    def encode(self, paths, device, batch=16):
        from .data import open_fundus
        tf, out = self.transform(), []      # OculoMoE decoded every JPEG with draft size 448 (= open_fundus(.., 224))
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(open_fundus(p, 224)) for p in paths[i:i + batch]]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                out.append(self(x).float().cpu().numpy())
        return np.concatenate(out)
