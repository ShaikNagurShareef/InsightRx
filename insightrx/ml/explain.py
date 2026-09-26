"""
Explainability for ViT fundus models: gradient x activation relevance on patch tokens.

We capture the tokens entering the last transformer block, back-propagate a scalar target (a head logit for the
fine-tuned DR model, or the linear systemic head's image term for a frozen encoder), and score each patch by
ReLU(sum_d grad * activation). The map is upsampled onto the same cropped/resized fundus the model saw.

This is *model attention*, not lesion segmentation or proof of causation (spec FR19).
"""
import io

import numpy as np
import torch
from PIL import Image, ImageFilter


def patch_relevance(forward_fn, blocks, x, target_fn):
    """forward_fn(x) -> model output; blocks: modules whose output tokens [B, 1+g*g, D] are explained (maps are
    robust-normalised per block and averaged); target_fn(output) -> [B] scalar targets.
    Returns [B, g, g] non-negative relevance."""
    blocks = blocks if isinstance(blocks, (list, tuple)) else [blocks]
    store, hooks = [], []

    def hook(_, __, out):
        out = out[0] if isinstance(out, tuple) else out
        out.retain_grad()
        store.append(out)

    for b in blocks:
        hooks.append(b.register_forward_hook(hook))
    try:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)       # make sure the graph reaches the hooked blocks
            out = forward_fn(x)
            target_fn(out).sum().backward()
    finally:
        for h in hooks:
            h.remove()
    maps = []
    for t in store:
        rel = torch.relu((t.grad[:, 1:] * t[:, 1:]).float().sum(-1)).detach()
        scale = torch.quantile(rel, 0.99, dim=1, keepdim=True) + 1e-8
        maps.append((rel / scale).clamp(max=1.0))
    rel = torch.stack(maps).mean(0)
    g = int(round(rel.shape[1] ** 0.5))
    return rel.reshape(-1, g, g).cpu().numpy()


def overlay_png(img: Image.Image, cam: np.ndarray, size=512, alpha=0.45) -> bytes:
    """Blend a relevance map onto the model-input view of the fundus; returns PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    from .data import FundusCrop
    base = FundusCrop()(img.convert("RGB")).resize((size, size), Image.BICUBIC)
    cam = cam / (np.quantile(cam, 0.99) + 1e-8)                # robust scale: one hot token cannot dominate
    heat = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    h = np.asarray(heat).astype(np.float32) / 255.0
    # outside the circular field of view there is no retina: ViT "artifact" tokens often park there
    fov = np.asarray(base.convert("L").filter(ImageFilter.MinFilter(9))) > 20
    h = h * fov
    if fov.any():                                          # display the top 15% most relevant retina only
        cut = np.quantile(h[fov], 0.85)
        h = np.where(h >= cut, (h - cut) / (h.max() - cut + 1e-8) * 0.75 + 0.25, 0.0)
    rgb = (colormaps["turbo"](h)[..., :3] * 255).astype(np.float32)
    b = np.asarray(base).astype(np.float32)
    a = (alpha * (h > 0))[..., None]
    out = (b * (1 - a) + rgb * a).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, "PNG")
    return buf.getvalue()


def systemic_image_direction(head, n_img):
    """For a linear systemic head (image: or fused:), the direction v in frozen-feature space such that the
    head's logit changes by v . e (per unit change of the patient-mean embedding e). None for metadata-only."""
    variant, model = head["variant"], head["model"]
    if variant == "metadata":
        return None
    if variant.startswith("image:"):
        scaler, pca, lr = model.steps[0][1], model.steps[1][1], model.steps[2][1]
        w = lr.coef_[0]
    else:
        ct, lr = model.steps[0][1], model.steps[1][1]
        img = ct.named_transformers_["img"]
        scaler, pca = img.steps[0][1], img.steps[1][1]
        w = lr.coef_[0][:pca.n_components_]
    return (w @ pca.components_) / scaler.scale_
