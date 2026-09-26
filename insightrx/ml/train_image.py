"""
Train the multi-task fundus model for one seed, then export per-image logits + embeddings for every image.

  python -m insightrx.ml.train_image --seed 42
  INSIGHTRX_SMOKE=1 python -m insightrx.ml.train_image --seed 42     # 60 patients/split, 224px, 2 epochs
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from . import config as C
from .data import FundusDataset, label_audit, label_matrix, load_split, load_tables
from .model import RetiModel, multitask_loss, pos_weights, split_logits


def num_workers():
    return max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', '6')) - 1)


@torch.no_grad()
def predict(model, loader, device, n):
    model.eval()
    k = sum(h[2] for h in C.HEADS)
    logits = np.zeros((n, k), np.float32)
    emb = np.zeros((n, 2 * model.backbone.config.hidden_size), np.float16)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out, g = model(x, return_embedding=True)
            out_f, g_f = model(torch.flip(x, dims=[3]), return_embedding=True)     # horizontal-flip TTA
        z = torch.cat([out[n].float() for n, *_ in C.HEADS], 1)
        z_f = torch.cat([out_f[n].float() for n, *_ in C.HEADS], 1)
        logits[idx.numpy()] = ((z + z_f) / 2).cpu().numpy()
        emb[idx.numpy()] = ((g.float() + g_f.float()) / 2).cpu().numpy().astype(np.float16)
    return logits, emb


def val_score(logits, y):
    """Mean AUROC over quality / referable DR / edema (the heads the app acts on)."""
    z = split_logits(logits)
    aucs = {}
    for j, (name, *_ ) in enumerate(C.HEADS):
        if name not in C.RETINAL_HEADS:
            continue
        ok = ~np.isnan(y[:, j])
        if len(np.unique(y[ok, j])) == 2:
            aucs[name] = float(roc_auc_score(y[ok, j], z[name][ok]))
    return float(np.mean(list(aucs.values()))) if aucs else 0.0, aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=C.SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    C.ensure_dirs()
    device = 'cuda'

    images, patients = load_tables()
    split = load_split()
    label_audit(images, patients, split)
    sets = {k: images[images['patient_id'].isin(v)].reset_index(drop=True) for k, v in split.items()}
    Y = {k: label_matrix(v) for k, v in sets.items()}
    pw = pos_weights(Y['train'])
    size, nw = C.IMAGE_SIZE, num_workers()
    tr = DataLoader(FundusDataset(sets['train']['path'], Y['train'], size, True), batch_size=C.BATCH_SIZE,
                    shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True)
    va = DataLoader(FundusDataset(sets['val']['path'], Y['val'], size, False), batch_size=C.BATCH_SIZE * 2,
                    num_workers=nw, pin_memory=True)

    model = RetiModel().to(device)
    print(f"seed {args.seed}: {len(sets['train'])}/{len(sets['val'])}/{len(sets['test'])} train/val/test images, "
          f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M, "
          f"pos_weights {pw}", flush=True)
    opt = torch.optim.AdamW(model.param_groups(), weight_decay=C.WEIGHT_DECAY)
    steps, warm = C.EPOCHS * len(tr), len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    ckpt = os.path.join(C.CKPT_DIR, f'image_seed{args.seed}.pth')
    best, bad, hist, t0 = -1.0, 0, [], time.time()
    for ep in range(C.EPOCHS):
        model.train()
        tl, n = 0.0, 0
        for b, (x, y, _) in enumerate(tr):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x.to(device, non_blocking=True))
            loss = multitask_loss(out, y.to(device), pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            tl, n = tl + float(loss.detach()), n + 1
            if b % 50 == 0:
                print(f"  ep {ep + 1} step {b}/{len(tr)} loss {tl / n:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        va_logits, _ = predict(model, va, device, len(sets['val']))
        score, aucs = val_score(va_logits, Y['val'])
        hist.append({'epoch': ep + 1, 'train_loss': tl / max(n, 1), 'val_score': score, 'val_auroc': aucs,
                     'minutes': (time.time() - t0) / 60})
        print(f"ep {ep + 1}: train {tl / max(n, 1):.4f} | val {aucs} mean {score:.4f} (best {best:.4f})", flush=True)
        if score > best:
            best, bad = score, 0
            torch.save({'state_dict': model.trainable_state_dict(), 'heads': C.HEADS, 'backbone': C.BACKBONE,
                        'image_size': size, 'seed': args.seed, 'epoch': ep + 1, 'val_score': score,
                        'val_auroc': aucs}, ckpt)
        else:
            bad += 1
            if bad >= C.PATIENCE:
                print(f"early stop at epoch {ep + 1}", flush=True)
                break
    json.dump({'history': hist, 'best_val_score': best, 'pos_weights': pw},
              open(os.path.join(C.CKPT_DIR, f'image_seed{args.seed}_history.json'), 'w'), indent=1)

    # export logits + embeddings for every split with the best weights
    model.load_state_dict(torch.load(ckpt, map_location=device)['state_dict'], strict=False)
    for name, df in sets.items():
        loader = DataLoader(FundusDataset(df['path'], Y[name], size, False), batch_size=C.BATCH_SIZE * 2,
                            num_workers=nw, pin_memory=True)
        logits, emb = predict(model, loader, device, len(df))
        np.save(os.path.join(C.FEAT_DIR, f'seed{args.seed}_{name}_logits.npy'), logits)
        np.save(os.path.join(C.FEAT_DIR, f'seed{args.seed}_{name}_emb.npy'), emb)
        df[['image_id', 'patient_id', 'eye']].to_csv(os.path.join(C.FEAT_DIR, f'{name}_index.csv'), index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min; best val {best:.4f}; ckpt {ckpt}", flush=True)


if __name__ == '__main__':
    main()
