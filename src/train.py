"""Train Model #1 (Jetson). Transfer-learn a 3D-CNN for binary accident detection."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset import (WindowDataset, video_level_split,
                         build_window_manifest)
from src.model import build_model
from src.utils import set_seed, compute_metrics, best_threshold


def make_loaders(cfg):
    tr_meta, va_meta = video_level_split(cfg)
    tr_df = build_window_manifest(tr_meta, cfg, train=True, seed=cfg.train.seed)
    va_df = build_window_manifest(va_meta, cfg, train=False, seed=cfg.train.seed)
    tr = WindowDataset(tr_df, cfg, train=True)
    va = WindowDataset(va_df, cfg, train=False)
    common = dict(num_workers=cfg.train.num_workers, pin_memory=True,
                  persistent_workers=cfg.train.num_workers > 0)
    tl = DataLoader(tr, batch_size=cfg.train.batch_size, shuffle=True,
                    drop_last=True, **common)
    vl = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False, **common)
    return tl, vl, tr_df, va_df


def cosine_warmup(step, total, warmup, base, min_ratio=0.02):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model, loader, device):
    """Window-level evaluation (deterministic windows -> single pass)."""
    model.eval()
    cp, cy = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=x.is_cuda):
            logit = model(x)
        cp.append(torch.sigmoid(logit).float().cpu())
        cy.append(y)
    return torch.cat(cy).numpy(), torch.cat(cp).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tl, vl, tr_df, va_df = make_loaders(cfg)
    print(f"train windows={len(tr_df)} (pos={int((tr_df['label']==1).sum())}) "
          f"val windows={len(va_df)} (pos={int((va_df['label']==1).sum())}) device={device}")

    model = build_model(cfg).to(device)
    init_from = cfg.train.get("init_from", None)
    if init_from:                              # fine-tune from an existing checkpoint
        sd = torch.load(init_from, map_location=device, weights_only=False)["model"]
        model.load_state_dict(sd)
        print(f"initialized weights from {init_from}")
    pg = model.param_groups(cfg.train.lr, cfg.train.backbone_lr_mult)
    opt = torch.optim.AdamW(pg, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)

    pos = (tr_df["label"] == 1).sum()
    neg = (tr_df["label"] == 0).sum()
    pos_weight = torch.tensor([neg / max(1, pos)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ls = cfg.train.label_smoothing

    steps_per_epoch = len(tl)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = steps_per_epoch * cfg.train.warmup_epochs

    best_metric, best_path = -1.0, out_dir / "best.pt"
    history, patience = [], 0
    gstep = 0
    for epoch in range(cfg.train.epochs):
        model.set_backbone_frozen(epoch < cfg.model.freeze_stem_epochs)
        model.train()
        running = 0.0
        pbar = tqdm(tl, desc=f"epoch {epoch+1}/{cfg.train.epochs}")
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yt = y * (1 - ls) + 0.5 * ls            # label smoothing
            lr = cosine_warmup(gstep, total_steps, warmup_steps, cfg.train.lr)
            for i, g in enumerate(opt.param_groups):
                g["lr"] = lr * (cfg.train.backbone_lr_mult if i == 0 else 1.0)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.train.amp):
                logit = model(x)
                loss = crit(logit, yt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            gstep += 1
            pbar.set_postfix(loss=f"{running/(pbar.n+1):.4f}", lr=f"{lr:.2e}")

        y_true, y_prob = evaluate(model, vl, device)
        m = compute_metrics(y_true, y_prob, 0.5)
        m["epoch"] = epoch + 1
        m["train_loss"] = running / steps_per_epoch
        history.append(m)
        print(f"  val auc={m['auc']:.4f} ap={m['ap']:.4f} f1={m['f1']:.4f} acc={m['acc']:.4f}")

        torch.save({"model": model.state_dict(), "cfg": dict(cfg), "metrics": m},
                   out_dir / "last.pt")
        score = m[cfg.train.monitor.replace("val_", "")]
        if score > best_metric:
            best_metric = score
            torch.save({"model": model.state_dict(), "cfg": dict(cfg),
                        "metrics": m}, best_path)
            patience = 0
            print(f"  ** new best {cfg.train.monitor}={score:.4f} -> {best_path}")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print("  early stopping.")
                break

    # final: reload best, tune window-level threshold (temporal eval is in evaluate.py)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    y_true, y_prob = evaluate(model, vl, device)
    thr = best_threshold(y_true, y_prob)
    final = compute_metrics(y_true, y_prob, thr)
    final["tuned_threshold"] = thr
    print(f"FINAL (window-level): {final}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    # persist tuned threshold next to the checkpoint for export/inference
    ckpt["tuned_threshold"] = thr
    ckpt["final_metrics"] = final
    torch.save(ckpt, best_path)
    print(f"saved metrics -> {out_dir}")


if __name__ == "__main__":
    main()
