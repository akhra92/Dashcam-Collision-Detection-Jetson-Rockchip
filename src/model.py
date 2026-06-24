"""Model factory: Kinetics-pretrained video backbone with a binary head.

Two families are supported behind one interface (single-logit output, the same
param_groups / set_backbone_frozen API):

  * torchvision 3D-CNNs  (r2plus1d_18 / r3d_18 / mc3_18 / s3d) — pure-conv, the
    lightest path, cleanest TensorRT export. Best for Jetson Orin Nano.
  * VideoMAE ViT         (videomae_base / videomae_large) — self-supervised
    transformer, higher Kinetics accuracy, heavier compute. Needs `transformers`.

All take input [B, 3, T, H, W] (our 3D-CNN convention) and return [B] logits.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import video as tvv


_BUILDERS = {
    "r2plus1d_18": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights, "fc"),
    "r3d_18":      (tvv.r3d_18,      tvv.R3D_18_Weights,      "fc"),
    "mc3_18":      (tvv.mc3_18,      tvv.MC3_18_Weights,      "fc"),
    "s3d":         (tvv.s3d,         tvv.S3D_Weights,         "classifier"),
}

# Hugging Face VideoMAE checkpoints (Kinetics-400 finetuned).
_VIDEOMAE = {
    "videomae_base":  "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae_large": "MCG-NJU/videomae-large-finetuned-kinetics",
    # VideoMAE v2 ViT-B shares the v1 architecture; point this at any HF-format
    # v2 classification checkpoint to use it.
    "videomaev2_base": "OpenGVLab/VideoMAEv2-Base",
}


class AccidentNet(nn.Module):
    def __init__(self, arch: str, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.arch = arch
        self.is_videomae = arch in _VIDEOMAE
        if self.is_videomae:
            self._build_videomae(arch, pretrained, dropout)
        elif arch in _BUILDERS:
            self._build_conv3d(arch, pretrained, dropout)
        else:
            raise ValueError(f"unknown arch {arch}; choose "
                             f"{list(_BUILDERS) + list(_VIDEOMAE)}")

    # ---- builders ----
    def _build_conv3d(self, arch, pretrained, dropout):
        builder, weights_enum, head_attr = _BUILDERS[arch]
        weights = weights_enum.KINETICS400_V1 if pretrained else None
        self.backbone = builder(weights=weights)
        self.head_attr = head_attr
        if arch == "s3d":
            in_ch = self.backbone.classifier[1].in_channels
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Conv3d(in_ch, 1, kernel_size=1))
        else:
            in_feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_feat, 1))

    def _build_videomae(self, arch, pretrained, dropout):
        from transformers import VideoMAEForVideoClassification
        repo = _VIDEOMAE[arch]
        kwargs = dict(num_labels=1, ignore_mismatched_sizes=True)
        if not pretrained:
            from transformers import VideoMAEConfig
            cfg = VideoMAEConfig.from_pretrained(repo, num_labels=1)
            self.backbone = VideoMAEForVideoClassification(cfg)
        else:
            self.backbone = VideoMAEForVideoClassification.from_pretrained(repo, **kwargs)
        # add dropout before the (reinitialised) classifier head
        hidden = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.head_attr = "classifier"
        self.backbone.gradient_checkpointing_enable()

    # ---- forward ----
    def forward(self, x):
        if self.is_videomae:
            # [B,C,T,H,W] -> VideoMAE wants pixel_values [B,T,C,H,W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            return self.backbone(pixel_values=x).logits.flatten(1).squeeze(1)
        out = self.backbone(x)
        if self.arch == "s3d":
            out = out.mean(dim=(2, 3, 4)) if out.ndim > 2 else out
        return out.flatten(1).squeeze(1)

    # ---- discriminative LR / freezing ----
    def param_groups(self, base_lr: float, backbone_mult: float):
        head, bb = [], []
        for name, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            (head if name.startswith(self.head_attr) else bb).append(p)
        return [
            {"params": bb, "lr": base_lr * backbone_mult},
            {"params": head, "lr": base_lr},
        ]

    def set_backbone_frozen(self, frozen: bool):
        for name, p in self.backbone.named_parameters():
            if not name.startswith(self.head_attr):
                p.requires_grad = not frozen


def build_model(cfg) -> AccidentNet:
    return AccidentNet(cfg.model.arch, cfg.model.pretrained, cfg.model.dropout)
