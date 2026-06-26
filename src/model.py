"""Model factory: Kinetics-pretrained video backbone with a binary head.

Three families are supported behind one interface (single-logit output, the same
param_groups / set_backbone_frozen API):

  * torchvision 3D-CNNs  (r2plus1d_18 / r3d_18 / mc3_18 / s3d) — pure-conv, the
    lightest path, cleanest TensorRT export. Best for Jetson Orin Nano.
  * VideoMAE ViT         (videomae_base / videomae_large) — self-supervised
    transformer, higher Kinetics accuracy, heavier compute. Needs `transformers`.
  * 2D-CNN + temporal head (mnv3s_temporal / mnv3l_temporal / resnet18_temporal)
    — a per-frame ImageNet 2D backbone + a small temporal head (1D-conv / GRU /
    pool). NO 3D convolutions, so it maps onto the Rockchip RK3588 NPU: the 2D
    backbone runs on the NPU (4D tensors, INT8) and the tiny head runs on CPU.

All take input [B, 3, T, H, W] (our 3D-CNN convention) and return [B] logits.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models as tvm
from torchvision.models import video as tvv


_BUILDERS = {
    "r2plus1d_18": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights, "fc"),
    "r3d_18":      (tvv.r3d_18,      tvv.R3D_18_Weights,      "fc"),
    "mc3_18":      (tvv.mc3_18,      tvv.MC3_18_Weights,      "fc"),
    "s3d":         (tvv.s3d,         tvv.S3D_Weights,         "classifier"),
}

# 2D per-frame backbones (ImageNet) -> (builder, weights_enum, pooled feature dim).
_BUILDERS2D = {
    "mnv3s_temporal":    (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights, 576),
    "mnv3l_temporal":    (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights, 960),
    "resnet18_temporal": (tvm.resnet18,           tvm.ResNet18_Weights,           512),
}


def _inflate_first_conv(conv: nn.Conv2d, in_chans: int) -> nn.Conv2d:
    """Replace a 3-channel first conv with an `in_chans` one, reusing pretrained
    weights: channels 0-2 copy RGB, extra channels copy RGB filters (so the
    motion-difference channels start from sensible edge detectors)."""
    if in_chans == conv.in_channels:
        return conv
    new = nn.Conv2d(in_chans, conv.out_channels, conv.kernel_size,
                    conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        for j in range(in_chans):
            new.weight[:, j] = conv.weight[:, j % conv.in_channels]
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class _Backbone2D(nn.Module):
    """2D CNN feature extractor: [N,Cin,H,W] -> [N, c_out] pooled features.

    This is the part that runs on the RK3588 NPU (4D in/out, INT8-friendly).
    `in_chans` is 3 (RGB) or 6 (RGB + temporal-difference motion channels)."""

    def __init__(self, arch, pretrained, in_chans=3):
        super().__init__()
        builder, weights_enum, c_out = _BUILDERS2D[arch]
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        net = builder(weights=weights)
        if arch.startswith("resnet"):
            net.conv1 = _inflate_first_conv(net.conv1, in_chans)
            self.features = nn.Sequential(*list(net.children())[:-1])  # -> [N,c,1,1]
            self.pool = nn.Identity()
        else:                                                          # mobilenet_v3
            net.features[0][0] = _inflate_first_conv(net.features[0][0], in_chans)
            self.features = net.features
            self.pool = nn.AdaptiveAvgPool2d(1)
        self.c_out = c_out

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


class TemporalHead(nn.Module):
    """Aggregate per-frame features [B,T,c_in] over time -> [B] logit.

    Tiny by design (runs on CPU after the NPU backbone). `tconv` (default) uses a
    causal-window 1D conv stack + global temporal max-pool — sensitive to the
    motion spike of a collision while staying export-friendly."""

    def __init__(self, c_in, head, feat_dim, dropout):
        super().__init__()
        self.head = head
        if head == "tpool":
            self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, 1))
        elif head == "tconv":
            self.tcn = nn.Sequential(
                nn.Conv1d(c_in, feat_dim, 3, padding=1), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
                nn.Conv1d(feat_dim, feat_dim, 3, padding=1), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
            )
            self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))
        elif head == "gru":
            self.gru = nn.GRU(c_in, feat_dim, batch_first=True)
            self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))
        else:
            raise ValueError(f"unknown temporal_head {head}; choose tpool|tconv|gru")

    def forward(self, feats):                       # feats [B,T,c_in]
        if self.head == "tpool":
            h = feats.mean(dim=1)
        elif self.head == "tconv":
            h = self.tcn(feats.transpose(1, 2)).amax(dim=2)   # [B,feat_dim]
        else:                                                  # gru
            out, _ = self.gru(feats)
            h = out[:, -1]
        return self.fc(h).squeeze(1)


class Frame2DTemporal(nn.Module):
    """Per-frame 2D backbone + temporal head, exposing the split for NPU export."""

    def __init__(self, arch, pretrained, dropout, head, feat_dim, in_chans=3):
        super().__init__()
        self.backbone = _Backbone2D(arch, pretrained, in_chans)
        self.temporal = TemporalHead(self.backbone.c_out, head, feat_dim, dropout)
        self.c_out = self.backbone.c_out
        self.in_chans = in_chans

    def forward_frames(self, frames):               # [N,Cin,H,W] -> [N,c_in]
        return self.backbone(frames)

    def forward(self, x):                           # [B,Cin,T,H,W] -> [B]
        B, C, T, H, W = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.backbone(flat).reshape(B, T, self.c_out)
        return self.temporal(feats)

# Hugging Face VideoMAE checkpoints (Kinetics-400 finetuned).
_VIDEOMAE = {
    "videomae_base":  "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae_large": "MCG-NJU/videomae-large-finetuned-kinetics",
    # VideoMAE v2 ViT-B shares the v1 architecture; point this at any HF-format
    # v2 classification checkpoint to use it.
    "videomaev2_base": "OpenGVLab/VideoMAEv2-Base",
}


class AccidentNet(nn.Module):
    def __init__(self, arch: str, pretrained: bool = True, dropout: float = 0.5,
                 temporal: dict | None = None):
        super().__init__()
        self.arch = arch
        self.is_videomae = arch in _VIDEOMAE
        self.is_frame2d = arch in _BUILDERS2D
        if self.is_videomae:
            self._build_videomae(arch, pretrained, dropout)
        elif self.is_frame2d:
            t = temporal or {}
            self.backbone = Frame2DTemporal(
                arch, pretrained, dropout,
                t.get("head", "tconv"), int(t.get("feat_dim", 256)),
                in_chans=int(t.get("in_chans", 3)))
            self.head_attr = "temporal"             # head params live under .temporal
        elif arch in _BUILDERS:
            self._build_conv3d(arch, pretrained, dropout)
        else:
            raise ValueError(f"unknown arch {arch}; choose "
                             f"{list(_BUILDERS) + list(_BUILDERS2D) + list(_VIDEOMAE)}")

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
        if self.is_frame2d:
            return self.backbone(x)                 # already [B]
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
    m = cfg.model
    in_chans = 6 if bool(cfg.input.get("motion", False)) else 3
    temporal = {"head": m.get("temporal_head", "tconv"),
                "feat_dim": int(m.get("feat_dim", 256)),
                "in_chans": in_chans}
    return AccidentNet(m.arch, m.pretrained, m.dropout, temporal=temporal)
