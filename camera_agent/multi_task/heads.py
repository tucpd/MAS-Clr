"""
Task-specific detection heads for multi-task model (FCOS-style)

Conventions:
  - Classification: 1 channel, raw logit → sigmoid = probability
  - Bbox regression: 4 channels (l, t, r, b) distances in pixels from grid center
    Head applies exp(clamp(raw)) * learnable_scale → always positive
  - Centerness: 1 channel, raw logit → sigmoid = center-ness score [0,1]

Each head applies shared conv weights across multiple FPN levels (FCOS/RetinaNet style).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# FPN level strides (determined by backbone + FPN)
STRIDES = {'p3': 8, 'p4': 16, 'p5': 32}


class ConvSubnet(nn.Module):
    """Shared convolutional subnet: N × (Conv3x3 + GroupNorm + ReLU)"""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            ch_in = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv2d(ch_in, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            ])
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Person Head
# ---------------------------------------------------------------------------

class PersonHead(nn.Module):
    """
    FCOS-style person detection head — multi-scale (p3, p4, p5)

    Outputs per FPN level:
      person_cls  [B, 1, H, W]  classification logit
      person_reg  [B, 4, H, W]  ltrb distances (pixels, positive)
      person_ctr  [B, 1, H, W]  centerness logit
    """

    def __init__(
        self,
        in_channels: int = 256,
        feat_levels: Tuple[str, ...] = ('p3', 'p4', 'p5'),
    ):
        super().__init__()
        self.feat_levels = feat_levels

        # Separate cls / reg subnets (FCOS convention)
        self.cls_subnet = ConvSubnet(in_channels, 256, num_convs=2)
        self.reg_subnet = ConvSubnet(in_channels, 256, num_convs=2)

        # Output heads
        self.cls_head = nn.Conv2d(256, 1, 3, padding=1)
        self.reg_head = nn.Conv2d(256, 4, 3, padding=1)
        self.ctr_head = nn.Conv2d(256, 1, 3, padding=1)

        # Learnable scale per FPN level
        self.scales = nn.ParameterDict({
            lvl: nn.Parameter(torch.ones(1)) for lvl in feat_levels
        })
        self._init_heads()

    def _init_heads(self):
        for m in [self.cls_head, self.reg_head, self.ctr_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        # Focal-loss prior: sigmoid(-4.6) ≈ 0.01
        nn.init.constant_(self.cls_head.bias, -4.6)

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        cls_out, reg_out, ctr_out = [], [], []
        for lvl in self.feat_levels:
            cls_feat = self.cls_subnet(features[lvl])
            reg_feat = self.reg_subnet(features[lvl])

            cls_out.append(self.cls_head(cls_feat))
            reg_out.append(
                torch.exp(self.reg_head(reg_feat).clamp(max=8.0))
                * self.scales[lvl]
            )
            ctr_out.append(self.ctr_head(cls_feat))

        return {
            'person_cls': cls_out,
            'person_reg': reg_out,
            'person_ctr': ctr_out,
        }


# ---------------------------------------------------------------------------
# Face Head
# ---------------------------------------------------------------------------

class FaceHead(nn.Module):
    """
    FCOS-style face detection head — multi-scale (p3, p4)

    Outputs per FPN level:
      face_cls  [B, 1, H, W]   classification logit
      face_reg  [B, 4, H, W]   ltrb distances (pixels)
      face_ctr  [B, 1, H, W]   centerness logit
    """

    def __init__(
        self,
        in_channels: int = 256,
        feat_levels: Tuple[str, ...] = ('p3', 'p4'),
    ):
        super().__init__()
        self.feat_levels = feat_levels

        self.cls_subnet = ConvSubnet(in_channels, 256, num_convs=2)
        self.reg_subnet = ConvSubnet(in_channels, 256, num_convs=2)

        self.cls_head = nn.Conv2d(256, 1, 3, padding=1)
        self.reg_head = nn.Conv2d(256, 4, 3, padding=1)
        self.ctr_head = nn.Conv2d(256, 1, 3, padding=1)

        self.scales = nn.ParameterDict({
            lvl: nn.Parameter(torch.ones(1)) for lvl in feat_levels
        })
        self._init_heads()

    def _init_heads(self):
        for m in [self.cls_head, self.reg_head, self.ctr_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_head.bias, -4.6)

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        cls_out, reg_out, ctr_out = [], [], []
        for lvl in self.feat_levels:
            cls_feat = self.cls_subnet(features[lvl])
            reg_feat = self.reg_subnet(features[lvl])

            cls_out.append(self.cls_head(cls_feat))
            reg_out.append(
                torch.exp(self.reg_head(reg_feat).clamp(max=8.0))
                * self.scales[lvl]
            )
            ctr_out.append(self.ctr_head(cls_feat))

        return {
            'face_cls': cls_out, 'face_reg': reg_out,
            'face_ctr': ctr_out,
        }


# ---------------------------------------------------------------------------
# Multi-Task Head (combines all heads)
# ---------------------------------------------------------------------------

class MultiTaskHead(nn.Module):
    """Combines PersonHead + FaceHead"""

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.person_head = PersonHead(in_channels=in_channels)
        self.face_head = FaceHead(in_channels=in_channels)

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        outputs = {}
        outputs.update(self.person_head(features))
        outputs.update(self.face_head(features))
        return outputs


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing FCOS-style Multi-Task Heads...\n")

    batch = 2
    features = {
        'p3': torch.randn(batch, 256, 80, 80),
        'p4': torch.randn(batch, 256, 40, 40),
        'p5': torch.randn(batch, 256, 20, 20),
    }

    for name, Head in [('PersonHead', PersonHead), ('FaceHead', FaceHead)]:
        head = Head(256)
        out = head(features)
        print(f"{name} (levels={head.feat_levels}):")
        for k, v_list in out.items():
            shapes = ', '.join(str(v.shape) for v in v_list)
            print(f"  {k}: [{shapes}]")
        print()

    print("MultiTaskHead:")
    mt = MultiTaskHead(256)
    mo = mt(features)
    for k, v_list in mo.items():
        shapes = ', '.join(str(v.shape) for v in v_list)
        print(f"  {k}: [{shapes}]")

    total = sum(p.numel() for p in mt.parameters())
    print(f"\nTotal head parameters: {total:,}")
