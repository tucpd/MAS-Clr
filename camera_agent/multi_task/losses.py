"""
Loss functions for multi-task model training

Individual losses:
  FocalLoss      — Dense classification (handles foreground/background imbalance)
  GIoULoss       — Bounding-box regression (scale-invariant, better gradients)
  CenternessLoss — BCE for FCOS centerness prediction
  WingLoss       — Facial landmark regression (precise for small errors)

Composition:
  MultiTaskLoss  — Uncertainty-weighted combination (Kendall et al., 2018)
                   Learns per-task σ to auto-balance heterogeneous loss magnitudes
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017)
    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    Reduces contribution from easy negatives → focuses training on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Raw logits, any shape
            target: Binary labels (0 or 1), same shape
        """
        p = torch.sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


# ---------------------------------------------------------------------------
# GIoU Loss
# ---------------------------------------------------------------------------

class GIoULoss(nn.Module):
    """
    Generalized IoU Loss (Rezatofighi et al., 2019)
    L = 1 − GIoU

    Gradient-friendly bbox regression; works even when boxes don't overlap.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Predicted boxes [N, 4] (x1 y1 x2 y2)
            target: GT boxes [N, 4] (x1 y1 x2 y2)
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0

        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # Areas
        pred_area = (pred[:, 2] - pred[:, 0]).clamp(0) * \
                    (pred[:, 3] - pred[:, 1]).clamp(0)
        tgt_area = (target[:, 2] - target[:, 0]).clamp(0) * \
                   (target[:, 3] - target[:, 1]).clamp(0)
        union = pred_area + tgt_area - inter + 1e-7

        iou = inter / union

        # Enclosing box
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])
        enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0) + 1e-7

        giou = iou - (enc_area - union) / enc_area
        loss = 1 - giou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Centerness Loss
# ---------------------------------------------------------------------------

class CenternessLoss(nn.Module):
    """
    Centerness loss (FCOS) — BCE on centerness prediction.

    Centerness target = sqrt( min(l,r)/max(l,r) × min(t,b)/max(t,b) )
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Centerness logits [N]
            target: Centerness targets [N] in [0, 1]
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return self.bce(pred, target)

    @staticmethod
    def compute_targets(ltrb: torch.Tensor) -> torch.Tensor:
        """
        Compute centerness targets from FCOS ltrb regression targets.

        Args:
            ltrb: [N, 4] — (left, top, right, bottom) distances
        Returns:
            centerness: [N] in [0, 1]
        """
        l, t, r, b = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
        lr = torch.min(l, r) / (torch.max(l, r) + 1e-7)
        tb = torch.min(t, b) / (torch.max(t, b) + 1e-7)
        return torch.sqrt(lr * tb)


# ---------------------------------------------------------------------------
# Wing Loss
# ---------------------------------------------------------------------------

class WingLoss(nn.Module):
    """
    Wing Loss (Feng et al., 2018)
    Designed for facial landmark regression.

    L(x) = w · ln(1 + |x|/ε)   if |x| < w
           |x| − C              otherwise
    where C = w − w · ln(1 + w/ε)

    Provides stronger gradient for small-medium errors than L1/L2.
    """

    def __init__(self, w: float = 10.0, epsilon: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Predicted landmarks [N, num_points × 2]
            target: GT landmarks        [N, num_points × 2]
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0

        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log1p(diff / self.epsilon),
            diff - self.C,
        )

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Multi-Task Loss (Uncertainty Weighting)
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss combiner (Kendall et al., 2018).

    L_total = Σ_i  (1 / (2·σ²_i)) · L_i  +  log(σ_i)

    σ_i is learned per sub-task, automatically balancing loss magnitudes.
    The module stores log(σ²) as parameters for numerical stability.

    Usage:
        loss_fn = MultiTaskLoss()

        # Compute individual losses externally
        losses = {
            'person_cls': focal_loss_val,
            'person_reg': giou_loss_val,
            'person_ctr': ctr_loss_val,
            'face_cls':   ...,
            ...
        }
        total, loss_dict = loss_fn(losses)
        total.backward()
    """

    # Default sub-task names matching the head outputs
    DEFAULT_TASKS = [
        'person_cls', 'person_reg', 'person_ctr',
        'face_cls',   'face_reg',   'face_ctr',  'face_lmk',
        'hand_cls',   'hand_reg',   'hand_ctr',  'hand_kpt', 'hand_hm',
    ]

    def __init__(self, task_names: List[str] = None):
        super().__init__()
        task_names = task_names or self.DEFAULT_TASKS

        # Learnable log(σ²) per sub-task — init 0 → σ=1 → equal weight
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def forward(
        self, losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            losses: {task_name: scalar loss tensor}
        Returns:
            total_loss:  Scalar for backward()
            loss_dict:   Per-task loss + weight values for logging
        """
        device = next(iter(self.parameters())).device
        total = torch.tensor(0.0, device=device)
        info: Dict[str, float] = {}

        for name, loss_val in losses.items():
            if name not in self.log_vars:
                continue
            precision = torch.exp(-self.log_vars[name])     # 1 / σ²
            weighted = 0.5 * precision * loss_val + 0.5 * self.log_vars[name]
            total = total + weighted

            info[name] = loss_val.item()
            info[f'{name}_w'] = precision.item()

        info['total'] = total.item()
        return total, info


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing loss functions...\n")

    # --- Focal Loss ---
    fl = FocalLoss()
    pred = torch.randn(100)
    tgt = (torch.rand(100) > 0.9).float()  # 90% negative
    print(f"FocalLoss:      {fl(pred, tgt).item():.4f}")

    # --- GIoU Loss ---
    gl = GIoULoss()
    pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float)
    gt_boxes = torch.tensor([[12, 12, 48, 48], [22, 22, 62, 62]], dtype=torch.float)
    print(f"GIoULoss:       {gl(pred_boxes, gt_boxes).item():.4f}")

    # --- Centerness Loss ---
    cl = CenternessLoss()
    ltrb = torch.tensor([[10, 20, 10, 20], [5, 30, 15, 10]], dtype=torch.float)
    ctr_target = CenternessLoss.compute_targets(ltrb)
    ctr_pred = torch.randn(2)
    print(f"CtrLoss:        {cl(ctr_pred, ctr_target).item():.4f}")
    print(f"  targets:      {ctr_target.tolist()}")

    # --- Wing Loss ---
    wl = WingLoss()
    pred_lmk = torch.randn(5, 10)
    gt_lmk = pred_lmk + torch.randn_like(pred_lmk) * 2
    print(f"WingLoss:       {wl(pred_lmk, gt_lmk).item():.4f}")

    # --- MultiTaskLoss ---
    mt = MultiTaskLoss()
    dummy_losses = {
        'person_cls': torch.tensor(1.5),
        'person_reg': torch.tensor(0.8),
        'person_ctr': torch.tensor(0.3),
        'face_cls':   torch.tensor(1.2),
        'face_reg':   torch.tensor(0.6),
    }
    total, info = mt(dummy_losses)
    print(f"\nMultiTaskLoss:")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
