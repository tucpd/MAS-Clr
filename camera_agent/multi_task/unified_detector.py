"""
Unified Detector for Multi-Task Inference

FCOS-style vectorized decode with NMS for person, face, and hand detection.
Single forward pass → all detections.
"""

import cv2
import torch
import numpy as np
from torchvision.ops import nms
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from .unified_model import create_unified_model
from .heads import STRIDES


# ---------------------------------------------------------------------------
# Detection dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PersonDetection:
    """Person detection result"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    track_id: Optional[int] = None


@dataclass
class FaceDetection:
    """Face detection result with landmarks"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # [5, 2] - 5 facial landmarks (x, y)
    confidence: float
    track_id: Optional[int] = None


@dataclass
class HandDetection:
    """Hand detection result with keypoints"""
    bbox: np.ndarray  # [x1, y1, x2, y2] - bounding box around hand
    keypoints: np.ndarray  # [21, 2] - 21 hand keypoints (x, y)
    confidence: float
    visibility: np.ndarray  # [21] - visibility score for each keypoint


# ---------------------------------------------------------------------------
# Unified Detector
# ---------------------------------------------------------------------------

class UnifiedDetector:
    """
    Unified detector: single forward pass → person + face + hand detections

    Uses FCOS-style decoding:
      score = sigmoid(cls) × sigmoid(centerness)
      bbox  = grid_center ± ltrb

    Post-processing: NMS per task.
    """
    
    def __init__(
        self,
        backbone_name: str = 'mobilenetv3_large_100',
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.5,
        nms_iou: float = 0.5,
        image_size: int = 640,
    ):
        """
        Args:
            backbone_name: Backbone architecture name (timm)
            checkpoint_path: Trained model checkpoint path (optional)
            device: 'cuda' or 'cpu'
            confidence_threshold: Min score (cls × centerness) to keep
            nms_iou: IoU threshold for NMS
            image_size: Input image size (square)
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_iou = nms_iou
        self.image_size = image_size

        print(f"Loading unified model on {device}...")
        self.model = create_unified_model(
            backbone_name=backbone_name,
            pretrained=True,
            feature_channels=256,
            checkpoint_path=checkpoint_path,
        )
        self.model.to(device)
        self.model.eval()

        # Get feat_levels from model for each task
        self.feat_levels = self.model.feat_levels

        print(f"Unified detector ready "
              f"(conf={confidence_threshold}, nms={nms_iou})")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self, image: np.ndarray
    ) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """
        Letterbox resize + ImageNet normalise

        Returns:
            (tensor [1,3,S,S], scale_factor, original_shape (H,W))
        """
        h, w = image.shape[:2]
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        img = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        return img.unsqueeze(0).to(self.device), scale, (h, w)

    # ------------------------------------------------------------------
    # FCOS decode helper (vectorized, single FPN level)
    # ------------------------------------------------------------------

    def _decode_fcos_level(
        self,
        cls: torch.Tensor,   # [B, 1, H, W]
        reg: torch.Tensor,   # [B, 4, H, W]  (positive ltrb pixels)
        ctr: torch.Tensor,   # [B, 1, H, W]
        stride: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor,
                         torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Vectorized FCOS decode for one FPN level (batch index 0).

        Returns None if no detections pass the threshold, otherwise:
            boxes      [N, 4]  —  xyxy in input-image pixel coords
            scores     [N]
            grid_cx    [N]     —  grid center x (needed for landmark decode)
            grid_cy    [N]
            flat_mask  [H*W]   —  boolean mask in flattened spatial grid
        """
        H, W = cls.shape[2], cls.shape[3]

        # Score = sigmoid(cls) × sigmoid(centerness)
        scores = (
            torch.sigmoid(cls[0, 0]) * torch.sigmoid(ctr[0, 0])
        ).reshape(-1)                                   # [H*W]

        mask = scores > self.confidence_threshold
        if not mask.any():
            return None

        # Grid centers in input-image pixel coordinates
        yv, xv = torch.meshgrid(
            torch.arange(H, device=cls.device, dtype=torch.float32),
            torch.arange(W, device=cls.device, dtype=torch.float32),
            indexing='ij',
        )
        cx = (xv.reshape(-1) + 0.5) * stride           # [H*W]
        cy = (yv.reshape(-1) + 0.5) * stride

        valid_scores = scores[mask]
        valid_cx = cx[mask]
        valid_cy = cy[mask]

        reg_flat = reg[0].reshape(4, -1)                # [4, H*W]
        l, t, r, b = reg_flat[:, mask]                  # each [N]

        boxes = torch.stack([
            valid_cx - l, valid_cy - t,
            valid_cx + r, valid_cy + b,
        ], dim=1)                                       # [N, 4]

        return boxes, valid_scores, valid_cx, valid_cy, mask

    # ------------------------------------------------------------------
    # Per-task detection
    # ------------------------------------------------------------------

    def detect_people(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        scale: float,
        orig_shape: Tuple[int, int],
    ) -> List[PersonDetection]:
        """Decode person detections from all FPN levels + NMS"""
        levels = self.feat_levels['person']
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []

        for i, lvl in enumerate(levels):
            result = self._decode_fcos_level(
                outputs['person_cls'][i],
                outputs['person_reg'][i],
                outputs['person_ctr'][i],
                STRIDES[lvl],
            )
            if result is None:
                continue
            boxes, scores, *_ = result
            all_boxes.append(boxes)
            all_scores.append(scores)

        if not all_boxes:
            return []

        boxes = torch.cat(all_boxes) / scale
        scores = torch.cat(all_scores)

        # Clip to original image
        boxes[:, 0].clamp_(0, orig_shape[1])
        boxes[:, 1].clamp_(0, orig_shape[0])
        boxes[:, 2].clamp_(0, orig_shape[1])
        boxes[:, 3].clamp_(0, orig_shape[0])

        keep = nms(boxes, scores, self.nms_iou)

        return [
            PersonDetection(
                bbox=boxes[k].cpu().numpy(),
                confidence=scores[k].item(),
            )
            for k in keep
        ]

    def detect_faces(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        scale: float,
        orig_shape: Tuple[int, int],
    ) -> List[FaceDetection]:
        """Decode face detections + landmarks from FPN levels + NMS"""
        levels = self.feat_levels['face']
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_lmks: List[torch.Tensor] = []

        for i, lvl in enumerate(levels):
            stride = STRIDES[lvl]
            result = self._decode_fcos_level(
                outputs['face_cls'][i],
                outputs['face_reg'][i],
                outputs['face_ctr'][i],
                stride,
            )
            if result is None:
                continue

            boxes, scores, valid_cx, valid_cy, mask = result

            # Decode landmarks: grid_center + (offset × stride)
            lmk_flat = outputs['face_lmk'][i][0].reshape(10, -1)   # [10, H*W]
            valid_lmk = lmk_flat[:, mask].reshape(5, 2, -1)        # [5, 2, N]

            lmk_x = valid_cx.unsqueeze(0) + valid_lmk[:, 0, :] * stride
            lmk_y = valid_cy.unsqueeze(0) + valid_lmk[:, 1, :] * stride
            landmarks = torch.stack(
                [lmk_x, lmk_y], dim=2
            ).permute(1, 0, 2)                                     # [N, 5, 2]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_lmks.append(landmarks)

        if not all_boxes:
            return []

        boxes = torch.cat(all_boxes) / scale
        scores = torch.cat(all_scores)
        lmks = torch.cat(all_lmks) / scale

        boxes[:, 0].clamp_(0, orig_shape[1])
        boxes[:, 1].clamp_(0, orig_shape[0])
        boxes[:, 2].clamp_(0, orig_shape[1])
        boxes[:, 3].clamp_(0, orig_shape[0])
        lmks[:, :, 0].clamp_(0, orig_shape[1])
        lmks[:, :, 1].clamp_(0, orig_shape[0])

        keep = nms(boxes, scores, self.nms_iou)

        return [
            FaceDetection(
                bbox=boxes[k].cpu().numpy(),
                landmarks=lmks[k].cpu().numpy().astype(int),
                confidence=scores[k].item(),
            )
            for k in keep
        ]

    def detect_hands(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        scale: float,
        orig_shape: Tuple[int, int],
    ) -> List[HandDetection]:
        """Decode hand detections + keypoints from FPN levels + NMS"""
        levels = self.feat_levels['hand']
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_kpts: List[torch.Tensor] = []
        all_vis: List[torch.Tensor] = []

        for i, lvl in enumerate(levels):
            stride = STRIDES[lvl]
            result = self._decode_fcos_level(
                outputs['hand_cls'][i],
                outputs['hand_reg'][i],
                outputs['hand_ctr'][i],
                stride,
            )
            if result is None:
                continue

            boxes, scores, valid_cx, valid_cy, mask = result

            # Decode keypoint offsets: grid_center + (offset × stride)
            off_flat = outputs['hand_offsets'][i][0].reshape(42, -1)
            hm_flat = outputs['hand_heatmaps'][i][0].reshape(21, -1)

            valid_off = off_flat[:, mask].reshape(21, 2, -1)   # [21, 2, N]
            valid_hm = hm_flat[:, mask]                        # [21, N]

            kpt_x = valid_cx.unsqueeze(0) + valid_off[:, 0, :] * stride
            kpt_y = valid_cy.unsqueeze(0) + valid_off[:, 1, :] * stride
            keypoints = torch.stack(
                [kpt_x, kpt_y], dim=2
            ).permute(1, 0, 2)                                 # [N, 21, 2]

            visibility = torch.sigmoid(valid_hm).permute(1, 0) # [N, 21]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_kpts.append(keypoints)
            all_vis.append(visibility)

        if not all_boxes:
            return []

        boxes = torch.cat(all_boxes) / scale
        scores = torch.cat(all_scores)
        kpts = torch.cat(all_kpts) / scale
        vis = torch.cat(all_vis)

        boxes[:, 0].clamp_(0, orig_shape[1])
        boxes[:, 1].clamp_(0, orig_shape[0])
        boxes[:, 2].clamp_(0, orig_shape[1])
        boxes[:, 3].clamp_(0, orig_shape[0])
        kpts[:, :, 0].clamp_(0, orig_shape[1])
        kpts[:, :, 1].clamp_(0, orig_shape[0])

        keep = nms(boxes, scores, self.nms_iou)

        return [
            HandDetection(
                bbox=boxes[k].cpu().numpy(),
                keypoints=kpts[k].cpu().numpy().astype(int),
                confidence=scores[k].item(),
                visibility=vis[k].cpu().numpy(),
            )
            for k in keep
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect(
        self, image: np.ndarray,
    ) -> Tuple[List[PersonDetection], List[FaceDetection],
               List[HandDetection], float]:
        """
        Unified detection on single image
        
        Args:
            image: BGR image [H, W, 3]
            
        Returns:
            (people, faces, hands, inference_time_ms)
        """
        start = time.time()
        
        # Preprocess
        img_tensor, scale, orig_shape = self.preprocess(image)
        
        # Forward pass
        outputs = self.model.inference(img_tensor)
        
        # Post-process each task
        people = self.detect_people(outputs, scale, orig_shape)
        faces = self.detect_faces(outputs, scale, orig_shape)
        hands = self.detect_hands(outputs, scale, orig_shape)

        inference_time = (time.time() - start) * 1000
        return people, faces, hands, inference_time


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing Unified Detector (FCOS-style)...\n")

    detector = UnifiedDetector(
        backbone_name='mobilenetv3_large_100',
        device='cpu',
        confidence_threshold=0.5,
        nms_iou=0.5,
    )
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    people, faces, hands, t = detector.detect(test_image)

    print(f"Results:")
    print(f"  People: {len(people)}")
    print(f"  Faces:  {len(faces)}")
    print(f"  Hands:  {len(hands)}")
    print(f"  Time:   {t:.2f} ms")

    print("\nBenchmarking (50 iterations)...")
    times = []
    for _ in range(50):
        _, _, _, t = detector.detect(test_image)
        times.append(t)

    print(f"Average: {np.mean(times):.2f} ms")
    print(f"FPS: {1000 / np.mean(times):.1f}")
