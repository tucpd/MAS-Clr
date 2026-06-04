"""
Unified Detector for Multi-Task Inference

FCOS-style vectorized decode with NMS for person and face detection.
Single forward pass → detections, with optional teacher recognition from face bbox.
"""

import json
import cv2
import torch
import numpy as np
from torchvision.ops import nms
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import time

from .unified_model import create_unified_model
from .heads import STRIDES

try:
    from ..old_architecture.arcface_recognizer import ArcFaceRecognizer, ArcFaceConfig
    ARCFACE_MODULE_AVAILABLE = True
except Exception:
    ArcFaceRecognizer = None
    ArcFaceConfig = None
    ARCFACE_MODULE_AVAILABLE = False


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
    """Face detection result"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    track_id: Optional[int] = None


@dataclass
class CameraAgentOutput:
    """Structured camera output for downstream control agent."""
    num_students: int = 0
    teacher_id: Optional[str] = None
    teacher_confidence: Optional[float] = None
    teacher_preferences: Dict[str, Any] = field(default_factory=dict)
    gesture_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_students': self.num_students,
            'teacher_id': self.teacher_id,
            'teacher_confidence': self.teacher_confidence,
            'teacher_preferences': self.teacher_preferences,
            'gesture_command': self.gesture_command,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Unified Detector
# ---------------------------------------------------------------------------

class UnifiedDetector:
    """
    Unified detector: single forward pass → person + face detections

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
        enable_teacher_recognition: bool = False,
        teacher_db_path: Optional[str] = None,
        teacher_preferences_path: Optional[str] = None,
        teacher_recognition_threshold: float = 0.5,
    ):
        """
        Args:
            backbone_name: Backbone architecture name (timm)
            checkpoint_path: Trained model checkpoint path (optional)
            device: 'cuda' or 'cpu'
            confidence_threshold: Min score (cls × centerness) to keep
            nms_iou: IoU threshold for NMS
            image_size: Input image size (square)
            enable_teacher_recognition: Enable ArcFace teacher recognition
            teacher_db_path: Path to ArcFace teacher embedding DB (pkl)
            teacher_preferences_path: Path to teacher preference DB (json/pkl)
            teacher_recognition_threshold: Cosine threshold for recognition
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_iou = nms_iou
        self.image_size = image_size
        self.enable_teacher_recognition = enable_teacher_recognition

        default_teacher_db = Path(__file__).resolve().parents[1] / 'teacher_database.pkl'
        self.teacher_db_path = str(teacher_db_path or default_teacher_db)
        self.teacher_preferences_path = teacher_preferences_path
        self.teacher_recognition_threshold = teacher_recognition_threshold
        self.teacher_preferences: Dict[str, Dict[str, Any]] = {}
        self.arcface_recognizer = None

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

        # Optional teacher recognition stack
        self._init_teacher_recognition()

        print(f"Unified detector ready "
              f"(conf={confidence_threshold}, nms={nms_iou})")

    def _init_teacher_recognition(self):
        """Initialize optional ArcFace recognizer and teacher preference DB."""
        self.teacher_preferences = self._load_teacher_preferences(
            self.teacher_preferences_path,
        )

        if not self.enable_teacher_recognition:
            return

        if not ARCFACE_MODULE_AVAILABLE or ArcFaceConfig is None:
            print('ArcFace module unavailable. Teacher recognition disabled.')
            self.enable_teacher_recognition = False
            return

        try:
            arcface_config = ArcFaceConfig()
            arcface_config.recognition_threshold = self.teacher_recognition_threshold
            arcface_config.device = self.device

            recognizer = ArcFaceRecognizer(
                config=arcface_config,
                database_path=self.teacher_db_path,
            )
            if recognizer.initialize():
                self.arcface_recognizer = recognizer
            else:
                print('Failed to initialize ArcFace recognizer. Teacher recognition disabled.')
                self.enable_teacher_recognition = False
        except Exception as exc:
            print(f'ArcFace initialization error: {exc}. Teacher recognition disabled.')
            self.enable_teacher_recognition = False

    @staticmethod
    def _load_teacher_preferences(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Load teacher preference mapping from json/pkl if provided."""
        if not path:
            return {}

        pref_path = Path(path)
        if not pref_path.exists():
            return {}

        try:
            if pref_path.suffix.lower() == '.json':
                raw = json.loads(pref_path.read_text(encoding='utf-8'))
            elif pref_path.suffix.lower() in {'.pkl', '.pickle'}:
                import pickle
                with pref_path.open('rb') as f:
                    raw = pickle.load(f)
            else:
                return {}
        except Exception:
            return {}

        if not isinstance(raw, dict):
            return {}

        normalized: Dict[str, Dict[str, Any]] = {}
        for teacher_id, value in raw.items():
            if isinstance(value, dict):
                if isinstance(value.get('preferences'), dict):
                    normalized[str(teacher_id)] = value['preferences']
                else:
                    normalized[str(teacher_id)] = value
        return normalized

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
            grid_cx    [N]     —  grid center x
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
        """Decode face detections from FPN levels + NMS"""
        levels = self.feat_levels['face']
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []

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

            boxes, scores, *_ = result

            all_boxes.append(boxes)
            all_scores.append(scores)

        if not all_boxes:
            return []

        boxes = torch.cat(all_boxes) / scale
        scores = torch.cat(all_scores)

        boxes[:, 0].clamp_(0, orig_shape[1])
        boxes[:, 1].clamp_(0, orig_shape[0])
        boxes[:, 2].clamp_(0, orig_shape[1])
        boxes[:, 3].clamp_(0, orig_shape[0])

        keep = nms(boxes, scores, self.nms_iou)

        return [
            FaceDetection(
                bbox=boxes[k].cpu().numpy(),
                confidence=scores[k].item(),
            )
            for k in keep
        ]

    def recognize_teacher(
        self,
        image: np.ndarray,
        faces: List[FaceDetection],
    ) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
        """Recognize teacher from detected face bboxes using ArcFace."""
        if (
            not self.enable_teacher_recognition
            or self.arcface_recognizer is None
            or not faces
        ):
            return None, None, {}

        h, w = image.shape[:2]
        sorted_faces = sorted(faces, key=lambda f: f.confidence, reverse=True)
        best_similarity: Optional[float] = None

        for face in sorted_faces[:3]:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            arc_faces = self.arcface_recognizer.detect_and_extract(
                image,
                roi=(x1, y1, x2, y2),
            )
            if not arc_faces:
                continue

            for arc_face in arc_faces:
                teacher_id, similarity = self.arcface_recognizer.recognize_face(
                    arc_face['embedding'],
                )
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity

                if teacher_id is not None:
                    preferences = self.teacher_preferences.get(teacher_id, {})
                    return teacher_id, float(similarity), preferences

        return None, best_similarity, {}

    def build_camera_output(
        self,
        image: np.ndarray,
        people: List[PersonDetection],
        faces: List[FaceDetection],
    ) -> CameraAgentOutput:
        """Build downstream JSON-compatible camera output."""
        teacher_id, teacher_confidence, preferences = self.recognize_teacher(
            image,
            faces,
        )
        return CameraAgentOutput(
            num_students=len(people),
            teacher_id=teacher_id,
            teacher_confidence=teacher_confidence,
            teacher_preferences=preferences,
            gesture_command=None,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect(
        self, image: np.ndarray,
    ) -> Tuple[List[PersonDetection], List[FaceDetection], float]:
        """
        Unified detection on single image
        
        Args:
            image: BGR image [H, W, 3]
            
        Returns:
            (people, faces, inference_time_ms)
        """
        start = time.time()
        
        # Preprocess
        img_tensor, scale, orig_shape = self.preprocess(image)
        
        # Forward pass
        outputs = self.model.inference(img_tensor)
        
        # Post-process each task
        people = self.detect_people(outputs, scale, orig_shape)
        faces = self.detect_faces(outputs, scale, orig_shape)

        inference_time = (time.time() - start) * 1000
        return people, faces, inference_time

    @torch.no_grad()
    def detect_with_output(
        self,
        image: np.ndarray,
    ) -> Tuple[List[PersonDetection], List[FaceDetection], CameraAgentOutput, float]:
        """Run detection and return camera output for control-agent integration."""
        people, faces, inference_time = self.detect(image)
        camera_output = self.build_camera_output(image, people, faces)
        return people, faces, camera_output, inference_time


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
    people, faces, t = detector.detect(test_image)

    print(f"Results:")
    print(f"  People: {len(people)}")
    print(f"  Faces:  {len(faces)}")
    print(f"  Time:   {t:.2f} ms")

    print("\nBenchmarking (50 iterations)...")
    times = []
    for _ in range(50):
        _, _, t = detector.detect(test_image)
        times.append(t)

    print(f"Average: {np.mean(times):.2f} ms")
    print(f"FPS: {1000 / np.mean(times):.1f}")

    _, _, cam_out, _ = detector.detect_with_output(test_image)
    print("\nCamera output:")
    print(cam_out.to_json())
