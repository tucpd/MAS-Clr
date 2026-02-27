"""
Unified Detector for Multi-Task Inference
Single detector for person, face, and hand detection
"""

import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from .unified_model import create_unified_model


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


class UnifiedDetector:
    """
    Unified detector for person, face, and hand detection
    
    Single forward pass detects all three types of objects efficiently
    """
    
    def __init__(
        self,
        backbone_name: str = 'mobilenetv3_large_100',
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.5,
        image_size: int = 640
    ):
        """
        Args:
            backbone_name: Backbone architecture name
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
            image_size: Input image size for model
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        
        # Create model
        print(f"Loading unified model on {device}...")
        self.model = create_unified_model(
            backbone_name=backbone_name,
            pretrained=True,
            feature_channels=256,
            checkpoint_path=checkpoint_path
        )
        self.model.to(device)
        self.model.eval()
        
        print(f"Unified detector ready!")
        print(f"Image size: {image_size}")
        print(f"Confidence threshold: {confidence_threshold}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """
        Preprocess image for model input
        
        Args:
            image: BGR image [H, W, 3]
            
        Returns:
            - Preprocessed tensor [1, 3, size, size]
            - Scale factor for coordinate conversion
            - Original image shape (H, W)
        """
        h, w = image.shape[:2]
        
        # Resize while maintaining aspect ratio
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        
        # Convert to tensor
        img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float()
        img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(self.device), scale, (h, w)
    
    def detect_people(
        self, 
        outputs: Dict[str, torch.Tensor],
        scale: float,
        orig_shape: Tuple[int, int]
    ) -> List[PersonDetection]:
        """
        Extract person detections from model outputs
        
        Args:
            outputs: Model output dictionary
            scale: Scale factor for coordinate conversion
            orig_shape: Original image shape (H, W)
            
        Returns:
            List of PersonDetection objects
        """
        person_cls = outputs['person_cls'][0]  # [2, H, W]
        person_reg = outputs['person_reg'][0]  # [4, H, W]
        
        # Get confidence scores (objectness * class probability)
        person_conf = torch.sigmoid(person_cls[1])  # [H, W]
        
        # Find high-confidence detections
        mask = person_conf > self.confidence_threshold
        indices = torch.nonzero(mask)  # [N, 2] (y, x)
        
        detections = []
        for idx in indices:
            y, x = idx[0].item(), idx[1].item()
            conf = person_conf[y, x].item()
            
            # Get bounding box
            bbox_pred = person_reg[:, y, x].cpu().numpy()
            
            # Convert to image coordinates
            stride = self.image_size / person_cls.shape[1]
            cx = (x + 0.5) * stride / scale
            cy = (y + 0.5) * stride / scale
            w = bbox_pred[2] * self.image_size / scale
            h = bbox_pred[3] * self.image_size / scale
            
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            
            detections.append(PersonDetection(
                bbox=np.array([x1, y1, x2, y2]),
                confidence=conf
            ))
        
        return detections
    
    def detect_faces(
        self,
        outputs: Dict[str, torch.Tensor],
        scale: float,
        orig_shape: Tuple[int, int]
    ) -> List[FaceDetection]:
        """
        Extract face detections from model outputs
        
        Args:
            outputs: Model output dictionary
            scale: Scale factor for coordinate conversion
            orig_shape: Original image shape (H, W)
            
        Returns:
            List of FaceDetection objects
        """
        face_cls = outputs['face_cls'][0]  # [2, H, W]
        face_bbox = outputs['face_bbox'][0]  # [4, H, W]
        face_landmarks = outputs['face_landmarks'][0]  # [10, H, W]
        
        # Get confidence scores
        face_conf = torch.sigmoid(face_cls[1])  # [H, W]
        
        # Find high-confidence detections
        mask = face_conf > self.confidence_threshold
        indices = torch.nonzero(mask)  # [N, 2] (y, x)
        
        detections = []
        for idx in indices:
            y, x = idx[0].item(), idx[1].item()
            conf = face_conf[y, x].item()
            
            # Get bounding box
            bbox_pred = face_bbox[:, y, x].cpu().numpy()
            
            # Get landmarks (5 points, each with x, y)
            landmarks_pred = face_landmarks[:, y, x].cpu().numpy().reshape(5, 2)
            
            # Convert to image coordinates
            stride = self.image_size / face_cls.shape[1]
            cx = (x + 0.5) * stride / scale
            cy = (y + 0.5) * stride / scale
            w = bbox_pred[2] * self.image_size / scale
            h = bbox_pred[3] * self.image_size / scale
            
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            # Clip bbox to image bounds
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            
            # Convert landmarks to image coordinates
            landmarks = landmarks_pred * self.image_size / scale
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0, orig_shape[1])
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0, orig_shape[0])
            
            detections.append(FaceDetection(
                bbox=np.array([x1, y1, x2, y2]),
                landmarks=landmarks.astype(int),
                confidence=conf
            ))
        
        return detections
    
    def detect_hands(
        self,
        outputs: Dict[str, torch.Tensor],
        scale: float,
        orig_shape: Tuple[int, int]
    ) -> List[HandDetection]:
        """
        Extract hand detections from model outputs
        
        Args:
            outputs: Model output dictionary
            scale: Scale factor for coordinate conversion
            orig_shape: Original image shape (H, W)
            
        Returns:
            List of HandDetection objects
        """
        hand_cls = outputs['hand_cls'][0]  # [2, H, W]
        hand_heatmaps = outputs['hand_heatmaps'][0]  # [21, H, W]
        hand_offsets = outputs['hand_offsets'][0]  # [42, H, W]
        
        # Get confidence scores
        hand_conf = torch.sigmoid(hand_cls[1])  # [H, W]
        
        # Find hand centers (local maxima in heatmaps)
        # For simplicity, use threshold for now
        mask = hand_conf > self.confidence_threshold
        indices = torch.nonzero(mask)  # [N, 2] (y, x)
        
        detections = []
        for idx in indices:
            y, x = idx[0].item(), idx[1].item()
            conf = hand_conf[y, x].item()
            
            # Get keypoint heatmaps and offsets
            heatmaps = hand_heatmaps[:, y, x].cpu().numpy()  # [21]
            offsets = hand_offsets[:, y, x].cpu().numpy().reshape(21, 2)  # [21, 2]
            
            # Convert to image coordinates
            stride = self.image_size / hand_cls.shape[1]
            keypoints = np.zeros((21, 2))
            
            for i in range(21):
                kp_x = (x + offsets[i, 0]) * stride / scale
                kp_y = (y + offsets[i, 1]) * stride / scale
                keypoints[i] = [kp_x, kp_y]
            
            # Clip keypoints to image bounds
            keypoints[:, 0] = np.clip(keypoints[:, 0], 0, orig_shape[1])
            keypoints[:, 1] = np.clip(keypoints[:, 1], 0, orig_shape[0])
            
            # Compute bounding box from keypoints
            x1 = int(np.min(keypoints[:, 0]))
            y1 = int(np.min(keypoints[:, 1]))
            x2 = int(np.max(keypoints[:, 0]))
            y2 = int(np.max(keypoints[:, 1]))
            
            # Visibility scores
            visibility = torch.sigmoid(hand_heatmaps[:, y, x]).cpu().numpy()
            
            detections.append(HandDetection(
                bbox=np.array([x1, y1, x2, y2]),
                keypoints=keypoints.astype(int),
                confidence=conf,
                visibility=visibility
            ))
        
        return detections
    
    @torch.no_grad()
    def detect(
        self, 
        image: np.ndarray
    ) -> Tuple[List[PersonDetection], List[FaceDetection], List[HandDetection], float]:
        """
        Unified detection on single image
        
        Args:
            image: BGR image [H, W, 3]
            
        Returns:
            - List of PersonDetection
            - List of FaceDetection
            - List of HandDetection
            - Inference time in milliseconds
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
        
        inference_time = (time.time() - start) * 1000  # ms
        
        return people, faces, hands, inference_time


if __name__ == '__main__':
    print("Testing Unified Detector...")
    
    # Create detector
    detector = UnifiedDetector(
        backbone_name='mobilenetv3_large_100',
        device='cpu',  # Use 'cuda' if available
        confidence_threshold=0.5
    )
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run detection
    people, faces, hands, inference_time = detector.detect(test_image)
    
    print(f"\nResults:")
    print(f"  People detected: {len(people)}")
    print(f"  Faces detected: {len(faces)}")
    print(f"  Hands detected: {len(hands)}")
    print(f"  Inference time: {inference_time:.2f} ms")
    
    # Benchmark
    print("\nBenchmarking...")
    times = []
    for _ in range(50):
        _, _, _, t = detector.detect(test_image)
        times.append(t)
    
    avg_time = np.mean(times)
    fps = 1000 / avg_time
    
    print(f"Average time: {avg_time:.2f} ms")
    print(f"FPS: {fps:.1f}")
