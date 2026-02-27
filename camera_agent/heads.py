"""
Task-specific detection heads for multi-task model
PersonHead: Detect person bounding boxes
FaceHead: Detect face bounding boxes + landmarks
HandHead: Detect hand keypoints (21 landmarks)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class PersonHead(nn.Module):
    """
    Detection head for person detection
    Outputs: bounding boxes + confidence scores
    """
    
    def __init__(self, in_channels: int = 256, num_classes: int = 1):
        """
        Args:
            in_channels: Input feature channels from backbone
            num_classes: Number of classes (1 for person only)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Detection layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Classification head (objectness + class)
        self.cls_head = nn.Conv2d(256, num_classes + 1, 1)  # +1 for objectness
        
        # Regression head (x, y, w, h)
        self.reg_head = nn.Conv2d(256, 4, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Multi-scale features from backbone {'p3', 'p4', 'p5'}
            
        Returns:
            Dictionary containing:
            - 'person_cls': Classification scores [B, num_classes+1, H, W]
            - 'person_reg': Bounding box regression [B, 4, H, W]
        """
        # Use P4 features (1/16 resolution) for person detection
        x = features['p4']
        
        # Apply detection layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Classification and regression
        cls_output = self.cls_head(x)
        reg_output = self.reg_head(x)
        
        return {
            'person_cls': cls_output,
            'person_reg': reg_output
        }


class FaceHead(nn.Module):
    """
    Detection head for face detection with landmarks
    Outputs: bounding boxes + 5 facial landmarks (eyes, nose, mouth corners)
    """
    
    def __init__(self, in_channels: int = 256, num_landmarks: int = 5):
        """
        Args:
            in_channels: Input feature channels from backbone
            num_landmarks: Number of facial landmarks (default 5: 2 eyes, nose, 2 mouth)
        """
        super().__init__()
        
        self.num_landmarks = num_landmarks
        
        # Detection layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Classification head (face/no-face)
        self.cls_head = nn.Conv2d(256, 2, 1)
        
        # Bounding box regression head
        self.bbox_head = nn.Conv2d(256, 4, 1)
        
        # Landmark regression head (x, y for each landmark)
        self.landmark_head = nn.Conv2d(256, num_landmarks * 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Multi-scale features from backbone {'p3', 'p4', 'p5'}
            
        Returns:
            Dictionary containing:
            - 'face_cls': Classification scores [B, 2, H, W]
            - 'face_bbox': Bounding box regression [B, 4, H, W]
            - 'face_landmarks': Landmark coordinates [B, num_landmarks*2, H, W]
        """
        # Use P3 features (1/8 resolution) for face detection (smaller objects)
        x = features['p3']
        
        # Apply detection layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Classification, bbox, and landmarks
        cls_output = self.cls_head(x)
        bbox_output = self.bbox_head(x)
        landmark_output = self.landmark_head(x)
        
        return {
            'face_cls': cls_output,
            'face_bbox': bbox_output,
            'face_landmarks': landmark_output
        }


class HandHead(nn.Module):
    """
    Detection head for hand keypoint detection
    Outputs: 21 hand keypoints (MediaPipe convention)
    """
    
    def __init__(self, in_channels: int = 256, num_keypoints: int = 21):
        """
        Args:
            in_channels: Input feature channels from backbone
            num_keypoints: Number of hand keypoints (21 for MediaPipe standard)
        """
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # Detection layers with more capacity for keypoint regression
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Hand presence classification
        self.cls_head = nn.Conv2d(256, 2, 1)  # hand/no-hand
        
        # Keypoint heatmap regression (one heatmap per keypoint)
        self.keypoint_head = nn.Conv2d(256, num_keypoints, 1)
        
        # Keypoint offset regression (x, y offset for each keypoint)
        self.offset_head = nn.Conv2d(256, num_keypoints * 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Multi-scale features from backbone {'p3', 'p4', 'p5'}
            
        Returns:
            Dictionary containing:
            - 'hand_cls': Hand presence classification [B, 2, H, W]
            - 'hand_heatmaps': Keypoint heatmaps [B, 21, H, W]
            - 'hand_offsets': Keypoint offsets [B, 42, H, W]
        """
        # Use P3 features (1/8 resolution) for hand keypoints
        x = features['p3']
        
        # Apply detection layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Classification, heatmaps, and offsets
        cls_output = self.cls_head(x)
        heatmap_output = self.keypoint_head(x)
        offset_output = self.offset_head(x)
        
        return {
            'hand_cls': cls_output,
            'hand_heatmaps': heatmap_output,
            'hand_offsets': offset_output
        }


class MultiTaskHead(nn.Module):
    """
    Combines all task-specific heads
    """
    
    def __init__(self, in_channels: int = 256):
        """
        Args:
            in_channels: Input feature channels from backbone
        """
        super().__init__()
        
        self.person_head = PersonHead(in_channels=in_channels)
        self.face_head = FaceHead(in_channels=in_channels)
        self.hand_head = HandHead(in_channels=in_channels)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Multi-scale features from backbone
            
        Returns:
            Dictionary containing all task outputs
        """
        # Get outputs from each head
        person_output = self.person_head(features)
        face_output = self.face_head(features)
        hand_output = self.hand_head(features)
        
        # Combine all outputs
        outputs = {}
        outputs.update(person_output)
        outputs.update(face_output)
        outputs.update(hand_output)
        
        return outputs


if __name__ == '__main__':
    # Test heads
    print("Testing Multi-Task Heads...")
    
    # Create dummy features
    batch_size = 2
    features = {
        'p3': torch.randn(batch_size, 256, 80, 80),  # 1/8 resolution
        'p4': torch.randn(batch_size, 256, 40, 40),  # 1/16 resolution
        'p5': torch.randn(batch_size, 256, 20, 20),  # 1/32 resolution
    }
    
    # Test individual heads
    print("\n1. PersonHead:")
    person_head = PersonHead(in_channels=256)
    person_out = person_head(features)
    for k, v in person_out.items():
        print(f"   {k}: {v.shape}")
    
    print("\n2. FaceHead:")
    face_head = FaceHead(in_channels=256)
    face_out = face_head(features)
    for k, v in face_out.items():
        print(f"   {k}: {v.shape}")
    
    print("\n3. HandHead:")
    hand_head = HandHead(in_channels=256)
    hand_out = hand_head(features)
    for k, v in hand_out.items():
        print(f"   {k}: {v.shape}")
    
    print("\n4. MultiTaskHead (combined):")
    multi_head = MultiTaskHead(in_channels=256)
    all_outputs = multi_head(features)
    for k, v in all_outputs.items():
        print(f"   {k}: {v.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in multi_head.parameters())
    print(f"\nTotal head parameters: {total_params:,}")
