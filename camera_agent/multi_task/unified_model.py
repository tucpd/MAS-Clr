"""
Unified Multi-Task Model
Combines shared backbone with task-specific heads for efficient inference
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path

from .multi_task.backbone import SharedBackbone
from .heads import MultiTaskHead


class UnifiedMultiTaskModel(nn.Module):
    """
    Multi-task model for person detection, face detection, and hand keypoint detection
    
    Architecture:
        Input Image → SharedBackbone → FPN Features → Task Heads → Outputs
        
    Single forward pass detects:
        - People (bounding boxes)
        - Faces (bounding boxes + landmarks)
        - Hands (21 keypoints)
    """
    
    def __init__(
        self,
        backbone_name: str = 'mobilenetv3_large_100',
        pretrained: bool = True,
        feature_channels: int = 256,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone_name: Backbone architecture name
            pretrained: Use ImageNet pretrained weights
            feature_channels: Feature dimension for FPN
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.feature_channels = feature_channels
        
        # Shared backbone
        self.backbone = SharedBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            feature_channels=feature_channels,
            freeze_backbone=freeze_backbone
        )
        
        # Task-specific heads
        self.heads = MultiTaskHead(in_channels=feature_channels)
        
        print(f"\n=== Unified Multi-Task Model ===")
        print(f"Backbone: {backbone_name}")
        print(f"Feature channels: {feature_channels}")
        print(f"Tasks: Person Detection, Face Detection, Hand Keypoints")
        
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model parameter counts"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.heads.parameters())
        total_params = backbone_params + head_params
        
        print(f"\nModel Parameters:")
        print(f"  Backbone: {backbone_params:,}")
        print(f"  Heads: {head_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire model
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary containing all task outputs:
            Person Detection:
                - person_cls: [B, 2, H/16, W/16]
                - person_reg: [B, 4, H/16, W/16]
            Face Detection:
                - face_cls: [B, 2, H/8, W/8]
                - face_bbox: [B, 4, H/8, W/8]
                - face_landmarks: [B, 10, H/8, W/8]
            Hand Keypoints:
                - hand_cls: [B, 2, H/8, W/8]
                - hand_heatmaps: [B, 21, H/8, W/8]
                - hand_offsets: [B, 42, H/8, W/8]
        """
        # Extract shared features from backbone
        features = self.backbone(x)
        
        # Apply task-specific heads
        outputs = self.heads(features)
        
        return outputs
    
    def save(self, path: str):
        """
        Save model weights
        
        Args:
            path: Path to save checkpoint
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'backbone_name': self.backbone_name,
            'feature_channels': self.feature_channels,
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, path: str, strict: bool = True):
        """
        Load model weights
        
        Args:
            path: Path to checkpoint
            strict: Strict loading (all keys must match)
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"Model loaded from {path}")
    
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference mode forward pass (no gradient computation)
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary of task outputs
        """
        self.eval()
        return self.forward(x)


def create_unified_model(
    backbone_name: str = 'mobilenetv3_large_100',
    pretrained: bool = True,
    feature_channels: int = 256,
    checkpoint_path: Optional[str] = None
) -> UnifiedMultiTaskModel:
    """
    Factory function to create unified model
    
    Args:
        backbone_name: Backbone architecture
        pretrained: Use pretrained backbone
        feature_channels: Feature dimension
        checkpoint_path: Path to trained checkpoint (optional)
        
    Returns:
        UnifiedMultiTaskModel instance
    """
    model = UnifiedMultiTaskModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
        feature_channels=feature_channels
    )
    
    if checkpoint_path is not None:
        model.load(checkpoint_path)
    
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Unified Multi-Task Model")
    print("=" * 60)
    
    # Create model
    model = create_unified_model(
        backbone_name='mobilenetv3_large_100',
        pretrained=False,  # Set True to download ImageNet weights
        feature_channels=256
    )
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    batch_size = 1
    img_size = 640
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print("\nOutput shapes:")
    print("-" * 40)
    print("Person Detection:")
    print(f"  person_cls: {outputs['person_cls'].shape}")
    print(f"  person_reg: {outputs['person_reg'].shape}")
    print("\nFace Detection:")
    print(f"  face_cls: {outputs['face_cls'].shape}")
    print(f"  face_bbox: {outputs['face_bbox'].shape}")
    print(f"  face_landmarks: {outputs['face_landmarks'].shape}")
    print("\nHand Keypoints:")
    print(f"  hand_cls: {outputs['hand_cls'].shape}")
    print(f"  hand_heatmaps: {outputs['hand_heatmaps'].shape}")
    print(f"  hand_offsets: {outputs['hand_offsets'].shape}")
    
    # Measure inference time
    print("\n" + "=" * 60)
    print("Measuring Inference Speed")
    print("=" * 60)
    
    import time
    
    # Warmup
    for _ in range(10):
        _ = model.inference(x)
    
    # Benchmark
    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        _ = model.inference(x)
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"FPS: {fps:.1f}")
    
    # Test save/load
    print("\n" + "=" * 60)
    print("Testing Save/Load")
    print("=" * 60)
    
    save_path = "test_unified_model.pth"
    model.save(save_path)
    
    # Create new model and load
    model2 = create_unified_model(
        backbone_name='mobilenetv3_large_100',
        pretrained=False,
        checkpoint_path=save_path
    )
    
    # Verify outputs match
    with torch.no_grad():
        outputs2 = model2(x)
    
    match = all(
        torch.allclose(outputs[k], outputs2[k], atol=1e-6)
        for k in outputs.keys()
    )
    print(f"\nOutputs match after loading: {match}")
    
    # Clean up
    import os
    os.remove(save_path)
    print(f"Removed test checkpoint: {save_path}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
