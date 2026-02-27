"""
Shared backbone network for multi-task learning
Supports multiple efficient architectures: EfficientViT, MobileNetV3, EfficientNet
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import timm


class SharedBackbone(nn.Module):
    """
    Shared feature extractor for multi-task detection
    
    Extracts features that will be used by PersonHead, FaceHead, and HandHead
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
            backbone_name: Name of backbone architecture from timm
                - 'mobilenetv3_large_100': Fast, 5.4M params
                - 'efficientnet_b1': Good balance, 7.8M params
                - 'efficientvit_b1': Best efficiency (if available)
            pretrained: Use ImageNet pretrained weights
            feature_channels: Output feature dimension
            freeze_backbone: Freeze backbone weights during training
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.feature_channels = feature_channels
        
        # Create backbone using timm
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,  # Return intermediate features
                out_indices=[2, 3, 4]  # Return features at 3 scales
            )
        except Exception as e:
            print(f"Error creating {backbone_name}: {e}")
            print("Falling back to mobilenetv3_large_100...")
            self.backbone = timm.create_model(
                'mobilenetv3_large_100',
                pretrained=pretrained,
                features_only=True,
                out_indices=[2, 3, 4]
            )
        
        # Get feature info to determine channels
        feature_info = self.backbone.feature_info
        self.backbone_channels = [info['num_chs'] for info in feature_info]
        
        print(f"Backbone: {backbone_name}")
        print(f"Feature channels: {self.backbone_channels}")
        
        # Feature Pyramid Network (FPN) for multi-scale features
        self.fpn = FeaturePyramidNetwork(
            in_channels=self.backbone_channels,
            out_channels=feature_channels
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary of features at multiple scales:
            - 'p3': 1/8 resolution features [B, 256, H/8, W/8]
            - 'p4': 1/16 resolution features [B, 256, H/16, W/16]
            - 'p5': 1/32 resolution features [B, 256, H/32, W/32]
        """
        # Extract multi-scale features from backbone
        features = self.backbone(x)  # List of [c3, c4, c5]
        
        # Apply FPN to get uniform channel dimension
        fpn_features = self.fpn(features)
        
        return fpn_features


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network to combine multi-scale features
    """
    
    def __init__(self, in_channels: list, out_channels: int = 256):
        """
        Args:
            in_channels: List of input channels at each scale [C3, C4, C5]
            out_channels: Uniform output channels
        """
        super().__init__()
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral3 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral5 = nn.Conv2d(in_channels[2], out_channels, 1)
        
        # Output convolutions (3x3 conv to reduce aliasing)
        self.output3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: list) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of feature maps [c3, c4, c5]
            
        Returns:
            Dictionary of FPN features {'p3', 'p4', 'p5'}
        """
        c3, c4, c5 = features
        
        # Top-down pathway with lateral connections
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + nn.functional.interpolate(
            p5, size=c4.shape[2:], mode='nearest'
        )
        p3 = self.lateral3(c3) + nn.functional.interpolate(
            p4, size=c3.shape[2:], mode='nearest'
        )
        
        # Apply output convolutions
        p5 = self.output5(p5)
        p4 = self.output4(p4)
        p3 = self.output3(p3)
        
        return {
            'p3': p3,  # 1/8 resolution
            'p4': p4,  # 1/16 resolution
            'p5': p5   # 1/32 resolution
        }


def create_backbone(
    backbone_name: str = 'mobilenetv3_large_100',
    pretrained: bool = True,
    feature_channels: int = 256
) -> SharedBackbone:
    """
    Factory function to create backbone
    
    Args:
        backbone_name: Architecture name
        pretrained: Use pretrained weights
        feature_channels: Output feature dimension
        
    Returns:
        SharedBackbone instance
    """
    return SharedBackbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        feature_channels=feature_channels
    )


if __name__ == '__main__':
    # Test backbone
    print("Testing SharedBackbone...")
    
    # Create backbone
    backbone = create_backbone(
        backbone_name='mobilenetv3_large_100',
        pretrained=False,  # Set True to download weights
        feature_channels=256
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    features = backbone(x)
    
    print("\nOutput features:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
