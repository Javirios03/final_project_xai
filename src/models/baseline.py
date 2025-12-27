"""
Baseline ResNet18 model for chest X-ray classification
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class BaselineModel(nn.Module):
    """
    ResNet18 pretrained on ImageNet, fine-tuned for binary classification
    
    Args:
        num_classes: Number of output classes (default: 2)
        pretrained: Use ImageNet pretrained weights (default: True)
        freeze_backbone: Freeze backbone layers (default: False)
        dropout: Dropout rate before final FC (default: 0.3)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super(BaselineModel, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Backbone frozen (only training final layer)")
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
        print(f"✅ BaselineModel created:")
        print(f"   Architecture: ResNet18")
        print(f"   Pretrained: {pretrained}")
        print(f"   Output classes: {num_classes}")
        print(f"   Dropout: {dropout}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before final FC layer
        Useful for XAI analysis
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def count_parameters(self) -> tuple:
        """Count trainable and total parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def get_model(
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = 'cuda'
) -> BaselineModel:
    """
    Factory function to create and initialize model
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        device: Device to move model to
    
    Returns:
        Initialized model on specified device
    """
    model = BaselineModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=False,
        dropout=0.3
    )
    
    model = model.to(device)
    
    # Print parameter counts
    trainable, total = model.count_parameters()
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Total parameters: {total:,}")
    
    return model


if __name__ == '__main__':
    """Test the model"""
    print("\n" + "="*70)
    print("TESTING BASELINE MODEL")
    print("="*70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Create model
    model = get_model(num_classes=2, pretrained=True, device=device)
    
    # Test forward pass
    print(f"\n🔍 Testing forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"   Features shape: {features.shape}")
    
    print("\n✅ Model test complete!")
