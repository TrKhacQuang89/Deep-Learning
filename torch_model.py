"""
PyTorch Object Detector Models

Same architecture as numpy SimpleDetector, but implemented in PyTorch.
Creates 4 model sizes for experimentation:
- TinyDetector: ~395K params (same as numpy version)
- SmallDetector: ~1.5M params
- MediumDetector: ~3.5M params
- LargeDetector: ~6.2M params

Input: (N, 3, 224, 224)
Output: 
    - Detection mode: (N, 8, 7, 7) = Grid 7x7, each cell predicts [x, y, w, h, conf, c1, c2, c3]
    - Classification mode: (N, num_classes) = Class probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectorBase(nn.Module):
    """
    Base class for grid-based object detector with ResNet-style architecture.
    
    Features:
        - ResNet-style residual blocks with SE attention
        - Multi-scale feature fusion (FPN-style lateral connections)
        - SiLU activation throughout
        - Configurable stage depths and channels
    
    Architecture:
        Stem: Conv 7x7/2 + BN + SiLU + MaxPool → 56×56
        Stage1: N× ResBlock (c1) → 56×56  ─┐
        Stage2: N× ResBlock (c2) → 28×28  ─┼─→ Lateral connections
        Stage3: N× ResBlock (c3) → 14×14  ─┤   (all fused at 7×7)
        Stage4: N× ResBlock (c4) → 7×7    ─┘
        Fusion: Concat all lateral outputs → Head
        Head: Conv3x3 → Conv3x3 → Conv1x1 → 7×7×8 (output)
    
    Input: (N, 3, 224, 224)
    Output: (N, 8, 7, 7)
    """
    
    def __init__(
        self, 
        stage_channels: list,      # [c1, c2, c3, c4] - channels for each stage
        stage_blocks: list,        # [n1, n2, n3, n4] - number of blocks per stage
        lateral_channels: int,     # Common channel size for lateral connections
        num_classes: int = 3,
        dropout: float = 0.1,
        use_se: bool = True,        # Whether to use SE attention
        classifier: bool = False
    ):
        """
        Args:
            stage_channels: List of 4 channel sizes [c1, c2, c3, c4]
            stage_blocks: List of 4 block counts [n1, n2, n3, n4]
            lateral_channels: Channel size for lateral projections
            num_classes: Number of object classes
            dropout: Dropout rate
            use_se: Whether to use SE attention in residual blocks
            classifier: Whether to use classification head instead of detection head
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.output_channels = 5 + num_classes
        self.use_se = use_se
        self.classifier = classifier
        
        c1, c2, c3, c4 = stage_channels
        n1, n2, n3, n4 = stage_blocks
        
        # Stem: 224 → 56 (stride 4 total: 2 from conv + 2 from pool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual stages
        self.stage1 = self._make_stage(c1, c1, n1, stride=1)   # 56×56
        self.stage2 = self._make_stage(c1, c2, n2, stride=2)   # 28×28
        self.stage3 = self._make_stage(c2, c3, n3, stride=2)   # 14×14
        self.stage4 = self._make_stage(c3, c4, n4, stride=2)   # 7×7
        
        # Lateral connections: project each stage to common channels
        self.lateral1 = nn.Sequential(
            nn.Conv2d(c1, lateral_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=8)  # 56→7
        )
        self.lateral2 = nn.Sequential(
            nn.Conv2d(c2, lateral_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)  # 28→7
        )
        self.lateral3 = nn.Sequential(
            nn.Conv2d(c3, lateral_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14→7
        )
        self.lateral4 = nn.Sequential(
            nn.Conv2d(c4, lateral_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_channels),
            nn.SiLU(inplace=True)  # 7→7
        )
        
        # Fused channels from 4 laterals
        fused_channels = lateral_channels * 4
        
        self.dropout = nn.Dropout2d(p=dropout)
        
        if self.classifier:
            # Classification head
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(fused_channels, num_classes)
            )
        else:
            # Detection head
            self.head = nn.Sequential(
                nn.Conv2d(fused_channels, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.SiLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
                nn.Conv2d(128, self.output_channels, kernel_size=1)
            )
        
        self._init_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        """Create a stage with multiple residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride=stride, use_se=self.use_se)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, use_se=self.use_se))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        """
        Forward pass with multi-scale residual connections.
        
        Args:
            x: Input tensor (N, 3, 224, 224)
            
        Returns:
            output: (N, 8, 7, 7) with format:
                    - channels 0-3: x, y, w, h (sigmoid applied)
                    - channel 4: confidence (sigmoid applied)
                    - channels 5-7: class probabilities (softmax applied)
        """
        # Stem
        x = self.stem(x)  # 56×56
        
        # Residual stages
        s1 = self.stage1(x)   # 56×56
        s2 = self.stage2(s1)  # 28×28
        s3 = self.stage3(s2)  # 14×14
        s4 = self.stage4(s3)  # 7×7
        
        # Lateral connections
        l1 = self.lateral1(s1)  # 7×7
        l2 = self.lateral2(s2)  # 7×7
        l3 = self.lateral3(s3)  # 7×7
        l4 = self.lateral4(s4)  # 7×7
        
        # Fuse all scales
        fused = torch.cat([l1, l2, l3, l4], dim=1)
        
        # Dropout
        fused = self.dropout(fused)
        
        if self.classifier:
            # Return raw logits (CrossEntropyLoss applies softmax internally)
            logits = self.head(fused)
            return logits
        else:
            # Detection head
            out = self.head(fused)
            
            # Apply activations
            box_xy = torch.sigmoid(out[:, 0:2, :, :])
            box_wh = torch.sigmoid(out[:, 2:4, :, :])
            conf = torch.sigmoid(out[:, 4:5, :, :])
            cls_probs = F.softmax(out[:, 5:, :, :], dim=1)
            
            output = torch.cat([box_xy, box_wh, conf, cls_probs], dim=1)
            return output
    
    def get_params_count(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class TinyDetector(DetectorBase):
    """
    Tiny detector - lightweight for quick experiments.
    Stages: [32, 64, 128, 256], Blocks: [1, 1, 1, 1]
    """
    def __init__(self, num_classes: int = 3, classifier: bool = False):
        super().__init__(
            stage_channels=[32, 64, 128, 256],
            stage_blocks=[1, 1, 1, 1],
            lateral_channels=32,
            num_classes=num_classes,
            use_se=False,  # No SE for tiny
            classifier=classifier
        )


class SmallDetector(DetectorBase):
    """
    Small detector.
    Stages: [32, 64, 128, 256], Blocks: [2, 2, 2, 2]
    """
    def __init__(self, num_classes: int = 3, classifier: bool = False):
        super().__init__(
            stage_channels=[32, 64, 128, 256],
            stage_blocks=[2, 2, 2, 2],
            lateral_channels=64,
            num_classes=num_classes,
            classifier=classifier
        )


class MediumDetector(DetectorBase):
    """
    Medium detector.
    Stages: [48, 96, 192, 384], Blocks: [2, 3, 3, 2]
    """
    def __init__(self, num_classes: int = 3, classifier: bool = False):
        super().__init__(
            stage_channels=[48, 96, 192, 384],
            stage_blocks=[2, 3, 3, 2],
            lateral_channels=96,
            num_classes=num_classes,
            classifier=classifier
        )


class LargeDetector(DetectorBase):
    """
    Large detector.
    Stages: [64, 128, 256, 512], Blocks: [3, 4, 6, 3] (like ResNet-50)
    """
    def __init__(self, num_classes: int = 3, classifier: bool = False):
        super().__init__(
            stage_channels=[64, 128, 256, 512],
            stage_blocks=[3, 4, 6, 3],
            lateral_channels=128,
            num_classes=num_classes,
            classifier=classifier
        )


class XLargeDetector(DetectorBase):
    """
    XLarge detector (~28M params).
    Stages: [64, 128, 256, 512], Blocks: [3, 4, 23, 3] (like ResNet-101)
    """
    def __init__(self, num_classes: int = 3, classifier: bool = False):
        super().__init__(
            stage_channels=[64, 128, 256, 512],
            stage_blocks=[3, 4, 23, 3],
            lateral_channels=128,
            num_classes=num_classes,
            classifier=classifier
        )


class HugeDetector(DetectorBase):
    """
    Huge detector (~50M params).
    Stages: [128, 256, 512, 1024], Blocks: [3, 4, 6, 3]
    """
    def __init__(self, num_classes: int = 3, classifier: bool = False):
        super().__init__(
            stage_channels=[128, 256, 512, 1024],
            stage_blocks=[3, 4, 6, 3],
            lateral_channels=256,
            num_classes=num_classes,
            classifier=classifier
        )


# =============================================================================
# Advanced Architecture Components
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with optional SE attention and downsampling."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.silu(out + identity)
        return out


class GiantDetector(nn.Module):
    """
    Giant detector with multi-scale residual architecture (~92M params).
    
    Features:
        - ResNet-style residual blocks with skip connections
        - Squeeze-and-Excitation (SE) attention mechanism
        - Multi-scale feature fusion (Residual-in-Residual)
        - Each stage has lateral connection to detection head
        - Dropout regularization
    
    Architecture:
        Stem: Conv 7x7 + BN + ReLU + MaxPool → 56×56    
        Stage1: 3x ResBlock (64ch) → 56×56  ─┐
        Stage2: 4x ResBlock (128ch) → 28×28 ─┼─→ Lateral connections
        Stage3: 6x ResBlock (256ch) → 14×14 ─┤   (all fused at 7×7)
        Stage4: 3x ResBlock (512ch) → 7×7   ─┘
        Fusion: Concat all lateral outputs → Head
        Head: Conv 1x1 → 7×7×8 (output)
    
    Input: (N, 3, 224, 224)
    Output: (N, 8, 7, 7)
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.output_channels = 5 + num_classes
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 56x56
        )
        
        # Residual stages (larger channels for ~50M params)
        self.stage1 = self._make_stage(64, 128, num_blocks=3, stride=1)    # 56x56
        self.stage2 = self._make_stage(128, 256, num_blocks=4, stride=2)   # 28x28
        self.stage3 = self._make_stage(256, 512, num_blocks=6, stride=2)   # 14x14
        self.stage4 = self._make_stage(512, 1024, num_blocks=3, stride=2)  # 7x7
        
        # Lateral connections: project each stage output to common channel size
        lateral_ch = 256  # Larger for more capacity

        # Stage1: 56x56 → 7x7 (8x downsample)
        self.lateral1 = nn.Sequential(
            nn.Conv2d(128, lateral_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=8)  # 56→7
        )
        
        # Stage2: 28x28 → 7x7 (4x downsample)
        self.lateral2 = nn.Sequential(
            nn.Conv2d(256, lateral_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)  # 28→7
        )
        
        # Stage3: 14x14 → 7x7 (2x downsample)
        self.lateral3 = nn.Sequential(
            nn.Conv2d(512, lateral_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14→7
        )
        
        # Stage4: 7x7 → 7x7 (no downsample)
        self.lateral4 = nn.Sequential(
            nn.Conv2d(1024, lateral_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_ch),
            nn.SiLU(inplace=True)
        )
        
        # Total fused channels: lateral_ch * 4 = 1024
        fused_channels = lateral_ch * 4
        
        # Detection head (receives fused multi-scale features)
        self.dropout = nn.Dropout2d(p=dropout)
        self.head = nn.Sequential(
            nn.Conv2d(fused_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, self.output_channels, kernel_size=1)
        )
        
        self._init_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        """Create a stage with multiple residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        """
        Forward pass with multi-scale residual connections.
        
        Args:
            x: Input tensor (N, 3, 224, 224)
            
        Returns:
            output: (N, 8, 7, 7) with same format as other detectors
        """
        # Feature extraction with multi-scale outputs
        x = self.stem(x)        # 56x56, 64ch
        
        s1 = self.stage1(x)     # 56x56, 64ch
        s2 = self.stage2(s1)    # 28x28, 128ch
        s3 = self.stage3(s2)    # 14x14, 256ch
        s4 = self.stage4(s3)    # 7x7, 512ch
        
        # Lateral connections: all to 7x7
        l1 = self.lateral1(s1)  # 7x7, 128ch
        l2 = self.lateral2(s2)  # 7x7, 128ch
        l3 = self.lateral3(s3)  # 7x7, 128ch
        l4 = self.lateral4(s4)  # 7x7, 128ch
        
        # Fuse all scales (concatenate)
        fused = torch.cat([l1, l2, l3, l4], dim=1)  # 7x7, 512ch
        
        # Detection head
        fused = self.dropout(fused)
        out = self.head(fused)  # (N, 8, 7, 7)
        
        # Apply activations (same as DetectorBase)
        box_xy = torch.sigmoid(out[:, 0:2, :, :])
        box_wh = torch.sigmoid(out[:, 2:4, :, :])
        conf = torch.sigmoid(out[:, 4:5, :, :])
        cls_probs = F.softmax(out[:, 5:, :, :], dim=1)
        
        output = torch.cat([box_xy, box_wh, conf, cls_probs], dim=1)
        return output
    
    def get_params_count(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

class GiantDetectorOrginal(nn.Module):
    """
    Giant detector with multi-scale residual architecture (~23M params).
    
    Features:
        - ResNet-style residual blocks with skip connections
        - Squeeze-and-Excitation (SE) attention mechanism
        - Multi-scale feature fusion (Residual-in-Residual)
        - Each stage has lateral connection to detection head
        - Dropout regularization
    
    Architecture:
        Stem: Conv 7x7 + BN + ReLU + MaxPool → 56×56
        Stage1: 3x ResBlock (64ch) → 56×56  ─┐
        Stage2: 4x ResBlock (128ch) → 28×28 ─┼─→ Lateral connections
        Stage3: 6x ResBlock (256ch) → 14×14 ─┤   (all fused at 7×7)
        Stage4: 3x ResBlock (512ch) → 7×7   ─┘
        Fusion: Concat all lateral outputs → Head
        Head: Conv 1x1 → 7×7×8 (output)
    
    Input: (N, 3, 224, 224)
    Output: (N, 8, 7, 7)
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.output_channels = 5 + num_classes
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 56x56
        )
        
        # Residual stages
        self.stage1 = self._make_stage(64, 64, num_blocks=3, stride=1)    # 56x56
        self.stage2 = self._make_stage(64, 128, num_blocks=4, stride=2)   # 28x28
        self.stage3 = self._make_stage(128, 256, num_blocks=6, stride=2)  # 14x14
        self.stage4 = self._make_stage(256, 512, num_blocks=3, stride=2)  # 7x7
        
        # Lateral connections: project each stage output to common channel size
        lateral_ch = 128  # Common channel size for fusion
        
        # Stage1: 56x56 → 7x7 (8x downsample)
        self.lateral1 = nn.Sequential(
            nn.Conv2d(64, lateral_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=8)  # 56→7
        )
        
        # Stage2: 28x28 → 7x7 (4x downsample)
        self.lateral2 = nn.Sequential(
            nn.Conv2d(128, lateral_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)  # 28→7
        )
        
        # Stage3: 14x14 → 7x7 (2x downsample)
        self.lateral3 = nn.Sequential(
            nn.Conv2d(256, lateral_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14→7
        )
        
        # Stage4: 7x7 → 7x7 (no downsample)
        self.lateral4 = nn.Sequential(
            nn.Conv2d(512, lateral_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(lateral_ch),
            nn.SiLU(inplace=True)
        )
        
        # Total fused channels: lateral_ch * 4 = 512
        fused_channels = lateral_ch * 4
        
        # Detection head (receives fused multi-scale features)
        self.dropout = nn.Dropout2d(p=dropout)
        self.head = nn.Sequential(
            nn.Conv2d(fused_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, self.output_channels, kernel_size=1)
        )
        
        self._init_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        """Create a stage with multiple residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        """
        Forward pass with multi-scale residual connections.
        
        Args:
            x: Input tensor (N, 3, 224, 224)
            
        Returns:
            output: (N, 8, 7, 7) with same format as other detectors
        """
        # Feature extraction with multi-scale outputs
        x = self.stem(x)        # 56x56, 64ch
        
        s1 = self.stage1(x)     # 56x56, 64ch
        s2 = self.stage2(s1)    # 28x28, 128ch
        s3 = self.stage3(s2)    # 14x14, 256ch
        s4 = self.stage4(s3)    # 7x7, 512ch
        
        # Lateral connections: all to 7x7
        l1 = self.lateral1(s1)  # 7x7, 128ch
        l2 = self.lateral2(s2)  # 7x7, 128ch
        l3 = self.lateral3(s3)  # 7x7, 128ch
        l4 = self.lateral4(s4)  # 7x7, 128ch
        
        # Fuse all scales (concatenate)
        fused = torch.cat([l1, l2, l3, l4], dim=1)  # 7x7, 512ch
        
        # Detection head
        fused = self.dropout(fused)
        out = self.head(fused)  # (N, 8, 7, 7)
        
        # Apply activations (same as DetectorBase)
        box_xy = torch.sigmoid(out[:, 0:2, :, :])
        box_wh = torch.sigmoid(out[:, 2:4, :, :])
        conf = torch.sigmoid(out[:, 4:5, :, :])
        cls_probs = F.softmax(out[:, 5:, :, :], dim=1)
        
        output = torch.cat([box_xy, box_wh, conf, cls_probs], dim=1)
        return output
    
    def get_params_count(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def get_model(size: str = 'tiny', num_classes: int = 3, classifier: bool = False) -> nn.Module:
    """
    Factory function to create a detector or classifier model.
    
    Args:
        size: Model size - 'tiny', 'small', 'medium', 'large', 'xlarge', 'huge', 'giant', or 'giant_original'
        num_classes: Number of object or classification classes
        classifier: Whether to use classification head
        
    Returns:
        Model instance
    """
    models = {
        'tiny': TinyDetector,
        'small': SmallDetector,
        'medium': MediumDetector,
        'large': LargeDetector,
        'xlarge': XLargeDetector,
        'huge': HugeDetector,
        'giant': GiantDetector,
        'giant_original': GiantDetectorOrginal,
    }
    
    if size not in models:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(models.keys())}")
    
    # Check if the class supports the classifier argument
    import inspect
    model_class = models[size]
    sig = inspect.signature(model_class.__init__)
    if 'classifier' in sig.parameters:
        return model_class(num_classes=num_classes, classifier=classifier)
    else:
        # GiantDetector and its variants might not support it yet unless updated
        if classifier:
             print(f"[WARNING] Model size '{size}' does not support classifier mode. Defaulting to detection.")
        return model_class(num_classes=num_classes)


# Convenience aliases for classifiers
TinyClassifier = lambda num_classes=3: get_model('tiny', num_classes, classifier=True)
SmallClassifier = lambda num_classes=3: get_model('small', num_classes, classifier=True)
MediumClassifier = lambda num_classes=3: get_model('medium', num_classes, classifier=True)
LargeClassifier = lambda num_classes=3: get_model('large', num_classes, classifier=True)
XLargeClassifier = lambda num_classes=3: get_model('xlarge', num_classes, classifier=True)
HugeClassifier = lambda num_classes=3: get_model('huge', num_classes, classifier=True)

SimpleClassifier = LargeClassifier


if __name__ == "__main__":
    print("Testing PyTorch Model (with Classifier Support)")
    x = torch.randn(2, 3, 224, 224)
    
    # Test detector
    print("\n--- Detector Mode ---")
    model = get_model('tiny', classifier=False)
    print(f"Parameters: {model.get_params_count():,}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test classifier
    print("\n--- Classifier Mode ---")
    classifier = get_model('tiny', num_classes=10, classifier=True)
    print(f"Parameters: {classifier.get_params_count():,}")
    output = classifier(x)
    print(f"Output shape: {output.shape}")
    print(f"Sum of probs per sample: {output.sum(dim=1)}")