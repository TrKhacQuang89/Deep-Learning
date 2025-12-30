"""
CuPy Object Detector Model - GPU Accelerated

Configurable ResNet-style architecture matching PyTorch DetectorBase:
- ResNet-style residual blocks with optional SE attention
- Multi-scale lateral connections (FPN-style)
- SiLU activations throughout
- Multiple model sizes: tiny, small, medium, large, xlarge, huge

Input: (N, 3, 224, 224)
Output: (N, 8, 7, 7) = Grid 7x7, each cell predicts [x, y, w, h, conf, c1, c2, c3]
"""

import cupy as cp
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cupy_core import Conv2d, BatchNorm2d, MaxPooling, SiLU, Sigmoid


# =============================================================================
# Model Configurations (matching PyTorch versions)
# =============================================================================

MODEL_CONFIGS = {
    'tiny': {
        'stage_channels': [32, 64, 128, 256],
        'stage_blocks': [1, 1, 1, 1],
        'lateral_channels': 32,
        'use_se': False,
    },
    'small': {
        'stage_channels': [32, 64, 128, 256],
        'stage_blocks': [2, 2, 2, 2],
        'lateral_channels': 64,
        'use_se': True,
    },
    'medium': {
        'stage_channels': [48, 96, 192, 384],
        'stage_blocks': [2, 3, 3, 2],
        'lateral_channels': 96,
        'use_se': True,
    },
    'large': {
        'stage_channels': [64, 128, 256, 512],
        'stage_blocks': [3, 4, 6, 3],
        'lateral_channels': 128,
        'use_se': True,
    },
    'xlarge': {
        'stage_channels': [64, 128, 256, 512],
        'stage_blocks': [3, 4, 23, 3],
        'lateral_channels': 128,
        'use_se': True,
    },
    'huge': {
        'stage_channels': [128, 256, 512, 1024],
        'stage_blocks': [3, 4, 6, 3],
        'lateral_channels': 256,
        'use_se': True,
    },
}


# =============================================================================
# Building Blocks
# =============================================================================

class SEBlock:
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        self.channels = channels
        reduced = max(1, channels // reduction)
        
        scale = cp.sqrt(2.0 / channels)
        self.fc1_W = cp.random.randn(channels, reduced).astype(cp.float32) * scale
        self.fc1_b = cp.zeros(reduced, dtype=cp.float32)
        
        scale = cp.sqrt(2.0 / reduced)
        self.fc2_W = cp.random.randn(reduced, channels).astype(cp.float32) * scale
        self.fc2_b = cp.zeros(channels, dtype=cp.float32)
        
        self.x = None
        self.squeeze = None
        self.excite1_sigmoid = None
        self.scale = None
    
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        
        self.squeeze = x.mean(axis=(2, 3))
        excite1 = self.squeeze @ self.fc1_W + self.fc1_b
        self.excite1_sigmoid = 1 / (1 + cp.exp(-excite1))
        excite1_act = excite1 * self.excite1_sigmoid
        
        excite2 = excite1_act @ self.fc2_W + self.fc2_b
        self.scale = 1 / (1 + cp.exp(-excite2))
        
        return x * self.scale.reshape(N, C, 1, 1)


class ResidualBlock:
    """Residual block with optional SE attention."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        self.stride = stride
        self.use_se = use_se
        
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else None
        
        self.use_shortcut = (stride != 1 or in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm2d(out_channels)
    
    def forward(self, x):
        if self.use_shortcut:
            identity = self.shortcut_bn.forward(self.shortcut_conv.forward(x))
        else:
            identity = x
        
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = out * (1 / (1 + cp.exp(-out)))  # SiLU
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        
        if self.se:
            out = self.se.forward(out)
        
        out = out + identity
        out = out * (1 / (1 + cp.exp(-out)))  # SiLU
        return out


# =============================================================================
# Main Detector
# =============================================================================

class DetectorBase:
    """
    Configurable Object Detector with ResNet-style architecture.
    
    Matches PyTorch DetectorBase exactly.
    """
    
    def __init__(
        self,
        stage_channels: list,
        stage_blocks: list,
        lateral_channels: int,
        num_classes: int = 3,
        dropout: float = 0.1,
        use_se: bool = True
    ):
        self.num_classes = num_classes
        self.output_channels = 5 + num_classes
        self.dropout_rate = dropout
        self.use_se = use_se
        
        c1, c2, c3, c4 = stage_channels
        n1, n2, n3, n4 = stage_blocks
        
        # Stem: 224 → 56
        self.stem_conv = Conv2d(3, c1, kernel_size=7, stride=2, padding=3)
        self.stem_bn = BatchNorm2d(c1)
        self.stem_pool = MaxPooling(kernel_size=3, stride=2)
        
        # Residual stages
        self.stage1 = self._make_stage(c1, c1, n1, stride=1)
        self.stage2 = self._make_stage(c1, c2, n2, stride=2)
        self.stage3 = self._make_stage(c2, c3, n3, stride=2)
        self.stage4 = self._make_stage(c3, c4, n4, stride=2)
        
        # Lateral connections
        self.lateral1 = self._make_lateral(c1, lateral_channels, pool_size=8)
        self.lateral2 = self._make_lateral(c2, lateral_channels, pool_size=4)
        self.lateral3 = self._make_lateral(c3, lateral_channels, pool_size=2)
        self.lateral4 = self._make_lateral(c4, lateral_channels, pool_size=None)
        
        # Detection head
        fused_channels = lateral_channels * 4
        self.head_conv1 = Conv2d(fused_channels, 256, kernel_size=3, padding=1)
        self.head_bn1 = BatchNorm2d(256)
        self.head_conv2 = Conv2d(256, 128, kernel_size=3, padding=1)
        self.head_bn2 = BatchNorm2d(128)
        self.head_conv3 = Conv2d(128, self.output_channels, kernel_size=1, padding=0)
        
        self.sigmoid = Sigmoid()
    
    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        blocks = [ResidualBlock(in_ch, out_ch, stride=stride, use_se=self.use_se)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_ch, out_ch, stride=1, use_se=self.use_se))
        return blocks
    
    def _make_lateral(self, in_ch, out_ch, pool_size):
        return {
            'conv': Conv2d(in_ch, out_ch, kernel_size=1, padding=0),
            'bn': BatchNorm2d(out_ch),
            'pool': MaxPooling(kernel_size=pool_size, stride=pool_size) if pool_size else None
        }
    
    def forward(self, x):
        # Stem
        x = self.stem_conv.forward(x)
        x = self.stem_bn.forward(x)
        x = x * (1 / (1 + cp.exp(-x)))  # SiLU
        x = cp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
        x = self.stem_pool.forward(x)
        
        # Stages
        s1 = x
        for block in self.stage1:
            s1 = block.forward(s1)
        
        s2 = s1
        for block in self.stage2:
            s2 = block.forward(s2)
        
        s3 = s2
        for block in self.stage3:
            s3 = block.forward(s3)
        
        s4 = s3
        for block in self.stage4:
            s4 = block.forward(s4)
        
        # Laterals
        l1 = self._forward_lateral(s1, self.lateral1)
        l2 = self._forward_lateral(s2, self.lateral2)
        l3 = self._forward_lateral(s3, self.lateral3)
        l4 = self._forward_lateral(s4, self.lateral4)
        
        # Fuse
        fused = cp.concatenate([l1, l2, l3, l4], axis=1)
        
        # Dropout
        if self.dropout_rate > 0:
            mask = (cp.random.rand(*fused.shape) > self.dropout_rate).astype(cp.float32)
            fused = fused * mask / (1 - self.dropout_rate)
        
        # Head
        out = self.head_conv1.forward(fused)
        out = self.head_bn1.forward(out)
        out = out * (1 / (1 + cp.exp(-out)))
        
        out = self.head_conv2.forward(out)
        out = self.head_bn2.forward(out)
        out = out * (1 / (1 + cp.exp(-out)))
        
        out = self.head_conv3.forward(out)
        
        # Output activations
        box_xy = self.sigmoid.forward(out[:, 0:2, :, :])
        box_wh = self.sigmoid.forward(out[:, 2:4, :, :])
        conf = self.sigmoid.forward(out[:, 4:5, :, :])
        
        cls_logits = out[:, 5:, :, :]
        cls_exp = cp.exp(cls_logits - cp.max(cls_logits, axis=1, keepdims=True))
        cls_probs = cls_exp / cp.sum(cls_exp, axis=1, keepdims=True)
        
        return cp.concatenate([box_xy, box_wh, conf, cls_probs], axis=1)
    
    def _forward_lateral(self, x, lateral):
        out = lateral['conv'].forward(x)
        out = lateral['bn'].forward(out)
        out = out * (1 / (1 + cp.exp(-out)))
        if lateral['pool']:
            out = lateral['pool'].forward(out)
        return out


# =============================================================================
# Factory Function
# =============================================================================

def get_model(size: str = 'large', num_classes: int = 3) -> DetectorBase:
    """
    Factory function to create detector model.
    
    Args:
        size: Model size - 'tiny', 'small', 'medium', 'large', 'xlarge', 'huge'
        num_classes: Number of object classes
        
    Returns:
        DetectorBase instance
    """
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[size]
    return DetectorBase(
        stage_channels=config['stage_channels'],
        stage_blocks=config['stage_blocks'],
        lateral_channels=config['lateral_channels'],
        num_classes=num_classes,
        use_se=config['use_se']
    )


# Convenience aliases
TinyDetector = lambda num_classes=3: get_model('tiny', num_classes)
SmallDetector = lambda num_classes=3: get_model('small', num_classes)
MediumDetector = lambda num_classes=3: get_model('medium', num_classes)
LargeDetector = lambda num_classes=3: get_model('large', num_classes)
XLargeDetector = lambda num_classes=3: get_model('xlarge', num_classes)
HugeDetector = lambda num_classes=3: get_model('huge', num_classes)

# Keep SimpleDetector as alias for LargeDetector (backward compatibility)
SimpleDetector = LargeDetector


def test_model():
    """Test all model sizes."""
    print("=" * 60)
    print("Testing CuPy Detector Models (GPU)")
    print("=" * 60)
    
    x = cp.random.randn(2, 3, 224, 224).astype(cp.float32)
    
    for size in MODEL_CONFIGS.keys():
        model = get_model(size)
        
        # Warm up
        _ = model.forward(x)
        cp.cuda.Stream.null.synchronize()
        
        import time
        start = time.time()
        output = model.forward(x)
        cp.cuda.Stream.null.synchronize()
        forward_time = (time.time() - start) * 1000
        
        print(f"{size:10s}: output={output.shape}, time={forward_time:.1f}ms")
    
    print("\n[OK] All models passed!")


if __name__ == "__main__":
    test_model()
