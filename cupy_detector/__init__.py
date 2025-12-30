"""
CuPy Detector - GPU Accelerated Object Detection

A drop-in replacement for detector/ running on NVIDIA GPU.

Usage:
    import cupy as cp
    from cupy_detector import SimpleDetector, DetectionLoss
    
    model = SimpleDetector(num_classes=3)
    loss_fn = DetectionLoss()
    
    # Data must be CuPy arrays
    x = cp.random.randn(4, 3, 224, 224).astype(cp.float32)
    output = model.forward(x)
"""

from .model import SimpleDetector
from .loss import DetectionLoss
from .utils import (
    letterbox_image,
    transform_bboxes_letterbox,
    create_target,
    decode_predictions_gpu,
    nms,
    compute_iou
)


__all__ = [
    'SimpleDetector',
    'DetectionLoss',
    'letterbox_image',
    'transform_bboxes_letterbox',
    'create_target',
    'decode_predictions_gpu',
    'nms',
    'compute_iou',
]
