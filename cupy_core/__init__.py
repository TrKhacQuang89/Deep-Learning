"""
CuPy Neural Network Core - GPU Accelerated

A drop-in replacement for NumPy-based core/, running on NVIDIA GPU.
All operations use CuPy arrays which reside in GPU memory.

Requirements:
    pip install cupy-cuda11x  # For CUDA 11.x
    # or
    pip install cupy-cuda12x  # For CUDA 12.x

Usage:
    import cupy as cp
    from cupy_core import Conv2d, ReLU, MaxPooling, Dense, Flatten
    
    # Create layers
    conv = Conv2d(3, 16, kernel_size=3, padding=1)
    
    # Input must be CuPy array
    x = cp.random.randn(4, 3, 224, 224).astype(cp.float32)
    out = conv.forward(x)
    
    # Convert back to NumPy if needed
    import numpy as np
    out_np = cp.asnumpy(out)
"""

from .base import Layer
from .activations import ReLU, SiLU, Sigmoid, LeakyReLU
from .conv import Conv2d, BatchNorm2d, im2col_indices, col2im_indices
from .dense import Dense, Flatten
from .pooling import MaxPooling, AvgPooling
from .losses import Softmax, CrossEntropyLoss, MSELoss, BCELoss


__all__ = [
    # Base
    'Layer',
    # Activations
    'ReLU', 'SiLU', 'Sigmoid', 'LeakyReLU',
    # Convolution
    'Conv2d', 'BatchNorm2d', 'im2col_indices', 'col2im_indices',
    # Dense
    'Dense', 'Flatten',
    # Pooling
    'MaxPooling', 'AvgPooling',
    # Losses
    'Softmax', 'CrossEntropyLoss', 'MSELoss', 'BCELoss',
]
