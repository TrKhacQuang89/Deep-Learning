"""
CuPy Dense/Fully Connected Layers - GPU accelerated
"""
import cupy as cp
from .base import Layer


class Flatten(Layer):
    """Flatten input to (N, -1) (GPU version)."""
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        N = x.shape[0]
        return x.reshape(N, -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)


class Dense(Layer):
    """Fully Connected Layer (GPU version)."""
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier Initialization on GPU
        scale = cp.sqrt(2.0 / in_features)
        self.W = cp.random.randn(in_features, out_features).astype(cp.float32) * scale
        self.b = cp.zeros(out_features, dtype=cp.float32)
        
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """Forward pass."""
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        """Backward pass."""
        self.dW = self.x.T @ dout
        self.db = cp.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx
