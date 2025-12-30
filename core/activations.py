import numpy as np
from .base import Layer

class ReLU(Layer):
    """
    Hàm kích hoạt ReLU (Rectified Linear Unit)
    """
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Lan truyền xuôi (Forward pass).
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Lan truyền ngược (Backward pass).
        """
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx


class Sigmoid(Layer):
    """
    Hàm kích hoạt Sigmoid: σ(x) = 1 / (1 + e^(-x))
    Output range: (0, 1)
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        Lan truyền xuôi (Forward pass).
        """
        self.out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip để tránh overflow
        return self.out

    def backward(self, dout):
        """
        Lan truyền ngược (Backward pass).
        Gradient: σ'(x) = σ(x) * (1 - σ(x))
        """
        return dout * self.out * (1 - self.out)
