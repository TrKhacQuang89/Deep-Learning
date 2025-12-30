"""
CuPy Activation Functions - GPU accelerated
"""
import cupy as cp
from .base import Layer


class ReLU(Layer):
    """ReLU activation function (GPU version)."""
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return cp.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx


class SiLU(Layer):
    """SiLU/Swish activation: x * sigmoid(x) (GPU version)."""
    def __init__(self):
        self.x = None
        self.sigmoid_x = None

    def forward(self, x):
        self.x = x
        self.sigmoid_x = 1 / (1 + cp.exp(-cp.clip(x, -500, 500)))
        return x * self.sigmoid_x

    def backward(self, dout):
        # d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        grad = self.sigmoid_x * (1 + self.x * (1 - self.sigmoid_x))
        return dout * grad


class Sigmoid(Layer):
    """Sigmoid activation: σ(x) = 1 / (1 + e^(-x)) (GPU version)."""
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + cp.exp(-cp.clip(x, -500, 500)))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class LeakyReLU(Layer):
    """Leaky ReLU activation (GPU version)."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.x = None

    def forward(self, x):
        self.x = x
        return cp.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        return cp.where(self.x > 0, dout, self.alpha * dout)
