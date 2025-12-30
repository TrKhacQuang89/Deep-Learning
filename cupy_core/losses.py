"""
CuPy Loss Functions - GPU accelerated
"""
import cupy as cp
from .base import Layer


class Softmax(Layer):
    """Softmax layer to convert logits to probabilities (GPU version)."""
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        # Numerical stability: subtract max
        exps = cp.exp(x - cp.max(x, axis=1, keepdims=True))
        self.out = exps / cp.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        return dout


class CrossEntropyLoss:
    """Cross Entropy Loss (GPU version)."""
    def __init__(self):
        self.softmax = Softmax()
        self.y = None
        self.probs = None

    def forward(self, x, y):
        """
        Args:
            x: Predictions (N, num_classes)
            y: Labels (N,) - integer indices
        """
        self.y = y
        self.probs = self.softmax.forward(x)
        N = x.shape[0]
        
        log_likelihood = -cp.log(self.probs[cp.arange(N), y] + 1e-12)
        loss = cp.sum(log_likelihood) / N
        return loss

    def backward(self):
        """Gradient of CrossEntropy + Softmax."""
        N = self.y.shape[0]
        dx = self.probs.copy()
        dx[cp.arange(N), self.y] -= 1
        dx /= N
        return dx


class MSELoss:
    """Mean Squared Error Loss (GPU version)."""
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (N, ...)
            target: Targets (N, ...)
        """
        self.pred = pred
        self.target = target
        return cp.mean((pred - target) ** 2)

    def backward(self):
        """Gradient of MSE."""
        N = self.pred.shape[0]
        return 2 * (self.pred - self.target) / (N * self.pred[0].size)


class BCELoss:
    """Binary Cross Entropy Loss (GPU version)."""
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (N, ...) after sigmoid
            target: Binary targets (N, ...)
        """
        self.pred = cp.clip(pred, 1e-7, 1 - 1e-7)
        self.target = target
        loss = -cp.mean(target * cp.log(self.pred) + (1 - target) * cp.log(1 - self.pred))
        return loss

    def backward(self):
        """Gradient of BCE."""
        N = self.pred.shape[0]
        return (self.pred - self.target) / (self.pred * (1 - self.pred) * N * self.pred[0].size + 1e-7)
