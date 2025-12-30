import numpy as np
from .base import Layer

class Flatten(Layer):
    """
    Làm phẳng đầu vào về dạng (N, -1).
    """
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        N = x.shape[0]
        return x.reshape(N, -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)


class Dense(Layer):
    """
    Lớp Kết nối đầy đủ (Fully Connected / Linear Layer).
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier Initialization
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Lan truyền xuôi (Forward pass).
        
        Tham số:
            x: Ma trận đầu vào (N, in_features).
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        """
        Lan truyền ngược (Backward pass).
        
        Tham số:
            dout: Gradient từ lớp tiếp theo (N, out_features).
        """
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx
