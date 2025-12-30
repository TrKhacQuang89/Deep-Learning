import numpy as np
from .base import Layer

class Softmax(Layer):
    """
    Lớp Softmax để chuyển đổi kết quả thành xác suất.
    """
    def forward(self, x):
        # Ổn định số học bằng cách trừ đi max
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        # Thông thường Softmax kết hợp với CrossEntropy, 
        # nhưng nếu dùng riêng lẻ thì gradient sẽ phức tạp hơn.
        # Ở đây chúng ta sẽ giả định dùng kết hợp để tối ưu.
        return dout

class CrossEntropyLoss:
    """
    Hàm mất mát Cross Entropy.
    """
    def __init__(self):
        self.softmax = Softmax()

    def forward(self, x, y):
        """
        x: Dự đoán (N, num_classes)
        y: Nhãn thực tế (N,) - dạng số nguyên 0..9
        """
        self.y = y
        self.probs = self.softmax.forward(x)
        N = x.shape[0]
        
        # Lấy log xác suất của nhãn đúng
        log_likelihood = -np.log(self.probs[np.arange(N), y] + 1e-12)
        loss = np.sum(log_likelihood) / N
        return loss

    def backward(self):
        """
        Gradient của Cross Entropy kết hợp với Softmax.
        dx = (probs - target) / N
        """
        N = self.y.shape[0]
        dx = self.probs.copy()
        dx[np.arange(N), self.y] -= 1
        dx /= N
        return dx
