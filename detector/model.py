"""
Simple Object Detector Model

Kiến trúc 6 Conv layers để phát hiện 3 loại objects:
- Input: (N, 3, 224, 224)
- Output: (N, 8, 7, 7) = Grid 7x7, mỗi cell dự đoán [x, y, w, h, conf, c1, c2, c3]

Parameters: ~395K
"""

import numpy as np
import sys
sys.path.append('..')

from core import Conv2d, MaxPooling, ReLU, Sigmoid


class SimpleDetector:
    """
    Simple Grid-based Object Detector.
    
    Architecture:
        Conv1 (3→16) + ReLU + MaxPool → 112×112×16
        Conv2 (16→32) + ReLU + MaxPool → 56×56×32
        Conv3 (32→64) + ReLU + MaxPool → 28×28×64
        Conv4 (64→128) + ReLU + MaxPool → 14×14×128
        Conv5 (128→256) + ReLU + MaxPool → 7×7×256
        Conv6 (256→8) → 7×7×8 (output)
    """
    
    def __init__(self, num_classes=3, input_size=224):
        self.num_classes = num_classes
        self.input_size = input_size
        self.output_channels = 5 + num_classes  # 4 box + 1 conf + num_classes
        
        # Feature Extractor (5 Conv blocks)
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPooling(kernel_size=2, stride=2)
        
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPooling(kernel_size=2, stride=2)
        
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = ReLU()
        self.pool3 = MaxPooling(kernel_size=2, stride=2)
        
        self.conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu4 = ReLU()
        self.pool4 = MaxPooling(kernel_size=2, stride=2)
        
        self.conv5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = ReLU()
        self.pool5 = MaxPooling(kernel_size=2, stride=2)
        
        # Detection Head
        self.conv6 = Conv2d(in_channels=256, out_channels=self.output_channels, kernel_size=1, padding=0)
        
        # Activation cho output
        self.sigmoid = Sigmoid()
        
        # Danh sách layers để dễ iterate
        self.feature_layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.conv3, self.relu3, self.pool3,
            self.conv4, self.relu4, self.pool4,
            self.conv5, self.relu5, self.pool5,
        ]
        
        self.head_layers = [self.conv6]
        
        self.all_layers = self.feature_layers + self.head_layers
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, 3, H, W), H=W=224
            
        Returns:
            output: (N, 8, 7, 7) với format:
                    - channels 0-3: x, y, w, h (sigmoid applied)
                    - channel 4: confidence (sigmoid applied)
                    - channels 5-7: class probabilities (softmax applied)
        """
        # Feature extraction
        for layer in self.feature_layers:
            x = layer.forward(x)
        
        # Detection head
        x = self.conv6.forward(x)
        
        # Apply activations to output
        # x shape: (N, 8, 7, 7)
        N, C, H, W = x.shape
        
        # Tách output thành các phần
        box_xy = self.sigmoid.forward(x[:, 0:2, :, :])     # x, y: 0-1
        box_wh = self.sigmoid.forward(x[:, 2:4, :, :])     # w, h: 0-1
        conf = self.sigmoid.forward(x[:, 4:5, :, :])       # confidence: 0-1
        
        # Softmax cho class probabilities (per spatial location)
        cls_logits = x[:, 5:, :, :]  # (N, num_classes, 7, 7)
        cls_exp = np.exp(cls_logits - np.max(cls_logits, axis=1, keepdims=True))
        cls_probs = cls_exp / np.sum(cls_exp, axis=1, keepdims=True)
        
        # Ghép lại
        output = np.concatenate([box_xy, box_wh, conf, cls_probs], axis=1)
        
        # Lưu lại để backward
        self.output = output
        self.raw_output = x
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient từ loss function (N, 8, 7, 7)
            
        Returns:
            grad_input: Gradient đối với input
        """
        N, C, H, W = grad_output.shape
        
        # Gradient qua activations
        # Cần tính gradient cho sigmoid và softmax
        
        # Gradient cho box_xy (channels 0-1)
        grad_box_xy = grad_output[:, 0:2, :, :] * self.output[:, 0:2, :, :] * (1 - self.output[:, 0:2, :, :])
        
        # Gradient cho box_wh (channels 2-3)
        grad_box_wh = grad_output[:, 2:4, :, :] * self.output[:, 2:4, :, :] * (1 - self.output[:, 2:4, :, :])
        
        # Gradient cho confidence (channel 4)
        grad_conf = grad_output[:, 4:5, :, :] * self.output[:, 4:5, :, :] * (1 - self.output[:, 4:5, :, :])
        
        # Gradient cho class probs (softmax) - simplified
        # Với cross-entropy loss kết hợp softmax: grad = probs - target
        # Ở đây ta truyền gradient trực tiếp
        grad_cls = grad_output[:, 5:, :, :]
        
        # Ghép gradient
        grad = np.concatenate([grad_box_xy, grad_box_wh, grad_conf, grad_cls], axis=1)
        
        # Backward qua conv6
        grad = self.conv6.backward(grad)
        
        # Backward qua feature layers (ngược lại)
        for layer in reversed(self.feature_layers):
            grad = layer.backward(grad)
        
        return grad
    
    def update_params(self, lr, max_grad=5.0):
        """
        Cập nhật weights với learning rate và gradient clipping.
        """
        for layer in self.all_layers:
            if hasattr(layer, 'W'):
                # Gradient clipping by value
                np.clip(layer.dW, -max_grad, max_grad, out=layer.dW)
                np.clip(layer.db, -max_grad, max_grad, out=layer.db)
                
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
    
    def get_params_count(self):
        """
        Đếm tổng số parameters.
        """
        total = 0
        for layer in self.all_layers:
            if hasattr(layer, 'W'):
                total += layer.W.size + layer.b.size
        return total


def test_model():
    """Test forward và backward pass."""
    print("=" * 50)
    print("Testing SimpleDetector")
    print("=" * 50)
    
    model = SimpleDetector(num_classes=3)
    print(f"Total parameters: {model.get_params_count():,}")
    
    # Test forward
    batch_size = 2
    x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    
    print(f"\nInput shape: {x.shape}")
    
    import time
    start = time.time()
    output = model.forward(x)
    forward_time = time.time() - start
    
    print(f"Output shape: {output.shape}")
    print(f"Forward time: {forward_time:.3f}s")
    
    # Check output ranges
    print(f"\nOutput ranges:")
    print(f"  box_xy (0-1): [{output[:, 0:2].min():.3f}, {output[:, 0:2].max():.3f}]")
    print(f"  box_wh (0-1): [{output[:, 2:4].min():.3f}, {output[:, 2:4].max():.3f}]")
    print(f"  conf (0-1): [{output[:, 4:5].min():.3f}, {output[:, 4:5].max():.3f}]")
    print(f"  class probs (sum=1): {output[:, 5:, 0, 0].sum(axis=1)}")
    
    # Test backward
    grad = np.ones_like(output)
    
    start = time.time()
    grad_input = model.backward(grad)
    backward_time = time.time() - start
    
    print(f"\nGradient input shape: {grad_input.shape}")
    print(f"Backward time: {backward_time:.3f}s")
    
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    test_model()
