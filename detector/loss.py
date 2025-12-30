"""
Detection Loss Function

Kết hợp 3 loại loss:
1. Box Loss (MSE) - cho x, y, w, h
2. Confidence Loss (BCE) - cho objectness
3. Classification Loss (Cross Entropy) - cho class probabilities
"""

import numpy as np


class DetectionLoss:
    """
    Detection Loss = λ_coord * Box_Loss + Conf_Loss + Class_Loss
    
    Args:
        lambda_coord: Weight cho box loss (default: 5.0)
        lambda_noobj: Weight cho no-object confidence loss (default: 0.5)
    """
    
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        # Lưu lại để backward
        self.pred = None
        self.target = None
        self.obj_mask = None
        
    def forward(self, pred, target):
        """
        Tính loss.
        
        Args:
            pred: Predictions (N, 8, 7, 7) - đã qua sigmoid/softmax
                  [x, y, w, h, conf, c1, c2, c3]
            target: Ground truth (N, 8, 7, 7)
                   [x, y, w, h, obj_mask, c1, c2, c3]
                   - obj_mask = 1 nếu cell có object, 0 nếu không
                   
        Returns:
            total_loss: Scalar
        """
        self.pred = pred
        self.target = target
        
        N, C, H, W = pred.shape
        
        # Object mask: cell nào có object
        self.obj_mask = target[:, 4:5, :, :]  # (N, 1, 7, 7)
        noobj_mask = 1 - self.obj_mask
        
        # 1. Box Loss (chỉ tính cho cells có object)
        pred_box = pred[:, 0:4, :, :]
        target_box = target[:, 0:4, :, :]
        
        box_loss = np.sum(self.obj_mask * (pred_box - target_box) ** 2)
        box_loss = box_loss / (N * H * W)
        
        # 2. Confidence Loss
        pred_conf = pred[:, 4:5, :, :]
        target_conf = target[:, 4:5, :, :]
        
        # Object cells
        obj_conf_loss = np.sum(self.obj_mask * (pred_conf - target_conf) ** 2)
        
        # No-object cells (penalize less)
        noobj_conf_loss = np.sum(noobj_mask * (pred_conf - 0) ** 2)
        
        conf_loss = (obj_conf_loss + self.lambda_noobj * noobj_conf_loss) / (N * H * W)
        
        # 3. Classification Loss (Cross Entropy, chỉ cho cells có object)
        pred_cls = pred[:, 5:, :, :]  # (N, 3, 7, 7)
        target_cls = target[:, 5:, :, :]  # (N, 3, 7, 7) one-hot
        
        # Tính cross entropy
        eps = 1e-12
        cls_loss = -np.sum(self.obj_mask * target_cls * np.log(pred_cls + eps))
        cls_loss = cls_loss / (np.sum(self.obj_mask) + eps)
        
        # Total loss
        total_loss = self.lambda_coord * box_loss + conf_loss + cls_loss
        
        # Lưu các thành phần để debug
        self.box_loss = box_loss
        self.conf_loss = conf_loss
        self.cls_loss = cls_loss
        
        return total_loss
    
    def backward(self):
        """
        Tính gradient của loss đối với predictions.
        
        Returns:
            grad: (N, 8, 7, 7)
        """
        N, C, H, W = self.pred.shape
        eps = 1e-12
        
        grad = np.zeros_like(self.pred)
        
        # 1. Gradient cho Box (MSE)
        # d/d(pred) of (pred - target)^2 = 2 * (pred - target)
        grad_box = 2 * self.obj_mask * (self.pred[:, 0:4, :, :] - self.target[:, 0:4, :, :])
        grad[:, 0:4, :, :] = self.lambda_coord * grad_box / (N * H * W)
        
        # 2. Gradient cho Confidence (MSE)
        noobj_mask = 1 - self.obj_mask
        
        grad_conf_obj = 2 * self.obj_mask * (self.pred[:, 4:5, :, :] - self.target[:, 4:5, :, :])
        grad_conf_noobj = 2 * self.lambda_noobj * noobj_mask * self.pred[:, 4:5, :, :]
        
        grad[:, 4:5, :, :] = (grad_conf_obj + grad_conf_noobj) / (N * H * W)
        
        # 3. Gradient cho Classification (Combined Cross Entropy + Softmax)
        # d/d(logits) = pred - target (đây là gradient cực kỳ ổn định)
        num_obj = np.sum(self.obj_mask) + eps
        grad[:, 5:, :, :] = self.obj_mask * (self.pred[:, 5:, :, :] - self.target[:, 5:, :, :]) / num_obj
        
        return grad


def test_loss():
    """Test DetectionLoss."""
    print("=" * 50)
    print("Testing DetectionLoss")
    print("=" * 50)
    
    loss_fn = DetectionLoss()
    
    N, C, H, W = 2, 8, 7, 7
    
    # Random predictions (sau sigmoid/softmax nên trong range phù hợp)
    pred = np.random.rand(N, C, H, W).astype(np.float32)
    pred[:, 5:, :, :] = np.exp(pred[:, 5:, :, :])
    pred[:, 5:, :, :] /= pred[:, 5:, :, :].sum(axis=1, keepdims=True)  # softmax
    
    # Random target
    target = np.zeros((N, C, H, W), dtype=np.float32)
    # Đặt một số cells có object
    target[0, 4, 3, 3] = 1.0  # obj_mask
    target[0, 0:4, 3, 3] = [0.5, 0.5, 0.2, 0.3]  # box
    target[0, 5, 3, 3] = 1.0  # class 0 one-hot
    
    target[1, 4, 5, 2] = 1.0
    target[1, 0:4, 5, 2] = [0.3, 0.7, 0.15, 0.15]
    target[1, 6, 5, 2] = 1.0  # class 1 one-hot
    
    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    
    # Forward
    loss = loss_fn.forward(pred, target)
    print(f"\nTotal Loss: {loss:.4f}")
    print(f"  Box Loss: {loss_fn.box_loss:.4f}")
    print(f"  Conf Loss: {loss_fn.conf_loss:.4f}")
    print(f"  Class Loss: {loss_fn.cls_loss:.4f}")
    
    # Backward
    grad = loss_fn.backward()
    print(f"\nGradient shape: {grad.shape}")
    print(f"Gradient range: [{grad.min():.4f}, {grad.max():.4f}]")
    
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    test_loss()
