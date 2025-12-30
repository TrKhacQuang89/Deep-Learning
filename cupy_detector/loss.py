"""
CuPy Detection Loss Function - GPU Accelerated

Combines 3 loss types:
1. Box Loss (MSE) - for x, y, w, h
2. Confidence Loss (BCE) - for objectness
3. Classification Loss (Cross Entropy) - for class probabilities
"""

import cupy as cp


class DetectionLoss:
    """
    Detection Loss = λ_coord * Box_Loss + Conf_Loss + Class_Loss
    
    All computations run on GPU using CuPy.
    """
    
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        self.pred = None
        self.target = None
        self.obj_mask = None
        
    def forward(self, pred, target):
        """
        Compute loss on GPU.
        
        Args:
            pred: Predictions (N, 8, 7, 7) - CuPy array after sigmoid/softmax
            target: Ground truth (N, 8, 7, 7) - CuPy array
                    
        Returns:
            total_loss: Scalar (CuPy)
        """
        self.pred = pred
        self.target = target
        
        N, C, H, W = pred.shape
        
        # Object mask
        self.obj_mask = target[:, 4:5, :, :]
        noobj_mask = 1 - self.obj_mask
        
        # 1. Box Loss (only for cells with objects)
        pred_box = pred[:, 0:4, :, :]
        target_box = target[:, 0:4, :, :]
        
        box_loss = cp.sum(self.obj_mask * (pred_box - target_box) ** 2)
        box_loss = box_loss / (N * H * W)
        
        # 2. Confidence Loss
        pred_conf = pred[:, 4:5, :, :]
        target_conf = target[:, 4:5, :, :]
        
        obj_conf_loss = cp.sum(self.obj_mask * (pred_conf - target_conf) ** 2)
        noobj_conf_loss = cp.sum(noobj_mask * pred_conf ** 2)
        
        conf_loss = (obj_conf_loss + self.lambda_noobj * noobj_conf_loss) / (N * H * W)
        
        # 3. Classification Loss (Cross Entropy)
        pred_cls = pred[:, 5:, :, :]
        target_cls = target[:, 5:, :, :]
        
        eps = 1e-12
        cls_loss = -cp.sum(self.obj_mask * target_cls * cp.log(pred_cls + eps))
        cls_loss = cls_loss / (cp.sum(self.obj_mask) + eps)
        
        # Total loss
        total_loss = self.lambda_coord * box_loss + conf_loss + cls_loss
        
        # Store components for debugging
        self.box_loss = float(box_loss)
        self.conf_loss = float(conf_loss)
        self.cls_loss = float(cls_loss)
        
        return total_loss
    
    def backward(self):
        """
        Compute gradient on GPU.
        
        Returns:
            grad: (N, 8, 7, 7) - CuPy array
        """
        N, C, H, W = self.pred.shape
        eps = 1e-12
        
        grad = cp.zeros_like(self.pred)
        
        # 1. Gradient for Box (MSE)
        grad_box = 2 * self.obj_mask * (self.pred[:, 0:4, :, :] - self.target[:, 0:4, :, :])
        grad[:, 0:4, :, :] = self.lambda_coord * grad_box / (N * H * W)
        
        # 2. Gradient for Confidence
        noobj_mask = 1 - self.obj_mask
        
        grad_conf_obj = 2 * self.obj_mask * (self.pred[:, 4:5, :, :] - self.target[:, 4:5, :, :])
        grad_conf_noobj = 2 * self.lambda_noobj * noobj_mask * self.pred[:, 4:5, :, :]
        
        grad[:, 4:5, :, :] = (grad_conf_obj + grad_conf_noobj) / (N * H * W)
        
        # 3. Gradient for Classification
        num_obj = cp.sum(self.obj_mask) + eps
        grad[:, 5:, :, :] = self.obj_mask * (self.pred[:, 5:, :, :] - self.target[:, 5:, :, :]) / num_obj
        
        return grad


def test_loss():
    """Test DetectionLoss on GPU."""
    print("=" * 50)
    print("Testing CuPy DetectionLoss (GPU)")
    print("=" * 50)
    
    loss_fn = DetectionLoss()
    
    N, C, H, W = 2, 8, 7, 7
    
    # Random predictions
    pred = cp.random.rand(N, C, H, W).astype(cp.float32)
    pred[:, 5:, :, :] = cp.exp(pred[:, 5:, :, :])
    pred[:, 5:, :, :] /= pred[:, 5:, :, :].sum(axis=1, keepdims=True)
    
    # Random target
    target = cp.zeros((N, C, H, W), dtype=cp.float32)
    target[0, 4, 3, 3] = 1.0
    target[0, 0:4, 3, 3] = cp.array([0.5, 0.5, 0.2, 0.3])
    target[0, 5, 3, 3] = 1.0
    
    target[1, 4, 5, 2] = 1.0
    target[1, 0:4, 5, 2] = cp.array([0.3, 0.7, 0.15, 0.15])
    target[1, 6, 5, 2] = 1.0
    
    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    
    # Forward
    loss = loss_fn.forward(pred, target)
    print(f"\nTotal Loss: {float(loss):.4f}")
    print(f"  Box Loss: {loss_fn.box_loss:.4f}")
    print(f"  Conf Loss: {loss_fn.conf_loss:.4f}")
    print(f"  Class Loss: {loss_fn.cls_loss:.4f}")
    
    # Backward
    grad = loss_fn.backward()
    print(f"\nGradient shape: {grad.shape}")
    print(f"Gradient range: [{float(grad.min()):.4f}, {float(grad.max()):.4f}]")
    
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    test_loss()
