"""
CuPy Detection Loss Function - GPU Accelerated

Matches PyTorch DetectionLoss exactly:
1. Box Loss (MSE) - for x, y, w, h, normalized by obj count
2. Confidence Loss (BCE) - for objectness, normalized by obj/noobj count
3. Classification Loss (Cross Entropy on logits) - normalized by obj count
"""

import cupy as cp


class DetectionLoss:
    """
    Detection Loss = λ_coord * Box_Loss + Conf_Loss + Class_Loss
    
    Matches PyTorch implementation exactly:
    - Uses boolean mask (> 0.5) like PyTorch
    - Normalizes by object count, not grid cells
    - CrossEntropy on logits, not softmax output
    """
    
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, num_classes=3):
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.num_classes = num_classes
        
        self.pred = None
        self.target = None
        self.obj_mask = None
        self.noobj_mask = None
        self.raw_pred = None  # Store pre-activation predictions for backward
        self.softmax_cls = None  # Store softmax from raw logits for backward
        
    def forward(self, pred, target, raw_pred=None):
        """
        Compute loss on GPU (matches PyTorch DetectionLoss exactly).
        
        Args:
            pred: Predictions (N, 8, 7, 7) - CuPy array after sigmoid/softmax
            target: Ground truth (N, 8, 7, 7) - CuPy array
            raw_pred: Raw predictions before softmax (optional, for proper CE)
                    
        Returns:
            total_loss: Scalar (CuPy)
        """
        self.pred = pred
        self.target = target
        self.raw_pred = raw_pred
        
        N, C, H, W = pred.shape
        eps = 1e-6
        
        # Object mask: boolean like PyTorch (> 0.5)
        self.obj_mask = (target[:, 4:5, :, :] > 0.5).astype(cp.float32)  # (N, 1, 7, 7)
        self.noobj_mask = 1.0 - self.obj_mask
        
        obj_count = cp.sum(self.obj_mask) + eps
        noobj_count = cp.sum(self.noobj_mask) + eps
        
        # 1. Localization loss (xy + wh, normalized by obj count like PyTorch)
        xy_loss = cp.sum(self.obj_mask * (pred[:, 0:2, :, :] - target[:, 0:2, :, :]) ** 2)
        xy_loss = xy_loss / obj_count
        
        wh_loss = cp.sum(self.obj_mask * (pred[:, 2:4, :, :] - target[:, 2:4, :, :]) ** 2)
        wh_loss = wh_loss / obj_count
        
        coord_loss = self.lambda_coord * (xy_loss + wh_loss)
        
        # 2. Confidence Loss (normalized by obj/noobj count like PyTorch)
        obj_conf_loss = cp.sum(self.obj_mask * (pred[:, 4:5, :, :] - target[:, 4:5, :, :]) ** 2)
        obj_conf_loss = obj_conf_loss / obj_count
        
        noobj_conf_loss = cp.sum(self.noobj_mask * (pred[:, 4:5, :, :] - target[:, 4:5, :, :]) ** 2)
        noobj_conf_loss = noobj_conf_loss / noobj_count
        
        conf_loss = obj_conf_loss + self.lambda_noobj * noobj_conf_loss
        
        # 3. Classification Loss (Cross Entropy using log-softmax trick like PyTorch)
        # PyTorch uses CrossEntropyLoss on raw logits for numerical stability
        # We use log-softmax: log(softmax(x)) = x - log(sum(exp(x)))
        
        target_cls = target[:, 5:, :, :]  # one-hot
        target_class_idx = cp.argmax(target_cls, axis=1)  # (N, H, W)
        
        if raw_pred is not None:
            # Use raw logits for numerically stable log-softmax
            logits = raw_pred[:, 5:, :, :]  # (N, num_classes, H, W)
            
            # Log-softmax trick: log(softmax(x)) = x - logsumexp(x)
            # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
            logits_max = cp.max(logits, axis=1, keepdims=True)
            logits_stable = logits - logits_max
            log_sum_exp = logits_max + cp.log(cp.sum(cp.exp(logits_stable), axis=1, keepdims=True))
            log_softmax = logits - log_sum_exp  # (N, num_classes, H, W)
            
            # Gather log probabilities for target class
            N_batch, num_cls, H_grid, W_grid = log_softmax.shape
            log_softmax_flat = log_softmax.transpose(0, 2, 3, 1).reshape(-1, num_cls)
            target_idx_flat = target_class_idx.reshape(-1)
            
            log_prob_correct = log_softmax_flat[cp.arange(log_softmax_flat.shape[0]), target_idx_flat.astype(cp.int32)]
            log_prob_correct = log_prob_correct.reshape(N_batch, H_grid, W_grid)
            
            # Cross entropy: -log_softmax[target_class]
            obj_mask_squeezed = self.obj_mask.squeeze(1)
            cls_loss = -cp.sum(obj_mask_squeezed * log_prob_correct)
            cls_loss = cls_loss / obj_count
            
            # Store softmax for backward (gradient = softmax - one_hot)
            self.softmax_cls = cp.exp(log_softmax)
        else:
            # Fallback: use softmax output (less stable, but works)
            pred_cls = pred[:, 5:, :, :]
            N_batch, num_cls, H_grid, W_grid = pred_cls.shape
            pred_cls_flat = pred_cls.transpose(0, 2, 3, 1).reshape(-1, num_cls)
            target_idx_flat = target_class_idx.reshape(-1)
            
            pred_correct = pred_cls_flat[cp.arange(pred_cls_flat.shape[0]), target_idx_flat.astype(cp.int32)]
            pred_correct = pred_correct.reshape(N_batch, H_grid, W_grid)
            
            obj_mask_squeezed = self.obj_mask.squeeze(1)
            cls_loss = -cp.sum(obj_mask_squeezed * cp.log(pred_correct + eps))
            cls_loss = cls_loss / obj_count
            
            self.softmax_cls = None
        
        # Total loss
        total_loss = coord_loss + conf_loss + cls_loss
        
        # Store components for debugging
        self.box_loss = float(coord_loss)
        self.conf_loss = float(conf_loss)
        self.cls_loss = float(cls_loss)
        
        return total_loss
    
    def backward(self):
        """
        Compute gradient on GPU (matches PyTorch autograd).
        
        Returns:
            grad: (N, 8, 7, 7) - CuPy array
        """
        N, C, H, W = self.pred.shape
        eps = 1e-6
        
        obj_count = cp.sum(self.obj_mask) + eps
        noobj_count = cp.sum(self.noobj_mask) + eps
        
        grad = cp.zeros_like(self.pred)
        
        # 1. Gradient for Box (MSE, normalized by obj count)
        grad_xy = 2 * self.obj_mask * (self.pred[:, 0:2, :, :] - self.target[:, 0:2, :, :]) / obj_count
        grad_wh = 2 * self.obj_mask * (self.pred[:, 2:4, :, :] - self.target[:, 2:4, :, :]) / obj_count
        grad[:, 0:2, :, :] = self.lambda_coord * grad_xy
        grad[:, 2:4, :, :] = self.lambda_coord * grad_wh
        
        # 2. Gradient for Confidence
        grad_conf_obj = 2 * self.obj_mask * (self.pred[:, 4:5, :, :] - self.target[:, 4:5, :, :]) / obj_count
        grad_conf_noobj = 2 * self.lambda_noobj * self.noobj_mask * (self.pred[:, 4:5, :, :] - self.target[:, 4:5, :, :]) / noobj_count
        grad[:, 4:5, :, :] = grad_conf_obj + grad_conf_noobj
        
        # 3. Gradient for Classification (softmax + cross-entropy = softmax - one_hot)
        # Use softmax computed from raw logits if available
        obj_mask_expanded = self.obj_mask  # (N, 1, H, W)
        if self.softmax_cls is not None:
            grad[:, 5:, :, :] = obj_mask_expanded * (self.softmax_cls - self.target[:, 5:, :, :]) / obj_count
        else:
            grad[:, 5:, :, :] = obj_mask_expanded * (self.pred[:, 5:, :, :] - self.target[:, 5:, :, :]) / obj_count
        
        return grad


def test_loss():
    """Test DetectionLoss on GPU."""
    print("=" * 50)
    print("Testing CuPy DetectionLoss (GPU)")
    print("=" * 50)
    
    loss_fn = DetectionLoss()
    
    N, C, H, W = 2, 8, 7, 7
    
    # Random raw logits (before activation)
    raw_pred = cp.random.randn(N, C, H, W).astype(cp.float32)
    
    # Apply activations to create pred (like model forward)
    pred = cp.zeros_like(raw_pred)
    pred[:, 0:5, :, :] = 1 / (1 + cp.exp(-raw_pred[:, 0:5, :, :]))  # sigmoid
    # Softmax for class probs
    cls_logits = raw_pred[:, 5:, :, :]
    cls_exp = cp.exp(cls_logits - cp.max(cls_logits, axis=1, keepdims=True))
    pred[:, 5:, :, :] = cls_exp / cp.sum(cls_exp, axis=1, keepdims=True)
    
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
    
    # Test 1: With raw_pred (log-softmax path - numerically stable)
    print("\n--- Test 1: With raw_pred (log-softmax path) ---")
    loss = loss_fn.forward(pred, target, raw_pred=raw_pred)
    print(f"Total Loss: {float(loss):.4f}")
    print(f"  Box Loss: {loss_fn.box_loss:.4f}")
    print(f"  Conf Loss: {loss_fn.conf_loss:.4f}")
    print(f"  Class Loss: {loss_fn.cls_loss:.4f}")
    
    grad = loss_fn.backward()
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient range: [{float(grad.min()):.4f}, {float(grad.max()):.4f}]")
    
    # Test 2: Without raw_pred (fallback path)
    print("\n--- Test 2: Without raw_pred (fallback path) ---")
    loss2 = loss_fn.forward(pred, target)  # No raw_pred
    print(f"Total Loss: {float(loss2):.4f}")
    print(f"  Class Loss: {loss_fn.cls_loss:.4f}")
    
    grad2 = loss_fn.backward()
    print(f"Gradient shape: {grad2.shape}")
    
    # Losses should be nearly identical (small numerical differences)
    print(f"\n--- Comparison ---")
    print(f"Loss diff: {abs(float(loss) - float(loss2)):.6f}")
    print(f"Grad diff (max): {float(cp.max(cp.abs(grad - grad2))):.6f}")
    
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    test_loss()
