"""
CuPy Optimizers - GPU accelerated

Implements Adam optimizer for neural network training.
"""
import cupy as cp


class Adam:
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Maintains first moment (mean) and second moment (uncentered variance)
    of gradients for each parameter.
    
    Args:
        lr: Learning rate (default: 1e-3)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0)
    """
    
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.t = 0  # Timestep
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
    
    def step(self, layers, max_grad_norm=5.0):
        """
        Perform one optimization step with norm-based gradient clipping.
        
        Args:
            layers: List of layers with trainable parameters
            max_grad_norm: Maximum gradient norm (like PyTorch clip_grad_norm_)
        """
        self.t += 1
        
        # First pass: collect all gradients to compute total norm
        all_grads = []
        grad_info = []  # Store (layer, param_name, grad_name) for later update
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W') and hasattr(layer, 'dW') and layer.dW is not None:
                all_grads.append(layer.dW.ravel())
                grad_info.append((f'{i}_W', layer, 'W', 'dW'))
                if layer.b is not None and hasattr(layer, 'db') and layer.db is not None:
                    all_grads.append(layer.db.ravel())
                    grad_info.append((f'{i}_b', layer, 'b', 'db'))
            
            if hasattr(layer, 'gamma') and hasattr(layer, 'dgamma') and layer.dgamma is not None:
                all_grads.append(layer.dgamma.ravel())
                grad_info.append((f'{i}_gamma', layer, 'gamma', 'dgamma'))
                all_grads.append(layer.dbeta.ravel())
                grad_info.append((f'{i}_beta', layer, 'beta', 'dbeta'))
            
            if hasattr(layer, 'fc1_W') and hasattr(layer, 'dfc1_W') and layer.dfc1_W is not None:
                all_grads.append(layer.dfc1_W.ravel())
                grad_info.append((f'{i}_fc1_W', layer, 'fc1_W', 'dfc1_W'))
                all_grads.append(layer.dfc2_W.ravel())
                grad_info.append((f'{i}_fc2_W', layer, 'fc2_W', 'dfc2_W'))
        
        # Compute total gradient norm
        if all_grads:
            total_grad = cp.concatenate(all_grads)
            total_norm = float(cp.sqrt(cp.sum(total_grad ** 2)))
            
            # Compute clip coefficient (like PyTorch)
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            clip_coef = min(clip_coef, 1.0)  # Only clip if norm exceeds max
        else:
            clip_coef = 1.0
        
        # Second pass: update parameters with clipped gradients
        for key, layer, param_name, grad_name in grad_info:
            self._update_param(key, layer, param_name, grad_name, clip_coef)
    
    def _update_param(self, key, layer, param_name, grad_name, clip_coef):
        """Update a single parameter using Adam with pre-computed clip coefficient."""
        param = getattr(layer, param_name)
        grad = getattr(layer, grad_name)
        
        # Apply norm-based gradient clipping
        grad = grad * clip_coef
        
        # Weight decay (L2 regularization)
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * param
        
        # Initialize moments if first time
        if key not in self.m:
            self.m[key] = cp.zeros_like(param)
            self.v[key] = cp.zeros_like(param)
        
        # Update biased first moment estimate
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        
        # Update biased second moment estimate
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        
        # Update parameter
        setattr(layer, param_name, param - self.lr * m_hat / (cp.sqrt(v_hat) + self.eps))
    
    def set_lr(self, lr):
        """Update learning rate (for schedulers)."""
        self.lr = lr


class SGD:
    """
    Stochastic Gradient Descent with optional momentum and norm-based clipping.
    
    Args:
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: L2 regularization coefficient (default: 0)
    """
    
    def __init__(self, lr=1e-3, momentum=0, weight_decay=0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}  # Velocity for momentum
    
    def step(self, layers, max_grad_norm=5.0):
        """Perform one optimization step with norm-based gradient clipping."""
        # First pass: collect all gradients to compute total norm
        all_grads = []
        grad_info = []
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W') and hasattr(layer, 'dW') and layer.dW is not None:
                all_grads.append(layer.dW.ravel())
                grad_info.append((f'{i}_W', layer, 'W', 'dW'))
                if layer.b is not None and hasattr(layer, 'db') and layer.db is not None:
                    all_grads.append(layer.db.ravel())
                    grad_info.append((f'{i}_b', layer, 'b', 'db'))
            
            if hasattr(layer, 'gamma') and hasattr(layer, 'dgamma') and layer.dgamma is not None:
                all_grads.append(layer.dgamma.ravel())
                grad_info.append((f'{i}_gamma', layer, 'gamma', 'dgamma'))
                all_grads.append(layer.dbeta.ravel())
                grad_info.append((f'{i}_beta', layer, 'beta', 'dbeta'))
            
            if hasattr(layer, 'fc1_W') and hasattr(layer, 'dfc1_W') and layer.dfc1_W is not None:
                all_grads.append(layer.dfc1_W.ravel())
                grad_info.append((f'{i}_fc1_W', layer, 'fc1_W', 'dfc1_W'))
                all_grads.append(layer.dfc2_W.ravel())
                grad_info.append((f'{i}_fc2_W', layer, 'fc2_W', 'dfc2_W'))
        
        # Compute total gradient norm
        if all_grads:
            total_grad = cp.concatenate(all_grads)
            total_norm = float(cp.sqrt(cp.sum(total_grad ** 2)))
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            clip_coef = min(clip_coef, 1.0)
        else:
            clip_coef = 1.0
        
        # Second pass: update parameters
        for key, layer, param_name, grad_name in grad_info:
            self._update_param(key, layer, param_name, grad_name, clip_coef)
    
    def _update_param(self, key, layer, param_name, grad_name, clip_coef):
        """Update a single parameter."""
        param = getattr(layer, param_name)
        grad = getattr(layer, grad_name)
        
        # Apply norm-based gradient clipping
        grad = grad * clip_coef
        
        # Weight decay
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * param
        
        if self.momentum > 0:
            if key not in self.v:
                self.v[key] = cp.zeros_like(param)
            self.v[key] = self.momentum * self.v[key] + grad
            setattr(layer, param_name, param - self.lr * self.v[key])
        else:
            setattr(layer, param_name, param - self.lr * grad)
    
    def set_lr(self, lr):
        """Update learning rate."""
        self.lr = lr
