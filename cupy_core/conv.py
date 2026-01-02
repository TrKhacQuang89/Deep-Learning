"""
CuPy Convolution Layer - GPU accelerated using im2col

Same im2col approach as NumPy version, but all operations run on GPU.
"""
import cupy as cp
from .base import Layer


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
    Calculate indices for im2col operation.
    
    Args:
        x_shape: Shape of input tensor (N, C, H, W)
        field_height: Kernel height
        field_width: Kernel width
        padding: Padding size
        stride: Stride size
        
    Returns:
        tuple: Indices (k, i, j) for indexing into padded input
    """
    N, C, H, W = x_shape
    
    # Use floor division like PyTorch (no assertions)
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = cp.repeat(cp.arange(field_height), field_width)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(field_width), field_height * C)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = cp.repeat(cp.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
    Transform 4D input tensor to 2D column matrix for vectorized convolution.
    
    Args:
        x: Input tensor (N, C, H, W) - CuPy array
        field_height: Kernel height
        field_width: Kernel width
        padding: Padding size
        stride: Stride size
        
    Returns:
        cols: Reshaped matrix for matrix multiplication
    """
    p = padding
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height, field_width, padding=1, stride=1):
    """
    Transform 2D column matrix back to 4D tensor (with accumulation).
    
    Args:
        cols: Column matrix (gradients)
        x_shape: Original input shape (N, C, H, W)
        field_height: Kernel height
        field_width: Kernel width
        padding: Padding size
        stride: Stride size
        
    Returns:
        x_padded: Reconstructed 4D tensor
    """
    import cupyx
    
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    
    # CuPy's scatter_add is in cupyx module
    cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2d(Layer):
    """
    Convolutional Layer using im2col for efficient GPU computation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Padding size
        bias: If True, adds a learnable bias (default: True)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Kaiming/He initialization on GPU (fan_out mode like PyTorch)
        fan_out = out_channels * kernel_size * kernel_size
        scale = cp.sqrt(2.0 / fan_out)
        self.W = cp.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(cp.float32) * scale
        
        if self.use_bias:
            self.b = cp.zeros(out_channels, dtype=cp.float32)
        else:
            self.b = None
        
        self.x = None
        self.x_cols = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """Forward pass."""
        self.x = x
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        N, C, H, W = x.shape
        
        out_h = (H + 2 * self.padding - h_filter) // self.stride + 1
        out_w = (W + 2 * self.padding - w_filter) // self.stride + 1

        # im2col transformation
        self.x_cols = im2col_indices(x, h_filter, w_filter, padding=self.padding, stride=self.stride)
        
        # Reshape weights
        w_col = self.W.reshape(n_filters, -1)

        # Matrix multiplication: Output = W @ X_cols (+ b if bias)
        out = w_col @ self.x_cols
        if self.use_bias:
            out = out + self.b.reshape(-1, 1)
        
        # Reshape to image format
        out = out.reshape(n_filters, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        """Backward pass."""
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        
        # Reshape dout
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        
        # Gradient of bias
        if self.use_bias:
            self.db = cp.sum(dout_reshaped, axis=1)
        else:
            self.db = None
        
        # Gradient of weights
        self.dW = (dout_reshaped @ self.x_cols.T).reshape(self.W.shape)
        
        # Gradient of input
        w_reshape = self.W.reshape(n_filters, -1)
        d_cols = w_reshape.T @ dout_reshaped
        
        dx = col2im_indices(d_cols, self.x.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)
        return dx


class BatchNorm2d(Layer):
    """
    Batch Normalization for 2D inputs (GPU version).
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = cp.ones(num_features, dtype=cp.float32)
        self.beta = cp.zeros(num_features, dtype=cp.float32)
        
        # Running statistics
        self.running_mean = cp.zeros(num_features, dtype=cp.float32)
        self.running_var = cp.ones(num_features, dtype=cp.float32)
        
        # Cache for backward
        self.x_norm = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        self.training = True
    
    def forward(self, x):
        """Forward pass (N, C, H, W)."""
        if self.training:
            # Compute mean and var over (N, H, W)
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
        
        self.std = cp.sqrt(var + self.eps)
        self.x_norm = (x - mean) / self.std
        
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        
        return gamma * self.x_norm + beta
    
    def backward(self, dout):
        """Backward pass."""
        N, C, H, W = dout.shape
        M = N * H * W
        
        gamma = self.gamma.reshape(1, -1, 1, 1)
        
        self.dgamma = (dout * self.x_norm).sum(axis=(0, 2, 3))
        self.dbeta = dout.sum(axis=(0, 2, 3))
        
        dx_norm = dout * gamma
        dx = (1 / M) * (1 / self.std) * (M * dx_norm - dx_norm.sum(axis=(0, 2, 3), keepdims=True) 
                                          - self.x_norm * (dx_norm * self.x_norm).sum(axis=(0, 2, 3), keepdims=True))
        return dx
