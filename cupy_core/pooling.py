"""
CuPy Pooling Layers - GPU accelerated
"""
import cupy as cp
from .base import Layer
from .conv import im2col_indices, col2im_indices


class MaxPooling(Layer):
    """Max Pooling Layer (GPU version)."""
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.x = None
        self.max_idx = None

    def forward(self, x):
        """Forward pass."""
        self.x = x
        N, C, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride
        
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1
        
        # im2col transformation
        col = im2col_indices(x, kh, kw, padding=0, stride=self.stride)
        
        # Reshape to (C, kh*kw, N*out_h*out_w)
        col = col.reshape(C, kh*kw, N*out_h*out_w)
        
        # Max over window dimension
        self.max_idx = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1)
        
        # Reshape back
        out = out.reshape(C, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        """Backward pass."""
        N, C, H, W = self.x.shape
        kh, kw = self.kernel_size, self.kernel_size
        
        dout_flat = dout.transpose(1, 2, 3, 0).ravel()
        
        d_col = cp.zeros((C, kh*kw, N * dout.shape[2] * dout.shape[3]), dtype=dout.dtype)
        
        c_idx = cp.repeat(cp.arange(C), N * dout.shape[2] * dout.shape[3])
        flat_max_idx = self.max_idx.ravel()
        batch_spatial_idx = cp.tile(cp.arange(N * dout.shape[2] * dout.shape[3]), C)
        
        d_col[c_idx, flat_max_idx, batch_spatial_idx] = dout_flat
        
        d_col = d_col.reshape(C*kh*kw, -1)
        dx = col2im_indices(d_col, (N, C, H, W), kh, kw, padding=0, stride=self.stride)
        return dx


class AvgPooling(Layer):
    """Average Pooling Layer (GPU version)."""
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.x = None

    def forward(self, x):
        """Forward pass."""
        self.x = x
        N, C, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride
        
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1
        
        col = im2col_indices(x, kh, kw, padding=0, stride=self.stride)
        col = col.reshape(C, kh*kw, N*out_h*out_w)
        
        out = cp.mean(col, axis=1)
        
        out = out.reshape(C, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        """Backward pass."""
        N, C, H, W = self.x.shape
        kh, kw = self.kernel_size, self.kernel_size
        pool_size = kh * kw
        
        dout_flat = dout.transpose(1, 2, 3, 0).ravel()
        
        # Distribute gradient equally
        d_col = cp.zeros((C, kh*kw, N * dout.shape[2] * dout.shape[3]), dtype=dout.dtype)
        d_col[:] = dout_flat.reshape(C, 1, -1) / pool_size
        
        d_col = d_col.reshape(C*kh*kw, -1)
        dx = col2im_indices(d_col, (N, C, H, W), kh, kw, padding=0, stride=self.stride)
        return dx
