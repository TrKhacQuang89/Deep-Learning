import numpy as np
from .base import Layer
from .conv import im2col_indices, col2im_indices # Reuse im2col helper

class MaxPooling(Layer):
    """
    Lớp Max Pooling.
    """
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.x = None

    def forward(self, x):
        """
        Lan truyền xuôi (Forward pass).
        
        Tham số:
            x: Tensor đầu vào có hình dạng (N, C, H, W).
        """
        self.x = x
        N, C, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride
        
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1
        
        # Reshape to easily take max
        col = im2col_indices(x, kh, kw, padding=0, stride=self.stride)
        
        # reshape to (C, kh*kw, N*out_h*out_w)
        col = col.reshape(C, kh*kw, N*out_h*out_w)
        
        # Max over the window dimension (axis 1)
        self.max_idx = np.argmax(col, axis=1) # Store indices for backward
        out = np.max(col, axis=1)
        
        # Reshape output back to (N, C, H, W)
        out = out.reshape(C, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        """
        Lan truyền ngược (Backward pass).
        
        Tham số:
            dout: Gradient từ lớp tiếp theo (N, C, out_h, out_w).
        """
        N, C, H, W = self.x.shape
        kh, kw = self.kernel_size, self.kernel_size
        
        # Initialize flat gradient matrix
        dout_flat = dout.transpose(1, 2, 3, 0).ravel() # (C * out_h * out_w * N)
        
        # Construct the sparse gradient matrix for the columns
        d_col = np.zeros((C, kh*kw, N * dout.shape[2] * dout.shape[3]))
        
        # Advanced indexing to place gradients
        c_idx = np.repeat(np.arange(C), N * dout.shape[2] * dout.shape[3])
        flat_max_idx = self.max_idx.ravel() # These are 0..kh*kw-1
        batch_spatial_idx = np.tile(np.arange(N * dout.shape[2] * dout.shape[3]), C)
        
        # Assign gradients to the max locations
        d_col[c_idx, flat_max_idx, batch_spatial_idx] = dout_flat
        
        # Reshape back to full column form for col2im
        d_col = d_col.reshape(C*kh*kw, -1)
        
        dx = col2im_indices(d_col, (N, C, H, W), kh, kw, padding=0, stride=self.stride)
        return dx
