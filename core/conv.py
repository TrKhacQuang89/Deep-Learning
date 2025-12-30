import numpy as np
from .base import Layer

"""
GIẢI THÍCH VỀ PHÉP TOÁN IM2COL (Image to Column):

1. im2col là gì?
   - im2col là một kỹ thuật biến đổi dữ liệu hình ảnh 4D (Batch, Channel, Height, Width) 
     thành một ma trận 2D.
   - Nó hoạt động bằng cách lấy mỗi vùng "cửa sổ trượt" (sliding window) mà kernel sẽ 
     đi qua và "trải phẳng" (flatten) vùng đó thành một cột trong ma trận mới.

2. Tại sao chúng ta cần im2col?
   - Phép tích chập (Convolution) thông thường yêu cầu nhiều vòng lặp lồng nhau 
     (thường là 6-7 vòng lặp cho Batch, Channels, Height, Width, Kernel_H, Kernel_W), 
     điều này cực kỳ chậm trong Python.
   - im2col chuyển đổi bài toán tích chập thành một phép NHÂN MA TRẬN duy nhất 
     (General Matrix Multiply - GEMM).
   - Các thư viện tính toán số học như Numpy, OpenBLAS, hay MKL được tối ưu hóa cực tốt 
     cho việc nhân ma trận. Việc sử dụng GEMM giúp tận dụng bộ nhớ đệm (cache) và 
     các tập lệnh SIMD của CPU, mang lại tốc độ nhanh hơn hàng chục, hàng trăm lần 
     so với vòng lặp thủ công.

3. Quy trình:
   - Bước 1: Dùng im2col biến đầu vào thành ma trận lớn (X_col).
   - Bước 2: Biến các bộ lọc (weights) thành một ma trận (W_col).
   - Bước 3: Nhân hai ma trận này để có kết quả (Y = W_col @ X_col).
   - Bước 4: Định dạng lại (Reshape) kết quả về dạng 4D ban đầu.
"""

# Helper function for im2col (used in Conv2d)
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
    Tính toán các chỉ số cho phép toán im2col.
    
    Tham số:
        x_shape: Hình dạng của tensor đầu vào (N, C, H, W).
        field_height: Chiều cao của vùng tiếp nhận (receptive field - kernel height).
        field_width: Chiều rộng của vùng tiếp nhận (receptive field - kernel width).
        padding: Kích thước đệm (padding).
        stride: Bước nhảy (stride).
        
    Trả về:
        tuple: Các chỉ số (k, i, j) được sử dụng để lập chỉ mục trong đầu vào đã được đệm.
    """
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
    Biến đổi tensor đầu vào 4D thành ma trận 2D (dạng cột) để thực hiện tích chập vectơ hóa.
    
    Tham số:
        x: Tensor đầu vào có hình dạng (N, C, H, W).
        field_height: Chiều cao của kernel.
        field_width: Chiều rộng của kernel.
        padding: Kích thước đệm.
        stride: Bước nhảy.
        
    Trả về:
        cols: Ma trận đã được định dạng lại để thực hiện phép nhân.
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height, field_width, padding=1, stride=1):
    """
    Biến đổi ma trận 2D trở lại thành tensor 4D, cộng dồn các giá trị (thường dùng cho gradient).
    
    Tham số:
        cols: Ma trận dạng cột (ví dụ: gradients đối với các cột).
        x_shape: Hình dạng của tensor đầu vào ban đầu (N, C, H, W).
        field_height: Chiều cao của kernel.
        field_width: Chiều rộng của kernel.
        padding: Kích thước đệm.
        stride: Bước nhảy.
        
    Trả về:
        x_padded: Tensor 4D đã được tái tạo (thường là gradients).
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class Conv2d(Layer):
    """
    Lớp Tích chập (Convolutional Layer) sử dụng im2col để tính toán hiệu quả.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Khởi tạo trọng số (sử dụng Xavier/Glorot initialization)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        
        self.x = None
        self.x_cols = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Lan truyền xuôi (Forward pass).
        
        Tham số:
            x: Tensor đầu vào có hình dạng (N, C, H, W).
            
        Trả về:
            out: Kết quả sau phép tích chập.
        """
        self.x = x
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        N, C, H, W = x.shape
        
        # Tính toán kích thước đầu ra
        out_h = (H + 2 * self.padding - h_filter) // self.stride + 1
        out_w = (W + 2 * self.padding - w_filter) // self.stride + 1

        # Biến đổi ảnh thành dạng cột để nhân ma trận
        self.x_cols = im2col_indices(x, h_filter, w_filter, padding=self.padding, stride=self.stride)
        
        # Định dạng lại trọng số thành các hàng
        w_col = self.W.reshape(n_filters, -1)

        # Phép nhân ma trận tiêu chuẩn: Output = Weights * Cols + Bias
        # (out_channels, N*out_h*out_w) = (out_channels, k*k*C) @ (k*k*C, N*out_h*out_w)
        out = w_col @ self.x_cols + self.b.reshape(-1, 1)
        
        # Định dạng lại về dạng ảnh (N, out_c, out_h, out_w)
        out = out.reshape(n_filters, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        """
        Lan truyền ngược (Backward pass).
        
        Tham số:
            dout: Gradient từ lớp tiếp theo (N, out_channels, out_h, out_w).
            
        Trả về:
            dx: Gradient đối với đầu vào x.
        """
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        
        # Định dạng lại dout thành (out_channels, N*out_h*out_w)
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        
        # Gradient của Bias: tổng trên tất cả các chiều batch và không gian
        self.db = np.sum(dout_reshaped, axis=1)
        
        # Gradient của Trọng số: dout @ x_cols.T
        self.dW = (dout_reshaped @ self.x_cols.T).reshape(self.W.shape)
        
        # Gradient của Đầu vào: W.T @ dout -> sau đó dùng col2im
        w_reshape = self.W.reshape(n_filters, -1)
        d_cols = w_reshape.T @ dout_reshaped
        
        dx = col2im_indices(d_cols, self.x.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)
        return dx
