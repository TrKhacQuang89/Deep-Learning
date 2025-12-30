class Layer:
    """Lớp cơ sở cho tất cả các lớp trong mạng thần kinh."""
    def forward(self, x):
        """Lan truyền xuôi."""
        pass
    
    def backward(self, dout):
        """Lan truyền ngược."""
        pass
