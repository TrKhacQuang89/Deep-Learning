class Layer:
    """Base class for all layers in the neural network (GPU version)."""
    def forward(self, x):
        """Forward pass."""
        pass
    
    def backward(self, dout):
        """Backward pass."""
        pass
