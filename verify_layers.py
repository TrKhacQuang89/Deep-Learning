import numpy as np
from core import Conv2d, MaxPooling, ReLU, Flatten, Dense

def test_conv2d():
    print("Testing Conv2d...")
    N, C, H, W = 2, 3, 10, 10
    x = np.random.randn(N, C, H, W)
    layer = Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
    
    out = layer.forward(x)
    print(f"Forward shape: {out.shape} (Expected: (2, 5, 10, 10))")
    assert out.shape == (2, 5, 10, 10)
    
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)
    print(f"Backward shape: {dx.shape} (Expected: (2, 3, 10, 10))")
    assert dx.shape == x.shape
    print("Conv2d Passed!\n")

def test_maxpool():
    print("Testing MaxPooling...")
    N, C, H, W = 2, 3, 8, 8
    x = np.random.randn(N, C, H, W)
    layer = MaxPooling(kernel_size=2, stride=2)
    
    out = layer.forward(x)
    print(f"Forward shape: {out.shape} (Expected: (2, 3, 4, 4))")
    assert out.shape == (2, 3, 4, 4)
    
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)
    print(f"Backward shape: {dx.shape} (Expected: (2, 3, 8, 8))")
    assert dx.shape == x.shape
    print("MaxPooling Passed!\n")

def test_relu():
    print("Testing ReLU...")
    x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    layer = ReLU()
    out = layer.forward(x)
    expected = np.array([[0.0, 2.0], [3.0, 0.0]])
    print(f"Forward output:\n{out}")
    assert np.allclose(out, expected)
    
    dout = np.array([[1.0, 1.0], [1.0, 1.0]])
    dx = layer.backward(dout)
    expected_dx = np.array([[0.0, 1.0], [1.0, 0.0]])
    print(f"Backward output:\n{dx}")
    assert np.allclose(dx, expected_dx)
    print("ReLU Passed!\n")

def test_integration():
    print("Testing simple integration (simulated tiny network)...")
    # Simulate: Input -> Conv -> Relu -> Pool -> Flatten -> Dense -> Output
    N, C, H, W = 4, 1, 6, 6
    x = np.random.randn(N, C, H, W)
    
    conv = Conv2d(1, 2, 3, padding=1)
    relu = ReLU()
    pool = MaxPooling(2, 2)
    flatten = Flatten()
    dense = Dense(2 * 3 * 3, 10) # 6x6 -> pool(2) -> 3x3. 2 channels.
    
    # Forward
    o1 = conv.forward(x)     # (4, 2, 6, 6)
    o2 = relu.forward(o1)    # (4, 2, 6, 6)
    o3 = pool.forward(o2)    # (4, 2, 3, 3)
    o4 = flatten.forward(o3) # (4, 18)
    o5 = dense.forward(o4)   # (4, 10)
    
    print(f"Final output shape: {o5.shape}")
    assert o5.shape == (4, 10)
    
    # Backward
    dout = np.random.randn(*o5.shape)
    d4 = dense.backward(dout)
    d3 = flatten.backward(d4)
    d2 = pool.backward(d3)
    d1 = relu.backward(d2)
    dx = conv.backward(d1)
    
    print(f"Input gradient shape: {dx.shape}")
    assert dx.shape == x.shape
    print("Integration Passed!\n")

if __name__ == "__main__":
    test_conv2d()
    test_maxpool()
    test_relu()
    test_integration()
