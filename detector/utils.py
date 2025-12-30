"""
Utility functions cho Object Detection.

Các hàm:
- decode_predictions: Chuyển grid output thành bounding boxes
- nms: Non-Maximum Suppression
- compute_iou: Tính Intersection over Union
- create_target: Tạo ground truth grid từ annotations
"""

import numpy as np


# =============================================================================
# IMAGE PREPROCESSING - LETTERBOX
# =============================================================================

def letterbox_image(image, target_size=224, pad_value=128):
    """
    Letterbox: Pad ảnh thành hình vuông rồi resize.
    Giữ nguyên aspect ratio, không méo objects.
    
    Args:
        image: numpy array (H, W, C) hoặc (H, W)
        target_size: Kích thước output (vuông)
        pad_value: Giá trị pixel để padding (0-255)
        
    Returns:
        resized: numpy array (target_size, target_size, C)
        info: dict chứa thông tin để transform bbox
              {'orig_h', 'orig_w', 'pad_top', 'pad_left', 'scale', 'new_size'}
    """
    # Handle grayscale
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    orig_h, orig_w, channels = image.shape
    
    # Tính padding để thành hình vuông
    max_side = max(orig_h, orig_w)
    
    # Padding amounts
    pad_top = (max_side - orig_h) // 2
    pad_bottom = max_side - orig_h - pad_top
    pad_left = (max_side - orig_w) // 2
    pad_right = max_side - orig_w - pad_left
    
    # Pad image
    padded = np.full((max_side, max_side, channels), pad_value, dtype=image.dtype)
    padded[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w, :] = image
    
    # Resize to target size
    # Simple resize using nearest neighbor (for NumPy without cv2)
    scale = target_size / max_side
    resized = _resize_image(padded, target_size, target_size)
    
    # Info for bbox transformation
    info = {
        'orig_h': orig_h,
        'orig_w': orig_w,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'padded_size': max_side,
        'scale': scale,
        'target_size': target_size
    }
    
    return resized, info


def _resize_image(image, new_h, new_w):
    """
    Resize image using bilinear interpolation (NumPy implementation).
    
    Args:
        image: (H, W, C) numpy array
        new_h, new_w: Target dimensions
        
    Returns:
        Resized image (new_h, new_w, C)
    """
    orig_h, orig_w, channels = image.shape
    
    # Create coordinate grids for new image
    x_ratio = orig_w / new_w
    y_ratio = orig_h / new_h
    
    # New coordinates mapped to original
    x = np.arange(new_w) * x_ratio
    y = np.arange(new_h) * y_ratio
    
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, orig_w - 1)
    y1 = np.minimum(y0 + 1, orig_h - 1)
    
    # Weights
    wx = x - x0
    wy = y - y0
    
    # Bilinear interpolation
    resized = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    
    for c in range(channels):
        for i in range(new_h):
            for j in range(new_w):
                # 4 corners
                v00 = image[y0[i], x0[j], c]
                v01 = image[y0[i], x1[j], c]
                v10 = image[y1[i], x0[j], c]
                v11 = image[y1[i], x1[j], c]
                
                # Interpolate
                v = (1 - wy[i]) * ((1 - wx[j]) * v00 + wx[j] * v01) + \
                    wy[i] * ((1 - wx[j]) * v10 + wx[j] * v11)
                resized[i, j, c] = v
    
    return resized


def transform_bbox_letterbox(bbox, info):
    """
    Transform bbox coordinates sau khi letterbox.
    
    Input bbox: [x_center, y_center, width, height] normalized (0-1) theo ảnh gốc
    Output bbox: [x_center, y_center, width, height] normalized (0-1) theo ảnh sau letterbox
    
    Args:
        bbox: [x, y, w, h] normalized theo ảnh gốc
        info: dict từ letterbox_image()
        
    Returns:
        new_bbox: [x, y, w, h] normalized theo ảnh vuông sau letterbox
    """
    x, y, w, h = bbox
    
    orig_h = info['orig_h']
    orig_w = info['orig_w']
    pad_top = info['pad_top']
    pad_left = info['pad_left']
    padded_size = info['padded_size']
    
    # Bước 1: Chuyển từ normalized → pixel coordinates (ảnh gốc)
    x_pixel = x * orig_w
    y_pixel = y * orig_h
    w_pixel = w * orig_w
    h_pixel = h * orig_h
    
    # Bước 2: Shift theo padding
    x_padded = x_pixel + pad_left
    y_padded = y_pixel + pad_top
    # Width và height không đổi (chỉ shift position)
    
    # Bước 3: Normalize theo kích thước mới (vuông)
    x_new = x_padded / padded_size
    y_new = y_padded / padded_size
    w_new = w_pixel / padded_size
    h_new = h_pixel / padded_size
    
    return [x_new, y_new, w_new, h_new]


def transform_bboxes_letterbox(bboxes, info):
    """
    Transform nhiều bboxes cùng lúc.
    
    Args:
        bboxes: List of [class_id, x, y, w, h] hoặc numpy array (N, 5)
        info: dict từ letterbox_image()
        
    Returns:
        new_bboxes: List of [class_id, x_new, y_new, w_new, h_new]
    """
    new_bboxes = []
    for bbox in bboxes:
        class_id = bbox[0]
        coords = bbox[1:5]
        new_coords = transform_bbox_letterbox(coords, info)
        new_bboxes.append([class_id] + new_coords)
    return new_bboxes


def inverse_transform_bbox_letterbox(bbox, info):
    """
    Chuyển bbox từ không gian letterbox về không gian ảnh gốc.
    Dùng khi decode predictions để vẽ lên ảnh gốc.
    
    Args:
        bbox: [x, y, w, h] normalized theo ảnh vuông
        info: dict từ letterbox_image()
        
    Returns:
        orig_bbox: [x, y, w, h] normalized theo ảnh gốc
    """
    x, y, w, h = bbox
    
    orig_h = info['orig_h']
    orig_w = info['orig_w']
    pad_top = info['pad_top']
    pad_left = info['pad_left']
    padded_size = info['padded_size']
    
    # Bước 1: Từ normalized → pixel trong ảnh vuông
    x_padded = x * padded_size
    y_padded = y * padded_size
    w_padded = w * padded_size
    h_padded = h * padded_size
    
    # Bước 2: Shift ngược lại (bỏ padding offset)
    x_pixel = x_padded - pad_left
    y_pixel = y_padded - pad_top
    # Width, height giữ nguyên
    
    # Bước 3: Normalize về kích thước ảnh gốc
    x_orig = x_pixel / orig_w
    y_orig = y_pixel / orig_h
    w_orig = w_padded / orig_w
    h_orig = h_padded / orig_h
    
    # Clip to valid range
    x_orig = np.clip(x_orig, 0, 1)
    y_orig = np.clip(y_orig, 0, 1)
    w_orig = np.clip(w_orig, 0, 1)
    h_orig = np.clip(h_orig, 0, 1)
    
    return [x_orig, y_orig, w_orig, h_orig]


# =============================================================================
# IoU AND BOX UTILITIES
# =============================================================================

def compute_iou(box1, box2):
    """
    Tính IoU giữa 2 boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] (corners format)
        
    Returns:
        iou: float
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
        
    return inter_area / union_area


def xywh_to_xyxy(box):
    """
    Chuyển từ center format sang corner format.
    
    Args:
        box: [x_center, y_center, width, height] (normalized 0-1)
        
    Returns:
        [x1, y1, x2, y2]
    """
    x, y, w, h = box
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]


def xyxy_to_xywh(box):
    """
    Chuyển từ corner format sang center format.
    
    Args:
        box: [x1, y1, x2, y2]
        
    Returns:
        [x_center, y_center, width, height]
    """
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [x, y, w, h]


def decode_predictions(output, conf_thresh=0.5, grid_size=7):
    """
    Chuyển grid output thành danh sách bounding boxes.
    
    Args:
        output: (8, 7, 7) - [x, y, w, h, conf, c1, c2, c3]
        conf_thresh: Ngưỡng confidence
        grid_size: Kích thước grid (7)
        
    Returns:
        boxes: List of [x1, y1, x2, y2, confidence, class_id]
    """
    boxes = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            conf = output[4, i, j]
            
            if conf < conf_thresh:
                continue
            
            # Decode box coordinates
            # x, y là offset trong cell → chuyển sang tọa độ global
            x = (j + output[0, i, j]) / grid_size
            y = (i + output[1, i, j]) / grid_size
            w = output[2, i, j]
            h = output[3, i, j]
            
            # Chuyển sang corner format
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            # Clip to [0, 1]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
            
            # Class
            class_probs = output[5:, i, j]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            # Final confidence = obj_conf * class_conf
            final_conf = conf * class_conf
            
            if final_conf >= conf_thresh:
                boxes.append([x1, y1, x2, y2, final_conf, class_id])
    
    return boxes


def nms(boxes, iou_thresh=0.5):
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: List of [x1, y1, x2, y2, confidence, class_id]
        iou_thresh: IoU threshold
        
    Returns:
        kept_boxes: List sau khi loại bỏ overlapping boxes
    """
    if len(boxes) == 0:
        return []
    
    # Sort by confidence (descending)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    kept = []
    
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        
        # Remove boxes with high IoU
        remaining = []
        for box in boxes:
            if box[5] != best[5]:  # Different class → keep
                remaining.append(box)
            else:
                iou = compute_iou(best[:4], box[:4])
                if iou < iou_thresh:
                    remaining.append(box)
        
        boxes = remaining
    
    return kept


def create_target(annotations, num_classes=3, grid_size=7):
    """
    Tạo ground truth grid từ annotations.
    
    Args:
        annotations: List of [class_id, x_center, y_center, w, h] (normalized 0-1)
        num_classes: Số classes
        grid_size: Kích thước grid
        
    Returns:
        target: (8, 7, 7) - [x, y, w, h, obj_mask, one_hot_classes]
    """
    target = np.zeros((5 + num_classes, grid_size, grid_size), dtype=np.float32)
    
    for ann in annotations:
        class_id, x, y, w, h = ann
        class_id = int(class_id)
        
        # Xác định cell chứa center của object
        grid_x = int(x * grid_size)
        grid_y = int(y * grid_size)
        
        # Clip to valid range
        grid_x = min(grid_size - 1, max(0, grid_x))
        grid_y = min(grid_size - 1, max(0, grid_y))
        
        # Offset trong cell
        x_offset = x * grid_size - grid_x
        y_offset = y * grid_size - grid_y
        
        # Chỉ assign nếu cell chưa có object (1 object/cell)
        if target[4, grid_y, grid_x] == 0:
            target[0, grid_y, grid_x] = x_offset
            target[1, grid_y, grid_x] = y_offset
            target[2, grid_y, grid_x] = w
            target[3, grid_y, grid_x] = h
            target[4, grid_y, grid_x] = 1.0  # Object mask
            target[5 + class_id, grid_y, grid_x] = 1.0  # One-hot class
    
    return target


def test_utils():
    """Test utility functions."""
    print("=" * 50)
    print("Testing Utility Functions")
    print("=" * 50)
    
    # Test IoU
    box1 = [0.1, 0.1, 0.5, 0.5]
    box2 = [0.3, 0.3, 0.7, 0.7]
    iou = compute_iou(box1, box2)
    print(f"IoU between boxes: {iou:.3f}")
    
    # Test create_target
    annotations = [
        [0, 0.5, 0.5, 0.2, 0.3],   # class 0, center at (0.5, 0.5)
        [1, 0.3, 0.7, 0.15, 0.15], # class 1, center at (0.3, 0.7)
        [2, 0.8, 0.2, 0.1, 0.2],   # class 2, center at (0.8, 0.2)
    ]
    
    target = create_target(annotations, num_classes=3, grid_size=7)
    print(f"\nTarget shape: {target.shape}")
    print(f"Number of cells with objects: {int(target[4].sum())}")
    
    # Test decode_predictions
    fake_output = np.random.rand(8, 7, 7).astype(np.float32)
    fake_output[4, :, :] = 0.1  # Low confidence everywhere
    fake_output[4, 3, 3] = 0.9  # High confidence at (3, 3)
    fake_output[0:4, 3, 3] = [0.5, 0.5, 0.2, 0.3]
    fake_output[5:, 3, 3] = [0.7, 0.2, 0.1]  # Class probs
    
    boxes = decode_predictions(fake_output, conf_thresh=0.3)
    print(f"\nDecoded boxes: {len(boxes)}")
    for box in boxes:
        print(f"  {box}")
    
    # Test NMS
    test_boxes = [
        [0.1, 0.1, 0.5, 0.5, 0.9, 0],
        [0.12, 0.12, 0.52, 0.52, 0.8, 0],  # Overlapping with first
        [0.7, 0.7, 0.9, 0.9, 0.7, 1],       # Different location
    ]
    
    kept = nms(test_boxes, iou_thresh=0.5)
    print(f"\nAfter NMS: {len(kept)} boxes (from {len(test_boxes)})")
    
    # Test letterbox transform
    print("\n" + "-" * 50)
    print("Testing Letterbox Transform")
    print("-" * 50)
    
    # Simulate portrait image (918 x 1232)
    orig_h, orig_w = 1232, 918
    padded_size = max(orig_h, orig_w)  # 1232
    pad_top = (padded_size - orig_h) // 2  # 0
    pad_left = (padded_size - orig_w) // 2  # 157
    
    info = {
        'orig_h': orig_h,
        'orig_w': orig_w,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'padded_size': padded_size,
        'scale': 224 / padded_size,
        'target_size': 224
    }
    
    print(f"   Original image: {orig_w}x{orig_h} (W x H)")
    print(f"   Padded to square: {padded_size}x{padded_size}")
    print(f"   Padding: top={pad_top}, left={pad_left}")
    
    # Test bbox at center of original image
    orig_bbox = [0.5, 0.5, 0.2, 0.15]  # center, 20% width, 15% height
    print(f"\n   Original bbox (normalized to orig): {orig_bbox}")
    
    # Transform to letterbox space
    new_bbox = transform_bbox_letterbox(orig_bbox, info)
    print(f"   After letterbox transform: [{new_bbox[0]:.4f}, {new_bbox[1]:.4f}, {new_bbox[2]:.4f}, {new_bbox[3]:.4f}]")
    
    # Inverse transform back
    recovered_bbox = inverse_transform_bbox_letterbox(new_bbox, info)
    print(f"   After inverse transform:   [{recovered_bbox[0]:.4f}, {recovered_bbox[1]:.4f}, {recovered_bbox[2]:.4f}, {recovered_bbox[3]:.4f}]")
    
    # Check round-trip accuracy
    error = np.max(np.abs(np.array(orig_bbox) - np.array(recovered_bbox)))
    print(f"\n   Round-trip error: {error:.6f}")
    assert error < 1e-6, "Letterbox transform round-trip failed!"
    print("   [OK] Round-trip test passed!")
    
    # Test batch transform
    test_bboxes = [
        [0, 0.5, 0.5, 0.2, 0.15],   # class 0
        [1, 0.25, 0.75, 0.1, 0.1],  # class 1
    ]
    transformed = transform_bboxes_letterbox(test_bboxes, info)
    print(f"\n   Batch transform: {len(transformed)} bboxes processed")

    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    test_utils()
