"""
CuPy Detector Utilities - GPU Accelerated

Includes:
- Letterbox preprocessing (NumPy - runs on CPU, images stay on CPU for I/O)
- Bbox transformations
- Target grid creation
- Prediction decoding
- NMS
"""

import numpy as np
import cupy as cp


# =============================================================================
# IMAGE PREPROCESSING - LETTERBOX (CPU - for I/O efficiency)
# =============================================================================

def letterbox_image(image, target_size=224, pad_value=128):
    """
    Letterbox: Pad image to square then resize.
    Maintains aspect ratio.
    
    Note: This runs on CPU since image I/O is CPU-bound.
    """
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    orig_h, orig_w, channels = image.shape
    max_side = max(orig_h, orig_w)
    
    pad_top = (max_side - orig_h) // 2
    pad_bottom = max_side - orig_h - pad_top
    pad_left = (max_side - orig_w) // 2
    pad_right = max_side - orig_w - pad_left
    
    padded = np.full((max_side, max_side, channels), pad_value, dtype=image.dtype)
    padded[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w, :] = image
    
    scale = target_size / max_side
    resized = _resize_image(padded, target_size, target_size)
    
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
    """Resize using bilinear interpolation (NumPy)."""
    orig_h, orig_w, channels = image.shape
    
    x_ratio = orig_w / new_w
    y_ratio = orig_h / new_h
    
    x = np.arange(new_w) * x_ratio
    y = np.arange(new_h) * y_ratio
    
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, orig_w - 1)
    y1 = np.minimum(y0 + 1, orig_h - 1)
    
    wx = x - x0
    wy = y - y0
    
    resized = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    
    for c in range(channels):
        for i in range(new_h):
            for j in range(new_w):
                v00 = image[y0[i], x0[j], c]
                v01 = image[y0[i], x1[j], c]
                v10 = image[y1[i], x0[j], c]
                v11 = image[y1[i], x1[j], c]
                
                v = (1 - wy[i]) * ((1 - wx[j]) * v00 + wx[j] * v01) + \
                    wy[i] * ((1 - wx[j]) * v10 + wx[j] * v11)
                resized[i, j, c] = v
    
    return resized


def transform_bbox_letterbox(bbox, info):
    """Transform bbox to letterbox space."""
    x, y, w, h = bbox
    
    orig_h = info['orig_h']
    orig_w = info['orig_w']
    pad_top = info['pad_top']
    pad_left = info['pad_left']
    padded_size = info['padded_size']
    
    x_pixel = x * orig_w
    y_pixel = y * orig_h
    w_pixel = w * orig_w
    h_pixel = h * orig_h
    
    x_padded = x_pixel + pad_left
    y_padded = y_pixel + pad_top
    
    x_new = x_padded / padded_size
    y_new = y_padded / padded_size
    w_new = w_pixel / padded_size
    h_new = h_pixel / padded_size
    
    return [x_new, y_new, w_new, h_new]


def transform_bboxes_letterbox(bboxes, info):
    """Transform multiple bboxes."""
    new_bboxes = []
    for bbox in bboxes:
        class_id = bbox[0]
        coords = bbox[1:5]
        new_coords = transform_bbox_letterbox(coords, info)
        new_bboxes.append([class_id] + new_coords)
    return new_bboxes


# =============================================================================
# TARGET CREATION (NumPy for CPU-side preprocessing)
# =============================================================================

def create_target(annotations, num_classes=3, grid_size=7):
    """
    Create ground truth grid from annotations.
    
    Returns:
        target: (8, 7, 7) NumPy array
    """
    target = np.zeros((5 + num_classes, grid_size, grid_size), dtype=np.float32)
    
    for ann in annotations:
        class_id, x, y, w, h = ann
        class_id = int(class_id)
        
        grid_x = int(x * grid_size)
        grid_y = int(y * grid_size)
        
        grid_x = min(grid_size - 1, max(0, grid_x))
        grid_y = min(grid_size - 1, max(0, grid_y))
        
        x_offset = x * grid_size - grid_x
        y_offset = y * grid_size - grid_y
        
        if target[4, grid_y, grid_x] == 0:
            target[0, grid_y, grid_x] = x_offset
            target[1, grid_y, grid_x] = y_offset
            target[2, grid_y, grid_x] = w
            target[3, grid_y, grid_x] = h
            target[4, grid_y, grid_x] = 1.0
            target[5 + class_id, grid_y, grid_x] = 1.0
    
    return target


# =============================================================================
# PREDICTION DECODING (GPU accelerated)
# =============================================================================

def decode_predictions_gpu(output, conf_thresh=0.5, grid_size=7):
    """
    Decode grid output to bounding boxes on GPU.
    
    Args:
        output: (8, 7, 7) - CuPy array
        conf_thresh: Confidence threshold
        
    Returns:
        boxes: List of [x1, y1, x2, y2, confidence, class_id] (NumPy)
    """
    # Move to CPU for box extraction (small data)
    output_np = cp.asnumpy(output) if isinstance(output, cp.ndarray) else output
    
    boxes = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            conf = output_np[4, i, j]
            
            if conf < conf_thresh:
                continue
            
            x = (j + output_np[0, i, j]) / grid_size
            y = (i + output_np[1, i, j]) / grid_size
            w = output_np[2, i, j]
            h = output_np[3, i, j]
            
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
            
            class_probs = output_np[5:, i, j]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            final_conf = conf * class_conf
            
            if final_conf >= conf_thresh:
                boxes.append([x1, y1, x2, y2, final_conf, class_id])
    
    return boxes


def nms(boxes, iou_thresh=0.5):
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []
    
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        
        remaining = []
        for box in boxes:
            if box[5] != best[5]:
                remaining.append(box)
            else:
                iou = compute_iou(best[:4], box[:4])
                if iou < iou_thresh:
                    remaining.append(box)
        
        boxes = remaining
    
    return kept


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
        
    return inter_area / union_area
