"""
CuPy Object Detector Inference Script

Loads trained model and visualizes predictions on images.

Usage:
    python cupy_inference.py --checkpoint checkpoints_cupy/best_model.npz
    python cupy_inference.py --checkpoint checkpoints_cupy/best_model.npz --image path/to/image.jpg
"""

import os
import sys
import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cupy_detector.model import get_model, MODEL_CONFIGS
from cupy_detector.utils import letterbox_image
from detector.utils import decode_predictions, nms


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/pcb-component-detection-DatasetNinja')
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data/preprocessed')

CLASS_NAMES = ['Cap1', 'Resistor', 'Transformer']
CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#FFE66D']  # Red, Teal, Yellow

TARGET_SIZE = 224
NUM_CLASSES = 3


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path, model_size='large', num_classes=3):
    """Load model and weights from checkpoint."""
    print(f"Loading model: {model_size}")
    model = get_model(model_size, num_classes=num_classes)
    model.set_training(False)  # Inference mode
    
    print(f"Loading weights: {checkpoint_path}")
    weights = np.load(checkpoint_path)
    
    loaded_count = 0
    for i, layer in enumerate(model.all_layers):
        # Conv2d weights
        if hasattr(layer, 'W'):
            key_W = f'layer_{i}_W'
            if key_W in weights:
                layer.W = cp.asarray(weights[key_W])
                loaded_count += 1
            key_b = f'layer_{i}_b'
            if key_b in weights and layer.b is not None:
                layer.b = cp.asarray(weights[key_b])
        
        # BatchNorm weights
        if hasattr(layer, 'gamma'):
            key_gamma = f'layer_{i}_gamma'
            if key_gamma in weights:
                layer.gamma = cp.asarray(weights[key_gamma])
                layer.beta = cp.asarray(weights[f'layer_{i}_beta'])
                layer.running_mean = cp.asarray(weights[f'layer_{i}_running_mean'])
                layer.running_var = cp.asarray(weights[f'layer_{i}_running_var'])
                loaded_count += 1
        
        # SEBlock weights
        if hasattr(layer, 'fc1_W'):
            key_fc1 = f'layer_{i}_fc1_W'
            if key_fc1 in weights:
                layer.fc1_W = cp.asarray(weights[key_fc1])
                layer.fc2_W = cp.asarray(weights[f'layer_{i}_fc2_W'])
                loaded_count += 1
    
    print(f"Loaded {loaded_count} layer weights")
    print(f"Model parameters: {model.get_params_count():,}")
    
    return model


# =============================================================================
# INFERENCE
# =============================================================================

def preprocess_image(image_path, target_size=224):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32)
    original_size = img_np.shape[:2]  # (H, W)
    
    # Apply letterbox
    img_letterbox, info = letterbox_image(img_np, target_size=target_size)
    
    # Normalize and transpose to (C, H, W)
    img_letterbox = img_letterbox / 255.0
    img_letterbox = np.transpose(img_letterbox, (2, 0, 1))
    
    return img_np, img_letterbox, info


def run_inference(model, image, conf_thresh=0.3, iou_thresh=0.5):
    """Run inference on a single image."""
    # Add batch dimension and move to GPU
    if image.ndim == 3:
        image = image[np.newaxis, ...]
    
    image_gpu = cp.asarray(image)
    
    # Forward pass
    output = model.forward(image_gpu)
    
    # Move back to CPU
    output_np = cp.asnumpy(output)[0]  # Remove batch dimension
    
    # Decode predictions and apply NMS
    boxes = decode_predictions(output_np, conf_thresh=conf_thresh)
    boxes = nms(boxes, iou_thresh=iou_thresh)
    
    return boxes, output_np


def transform_boxes_to_original(boxes, info):
    """
    Transform boxes from letterbox (normalized) coordinates back to original image.
    
    Letterbox pipeline:
    1. Original (orig_h, orig_w) -> Pad to square (padded_size x padded_size) with padding
    2. Resize square to (target_size x target_size)
    3. Model outputs normalized [0,1] coords
    
    Inverse:
    1. Normalized [0,1] -> target_size pixels
    2. Scale up to padded_size pixels (divide by scale = target_size/padded_size)
    3. Subtract padding to get original coords
    """
    scale = info['scale']  # target_size / padded_size
    pad_top = info['pad_top']  # padding in padded_size space
    pad_left = info['pad_left']
    target_size = info['target_size']
    orig_h = info.get('orig_h', target_size)
    orig_w = info.get('orig_w', target_size)
    
    transformed = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        
        # Step 1: Normalized [0,1] -> target_size pixel coordinates
        x1_target = x1 * target_size
        y1_target = y1 * target_size
        x2_target = x2 * target_size
        y2_target = y2 * target_size
        
        # Step 2: Scale from target_size back to padded_size
        # scale = target_size / padded_size, so padded = target / scale
        x1_padded = x1_target / scale
        y1_padded = y1_target / scale
        x2_padded = x2_target / scale
        y2_padded = y2_target / scale
        
        # Step 3: Remove padding (padding was added to center the image)
        x1_orig = x1_padded - pad_left
        y1_orig = y1_padded - pad_top
        x2_orig = x2_padded - pad_left
        y2_orig = y2_padded - pad_top
        
        # Clip to original image bounds
        x1_orig = max(0, min(orig_w, x1_orig))
        y1_orig = max(0, min(orig_h, y1_orig))
        x2_orig = max(0, min(orig_w, x2_orig))
        y2_orig = max(0, min(orig_h, y2_orig))
        
        transformed.append([x1_orig, y1_orig, x2_orig, y2_orig, conf, int(cls)])
    
    return transformed


# =============================================================================
# VISUALIZATION
# =============================================================================

def boxes_to_pixels(boxes, image_size):
    """Convert normalized boxes [0,1] to pixel coordinates."""
    pixel_boxes = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        pixel_boxes.append([
            x1 * image_size, y1 * image_size,
            x2 * image_size, y2 * image_size,
            conf, int(cls)
        ])
    return pixel_boxes


def draw_boxes_on_ax(ax, boxes, image):
    """Draw bounding boxes on matplotlib axis."""
    ax.imshow(image.astype(np.uint8))
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        
        # Create rectangle
        width = x2 - x1
        height = y2 - y1
        color = CLASS_COLORS[int(cls_id)]
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{CLASS_NAMES[int(cls_id)]}: {conf:.2f}"
        ax.text(
            x1, max(y1 - 5, 10),
            label,
            color='white',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
        )


def visualize_predictions(letterbox_img, original_img, boxes_normalized, boxes_original, 
                          title="Predictions", save_path=None, show=True):
    """
    Visualize predictions on both letterbox and original image side by side.
    
    Args:
        letterbox_img: Letterbox image (C, H, W) or (H, W, C) numpy array, [0,1] or [0,255]
        original_img: Original image (H, W, 3) numpy array [0,255]
        boxes_normalized: List of [x1, y1, x2, y2, conf, class_id] in normalized [0,1] coords
        boxes_original: List of [x1, y1, x2, y2, conf, class_id] in original pixel coords
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the plot (set False for batch mode)
    """
    # Prepare letterbox image for display
    if letterbox_img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        letterbox_display = letterbox_img.transpose(1, 2, 0)
    else:
        letterbox_display = letterbox_img
    
    if letterbox_display.max() <= 1.0:
        letterbox_display = (letterbox_display * 255).astype(np.uint8)
    else:
        letterbox_display = letterbox_display.astype(np.uint8)
    
    letterbox_size = letterbox_display.shape[0]  # Assuming square
    
    # Convert normalized boxes to letterbox pixel coordinates
    boxes_letterbox_px = boxes_to_pixels(boxes_normalized, letterbox_size)
    
    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Letterbox image (what model sees)
    draw_boxes_on_ax(axes[0], boxes_letterbox_px, letterbox_display)
    axes[0].set_title(f"Letterbox Input ({letterbox_size}x{letterbox_size})", fontsize=12)
    axes[0].axis('off')
    
    # Right: Original image
    draw_boxes_on_ax(axes[1], boxes_original, original_img)
    axes[1].set_title(f"Original Image ({original_img.shape[1]}x{original_img.shape[0]})", fontsize=12)
    axes[1].axis('off')
    
    plt.suptitle(f"{title} - {len(boxes_normalized)} detections", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_grid_output(output, title="Model Output Grid"):
    """Visualize the raw grid output from the model."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    channel_names = ['X offset', 'Y offset', 'Width', 'Height', 
                     'Confidence', 'Cap1', 'Resistor', 'Transformer']
    
    for i, (ax, name) in enumerate(zip(axes.flat, channel_names)):
        im = ax.imshow(output[i], cmap='viridis' if i < 5 else 'hot')
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def get_random_validation_image():
    """Get a random image path from validation set."""
    # Always load from raw files to ensure original and letterbox match
    val_img_dir = os.path.join(DATA_DIR, 'validation', 'img')
    ann_dir = os.path.join(DATA_DIR, 'validation', 'ann')
    
    if os.path.exists(ann_dir):
        # Get annotation files (these have corresponding images)
        ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        if ann_files:
            idx = np.random.randint(0, len(ann_files))
            img_name = ann_files[idx].replace('.json', '')
            img_path = os.path.join(val_img_dir, img_name)
            if os.path.exists(img_path):
                print(f"Selected image: {img_name}")
                return img_path
    
    # Fallback: try image directory directly
    if os.path.exists(val_img_dir):
        img_files = [f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if img_files:
            idx = np.random.randint(0, len(img_files))
            img_path = os.path.join(val_img_dir, img_files[idx])
            print(f"Selected image: {img_files[idx]}")
            return img_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='CuPy Object Detector Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_cupy/best_model.npz',
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='large',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model size')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to specific image (optional, uses random val image if not provided)')
    parser.add_argument('--conf_thresh', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--show_grid', action='store_true',
                        help='Show raw grid output visualization')
    parser.add_argument('--save', type=str, default=None,
                        help='Save visualization to file')
    parser.add_argument('--free', action='store_true',
                        help='Run continuously until stopped (Ctrl+C). Saves to inference_result.png')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CuPy Object Detector Inference")
    print("=" * 60)
    
    # Load model
    model = load_model(args.checkpoint, model_size=args.model, num_classes=NUM_CLASSES)
    
    # Check if free mode
    if args.free:
        run_free_mode(model, args)
        return
    
    # Get image
    if args.image:
        # Load specific image
        print(f"\nLoading image: {args.image}")
        img_path = args.image
    else:
        # Use random validation image
        print("\nSelecting random validation image...")
        img_path = get_random_validation_image()
        
        if img_path is None:
            print("ERROR: No validation images found!")
            return
    
    # Load and preprocess the image (ensures original and letterbox match)
    original_img, preprocessed_img, info = preprocess_image(img_path, TARGET_SIZE)
    
    # Run inference
    print(f"\nRunning inference (conf_thresh={args.conf_thresh}, iou_thresh={args.iou_thresh})...")
    boxes, raw_output = run_inference(
        model, preprocessed_img, 
        conf_thresh=args.conf_thresh, 
        iou_thresh=args.iou_thresh
    )
    
    print(f"Detected {len(boxes)} objects")
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        print(f"  - {CLASS_NAMES[int(cls)]}: conf={conf:.3f}, box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
    
    # Transform boxes to original image coordinates
    boxes_original = transform_boxes_to_original(boxes, info)
    
    # Visualize both letterbox and original images side by side
    print("\nVisualizing predictions...")
    visualize_predictions(
        letterbox_img=preprocessed_img,
        original_img=original_img,
        boxes_normalized=boxes,
        boxes_original=boxes_original,
        title=f"CuPy Detector ({args.model})", 
        save_path=args.save,
        show=True
    )
    
    if args.show_grid:
        visualize_grid_output(raw_output, title="Raw Model Output")
    
    print("\nDone!")


def run_free_mode(model, args):
    """Run inference continuously until stopped. Saves to inference_result.png."""
    import time
    
    print(f"\n{'='*60}")
    print("FREE MODE - Running continuously (Ctrl+C to stop)")
    print("Saving to: inference_result.png")
    print(f"{'='*60}\n")
    
    # Get all validation images
    val_img_dir = os.path.join(DATA_DIR, 'validation', 'img')
    ann_dir = os.path.join(DATA_DIR, 'validation', 'ann')
    
    img_paths = []
    if os.path.exists(ann_dir):
        for f in os.listdir(ann_dir):
            if f.endswith('.json'):
                img_name = f.replace('.json', '')
                img_path = os.path.join(val_img_dir, img_name)
                if os.path.exists(img_path):
                    img_paths.append(img_path)
    
    if not img_paths:
        print("ERROR: No validation images found!")
        return
    
    print(f"Found {len(img_paths)} images\n")
    
    count = 0
    try:
        while True:
            # Pick random image
            img_path = img_paths[np.random.randint(0, len(img_paths))]
            img_name = os.path.basename(img_path)
            
            # Load and preprocess
            original_img, preprocessed_img, info = preprocess_image(img_path, TARGET_SIZE)
            
            # Run inference
            boxes, _ = run_inference(
                model, preprocessed_img,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh
            )
            
            # Transform boxes
            boxes_original = transform_boxes_to_original(boxes, info)
            
            # Save visualization to inference_result.png (overwrite each time)
            visualize_predictions(
                letterbox_img=preprocessed_img,
                original_img=original_img,
                boxes_normalized=boxes,
                boxes_original=boxes_original,
                title=f"CuPy Detector ({args.model})",
                save_path='inference_result.png',
                show=False
            )
            
            count += 1
            print(f"[{count}] {img_name}: {len(boxes)} detections")
            
            # Small delay
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\n\nStopped! Processed {count} images.")


if __name__ == "__main__":
    main()
