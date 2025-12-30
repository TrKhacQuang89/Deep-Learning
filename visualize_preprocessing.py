"""
Dataset Preprocessing Visualization Script

Visualizes the preprocessing pipeline to debug potential issues:
1. Load original image and annotations
2. Draw original bboxes on original image
3. Apply letterbox preprocessing
4. Draw transformed bboxes on letterboxed image
5. Save side-by-side comparison
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from detector.utils import (
    letterbox_image,
    transform_bboxes_letterbox,
    create_target,
    decode_predictions
)
from detector.dataset import CLASS_MAPPING


# Colors for each class
COLORS = {
    0: (255, 0, 0),     # Cap1 - Red
    1: (0, 255, 0),     # Resistor - Green  
    2: (0, 0, 255),     # Transformer - Blue
}

CLASS_NAMES = {v: k for k, v in CLASS_MAPPING.items()}


def draw_bbox_on_image(image, bboxes, title="", normalized=True):
    """
    Draw bounding boxes on image.
    
    Args:
        image: PIL Image or numpy array
        bboxes: List of [class_id, x_center, y_center, width, height]
        title: Title to draw
        normalized: Whether bbox coords are normalized (0-1)
    
    Returns:
        PIL Image with bboxes drawn
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8))
    else:
        img = image.copy()
    
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    for bbox in bboxes:
        class_id = int(bbox[0])
        x_c, y_c, bw, bh = bbox[1:5]
        
        if normalized:
            x_c *= w
            y_c *= h
            bw *= w
            bh *= h
        
        # Convert center to corners
        x1 = x_c - bw / 2
        y1 = y_c - bh / 2
        x2 = x_c + bw / 2
        y2 = y_c + bh / 2
        
        color = COLORS.get(class_id, (255, 255, 0))
        class_name = CLASS_NAMES.get(class_id, f"cls_{class_id}")
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{class_name}"
        draw.text((x1, y1 - 15), label, fill=color)
    
    # Draw title
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))
    
    return img


def draw_grid_target(target, grid_size=7, cell_size=32):
    """
    Visualize the target grid tensor.
    
    Args:
        target: (8, 7, 7) numpy array
        grid_size: Size of grid
        cell_size: Size of each cell in pixels
    
    Returns:
        PIL Image showing the grid
    """
    img_size = grid_size * cell_size
    img = Image.new('RGB', (img_size, img_size), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Draw grid lines
    for i in range(grid_size + 1):
        pos = i * cell_size
        draw.line([(pos, 0), (pos, img_size)], fill=(100, 100, 100), width=1)
        draw.line([(0, pos), (img_size, pos)], fill=(100, 100, 100), width=1)
    
    # Draw cells with objects
    for i in range(grid_size):
        for j in range(grid_size):
            conf = target[4, i, j]
            if conf > 0:
                # Cell has object
                x_off = target[0, i, j]
                y_off = target[1, i, j]
                w = target[2, i, j]
                h = target[3, i, j]
                
                # Get class
                class_probs = target[5:, i, j]
                class_id = np.argmax(class_probs)
                color = COLORS.get(class_id, (255, 255, 0))
                
                # Highlight cell
                cx = j * cell_size
                cy = i * cell_size
                draw.rectangle([cx+1, cy+1, cx+cell_size-1, cy+cell_size-1], 
                              outline=color, width=2)
                
                # Draw center point
                center_x = (j + x_off) * cell_size
                center_y = (i + y_off) * cell_size
                r = 3
                draw.ellipse([center_x-r, center_y-r, center_x+r, center_y+r], 
                            fill=color)
                
                # Draw bbox (scaled to grid)
                box_w = w * img_size
                box_h = h * img_size
                x1 = center_x - box_w / 2
                y1 = center_y - box_h / 2
                x2 = center_x + box_w / 2
                y2 = center_y + box_h / 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
                
                # Draw class info
                draw.text((cx + 2, cy + 2), f"{class_id}", fill=color)
    
    return img


def parse_annotation(ann_path, class_mapping):
    """Parse Supervisely JSON annotation."""
    bboxes = []
    
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_w = data.get('size', {}).get('width', 1)
    img_h = data.get('size', {}).get('height', 1)
    
    for obj in data.get('objects', []):
        class_name = obj.get('classTitle', '')
        
        if class_name not in class_mapping:
            continue
        
        class_id = class_mapping[class_name]
        
        exterior = obj.get('points', {}).get('exterior', [])
        if len(exterior) != 2:
            continue
        
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        
        # Convert to center format and normalize
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h
        
        bboxes.append([class_id, x_center, y_center, width, height])
    
    return bboxes, img_w, img_h


def visualize_sample(data_dir, split='train', sample_idx=0, output_path='visualization_output.png', target_size=224):
    """
    Visualize preprocessing for a specific sample.
    
    Args:
        data_dir: Path to dataset root
        split: 'train' or 'validation'
        sample_idx: Index of sample to visualize
        output_path: Path to save output image
        target_size: Target size for letterbox
    """
    img_dir = os.path.join(data_dir, split, 'img')
    ann_dir = os.path.join(data_dir, split, 'ann')
    
    # Get list of valid samples
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
    
    valid_samples = []
    for ann_file in ann_files:
        img_name = ann_file.replace('.json', '')
        img_path = os.path.join(img_dir, img_name)
        ann_path = os.path.join(ann_dir, ann_file)
        
        if os.path.exists(img_path):
            bboxes, _, _ = parse_annotation(ann_path, CLASS_MAPPING)
            if len(bboxes) > 0:
                valid_samples.append({
                    'img_path': img_path,
                    'ann_path': ann_path,
                    'img_name': img_name
                })
    
    if sample_idx >= len(valid_samples):
        print(f"Sample index {sample_idx} out of range. Max: {len(valid_samples)-1}")
        return
    
    sample = valid_samples[sample_idx]
    print(f"\n{'='*60}")
    print(f"Visualizing sample {sample_idx}: {sample['img_name']}")
    print(f"{'='*60}")
    
    # 1. Load original image
    orig_img = Image.open(sample['img_path']).convert('RGB')
    orig_np = np.array(orig_img, dtype=np.float32)
    print(f"Original image size: {orig_img.size[0]} x {orig_img.size[1]}")
    
    # 2. Parse annotations
    bboxes, img_w, img_h = parse_annotation(sample['ann_path'], CLASS_MAPPING)
    print(f"Found {len(bboxes)} objects:")
    for i, bbox in enumerate(bboxes):
        class_name = CLASS_NAMES.get(int(bbox[0]), f"cls_{bbox[0]}")
        print(f"  [{i}] {class_name}: center=({bbox[1]:.4f}, {bbox[2]:.4f}), size=({bbox[3]:.4f}, {bbox[4]:.4f})")
    
    # 3. Draw original image with bboxes
    orig_with_boxes = draw_bbox_on_image(
        orig_img, bboxes, 
        title=f"Original: {img_w}x{img_h}", 
        normalized=True
    )
    
    # 4. Apply letterbox preprocessing
    letterbox_np, info = letterbox_image(orig_np, target_size=target_size, pad_value=128)
    print(f"\nLetterbox info:")
    print(f"  Original: {info['orig_w']}x{info['orig_h']}")
    print(f"  Padded to: {info['padded_size']}x{info['padded_size']}")
    print(f"  Padding: top={info['pad_top']}, left={info['pad_left']}")
    print(f"  Scale: {info['scale']:.4f}")
    
    # 5. Transform bboxes
    bboxes_letterbox = transform_bboxes_letterbox(bboxes, info)
    print(f"\nTransformed bboxes:")
    for i, bbox in enumerate(bboxes_letterbox):
        class_name = CLASS_NAMES.get(int(bbox[0]), f"cls_{bbox[0]}")
        print(f"  [{i}] {class_name}: center=({bbox[1]:.4f}, {bbox[2]:.4f}), size=({bbox[3]:.4f}, {bbox[4]:.4f})")
    
    # 6. Draw letterboxed image with transformed bboxes
    letterbox_with_boxes = draw_bbox_on_image(
        letterbox_np, bboxes_letterbox,
        title=f"Letterbox: {target_size}x{target_size}",
        normalized=True
    )
    
    # 7. Create target grid
    target = create_target(bboxes_letterbox, num_classes=3, grid_size=7)
    print(f"\nTarget grid:")
    print(f"  Shape: {target.shape}")
    print(f"  Objects in grid: {int(target[4].sum())}")
    
    # Find which cells have objects
    for i in range(7):
        for j in range(7):
            if target[4, i, j] > 0:
                class_id = np.argmax(target[5:, i, j])
                class_name = CLASS_NAMES.get(class_id, f"cls_{class_id}")
                print(f"  Cell ({j}, {i}): {class_name}, offset=({target[0,i,j]:.3f}, {target[1,i,j]:.3f}), size=({target[2,i,j]:.3f}, {target[3,i,j]:.3f})")
    
    # 8. Draw target grid visualization
    grid_vis = draw_grid_target(target, grid_size=7, cell_size=32)
    
    # 9. Create comparison image
    # Resize original to comparable size
    orig_resized = orig_with_boxes.resize((target_size, target_size), Image.LANCZOS)
    
    # Create side-by-side comparison
    gap = 10
    grid_width = 7 * 32  # 224
    total_width = target_size + gap + target_size + gap + grid_width
    total_height = target_size
    
    comparison = Image.new('RGB', (total_width, total_height), (30, 30, 30))
    comparison.paste(orig_resized, (0, 0))
    comparison.paste(letterbox_with_boxes, (target_size + gap, 0))
    comparison.paste(grid_vis, (target_size + gap + target_size + gap, 0))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    draw.text((10, target_size - 20), "Original", fill=(255, 255, 255))
    draw.text((target_size + gap + 10, target_size - 20), "After Letterbox", fill=(255, 255, 255))
    draw.text((target_size + gap + target_size + gap + 10, target_size - 20), "Target Grid", fill=(255, 255, 255))
    
    # Save
    comparison.save(output_path)
    print(f"\n[OK] Saved visualization to: {output_path}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Visualize dataset preprocessing')
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'data', 'pcb-component-detection-DatasetNinja'),
                        help='Path to dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation'],
                        help='Dataset split')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--output', type=str, default='preprocessing_visualization.png',
                        help='Output image path')
    parser.add_argument('--target_size', type=int, default=224, help='Target size for letterbox')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    data_dir = os.path.abspath(args.data_dir)
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Dataset not found at: {data_dir}")
        return
    
    print(f"Dataset: {data_dir}")
    print(f"Split: {args.split}")
    
    for i in range(args.num_samples):
        idx = args.sample + i
        if args.num_samples > 1:
            output = args.output.replace('.png', f'_{idx}.png')
        else:
            output = args.output
        
        visualize_sample(
            data_dir=data_dir,
            split=args.split,
            sample_idx=idx,
            output_path=output,
            target_size=args.target_size
        )


if __name__ == "__main__":
    main()
