"""
Preprocess PCB Dataset for Fast Training

Preprocesses all images and saves as PyTorch tensors (.pt files).
This eliminates the slow letterboxing/resizing during training.

Usage:
    python preprocess_dataset.py

Output:
    data/preprocessed/train/    - preprocessed training data
    data/preprocessed/val/      - preprocessed validation data
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector.utils import letterbox_image, transform_bboxes_letterbox, create_target

# Config
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/pcb-component-detection-DatasetNinja')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data/preprocessed')
TARGET_SIZE = 224
GRID_SIZE = 7
NUM_CLASSES = 3
PAD_VALUE = 128
NUM_WORKERS = 8  # For parallel processing

CLASS_MAPPING = {
    'Cap1': 0,
    'Resistor': 1,
    'Transformer': 2,
}


def parse_annotation(ann_path):
    """Parse Supervisely JSON annotation."""
    bboxes = []
    
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return bboxes
    
    img_w = data.get('size', {}).get('width', 1)
    img_h = data.get('size', {}).get('height', 1)
    
    for obj in data.get('objects', []):
        class_name = obj.get('classTitle', '')
        
        if class_name not in CLASS_MAPPING:
            continue
        
        class_id = CLASS_MAPPING[class_name]
        
        exterior = obj.get('points', {}).get('exterior', [])
        if len(exterior) != 2:
            continue
        
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h
        
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)
        
        bboxes.append([class_id, x_center, y_center, width, height])
    
    return bboxes


def process_single_image(args):
    """Process a single image and return tensors."""
    img_path, ann_path, bboxes = args
    
    try:
        # Load and process image (as uint8 first, more memory efficient)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        
        # Letterbox preprocessing
        img_letterbox, info = letterbox_image(
            img.astype(np.float32), 
            target_size=TARGET_SIZE,
            pad_value=PAD_VALUE
        )
        
        # Transform bboxes
        bboxes_letterbox = transform_bboxes_letterbox(bboxes, info)
        
        # Create target grid
        target = create_target(
            bboxes_letterbox,
            num_classes=NUM_CLASSES,
            grid_size=GRID_SIZE
        )
        
        # Normalize and convert to tensor format
        img_letterbox = img_letterbox / 255.0
        img_letterbox = np.transpose(img_letterbox, (2, 0, 1))  # (3, H, W)
        
        # Convert to half precision to save space (16-bit instead of 32-bit)
        image_tensor = torch.from_numpy(img_letterbox.astype(np.float16))
        target_tensor = torch.from_numpy(target.astype(np.float16))
        
        return image_tensor, target_tensor, os.path.basename(img_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None, None


def get_samples(split):
    """Get list of samples for a split."""
    img_dir = os.path.join(DATA_DIR, split, 'img')
    ann_dir = os.path.join(DATA_DIR, split, 'ann')
    
    samples = []
    
    if not os.path.exists(ann_dir):
        return samples
    
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
    
    for ann_file in ann_files:
        img_name = ann_file.replace('.json', '')
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        ann_path = os.path.join(ann_dir, ann_file)
        bboxes = parse_annotation(ann_path)
        
        if len(bboxes) > 0:
            samples.append((img_path, ann_path, bboxes))
    
    return samples


def preprocess_split(split):
    """Preprocess all samples in a split."""
    print(f"\n{'='*60}")
    print(f"Preprocessing {split} split...")
    print(f"{'='*60}")
    
    samples = get_samples(split)
    print(f"Found {len(samples)} samples")
    
    if len(samples) == 0:
        return
    
    # Create output directory
    output_split = 'val' if split == 'validation' else split
    output_dir = os.path.join(OUTPUT_DIR, output_split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in parallel
    all_images = []
    all_targets = []
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_image, sample): i 
                   for i, sample in enumerate(samples)}
        
        for future in tqdm(as_completed(futures), total=len(samples), desc=f"Processing {split}"):
            image_tensor, target_tensor, name = future.result()
            if image_tensor is not None:
                all_images.append(image_tensor)
                all_targets.append(target_tensor)
    
    # Stack all tensors and save as single file
    print(f"Stacking {len(all_images)} samples...")
    images_tensor = torch.stack(all_images)
    targets_tensor = torch.stack(all_targets)
    
    # Save
    output_file = os.path.join(output_dir, 'data.pt')
    torch.save({
        'images': images_tensor,
        'targets': targets_tensor,
        'num_samples': len(all_images)
    }, output_file)
    
    print(f"Saved to: {output_file}")
    print(f"Images shape: {images_tensor.shape}")
    print(f"Targets shape: {targets_tensor.shape}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")


def main():
    print("="*60)
    print("PCB Dataset Preprocessor")
    print("="*60)
    print(f"Input:  {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process both splits
    preprocess_split('train')
    preprocess_split('validation')
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
